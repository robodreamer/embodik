#pragma once

// Source-private enclosure arithmetic for bounded offline/path validation.
// Create ScopedIntervalEnvironment at each outer validation entry. This
// code must not be compiled with reassociation/fast-math or run with FTZ/DAZ.
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Bounded interval validation requires IEEE arithmetic without fast-math"
#endif

#include <algorithm>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace embodik::detail {

inline void validate_interval_environment() {
  volatile double normal = std::numeric_limits<double>::min();
  volatile double tiny = std::numeric_limits<double>::denorm_min();
  volatile double half = 0.5;
  volatile double one = 1.0;
  if (!std::numeric_limits<double>::is_iec559 ||
      std::numeric_limits<double>::digits != 53 || FLT_EVAL_METHOD != 0 ||
      std::fegetround() != FE_TONEAREST || normal * half == 0.0 ||
      tiny * one == 0.0) {
    throw std::runtime_error("interval validation requires nearest IEEE arithmetic and subnormals");
  }
}

class ScopedIntervalEnvironment {
public:
  ScopedIntervalEnvironment() {
    if (std::fegetenv(&previous_) != 0)
      throw std::runtime_error("cannot save interval floating-point environment");
    std::fenv_t ignored;
    if (std::feholdexcept(&ignored) != 0) {
      std::fesetenv(&previous_);
      throw std::runtime_error("cannot mask interval floating-point traps");
    }
    try {
      validate_interval_environment();
    } catch (...) {
      std::fesetenv(&previous_);
      throw;
    }
  }
  ~ScopedIntervalEnvironment() { std::fesetenv(&previous_); }
  ScopedIntervalEnvironment(const ScopedIntervalEnvironment &) = delete;
  ScopedIntervalEnvironment &operator=(const ScopedIntervalEnvironment &) = delete;
private:
  std::fenv_t previous_;
};

class Interval {
public:
  explicit Interval(double value = 0.0) : Interval(value, value) {}
  Interval(double lower, double upper) : lower_(lower), upper_(upper) {
    if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper)
      throw std::domain_error("finite ordered interval required");
  }
  double lower() const { return lower_; }
  double upper() const { return upper_; }
  bool point(double value) const { return lower_ == value && upper_ == value; }

  static double downward(double value) {
    return outward(value, -std::numeric_limits<double>::infinity());
  }
  static double upward(double value) {
    return outward(value, std::numeric_limits<double>::infinity());
  }

  friend Interval operator-(const Interval &a) {
    return Interval(-a.upper_, -a.lower_);
  }
  friend Interval operator+(const Interval &a, const Interval &b) {
    // Adding zero is exact, including subnormal endpoints. Preserve structural
    // zeros rather than introducing artificial negative norm-squared bounds.
    const auto sum = [](double x, double y, bool upper) {
      if (x == 0.0) return y;
      if (y == 0.0) return x;
      return upper ? upward(x + y) : downward(x + y);
    };
    return Interval(sum(a.lower_, b.lower_, false), sum(a.upper_, b.upper_, true));
  }
  friend Interval operator-(const Interval &a, const Interval &b) {
    if (a.lower_ == a.upper_ && b.point(a.lower_)) return Interval(0.0);
    return a + (-b);
  }
  friend Interval operator*(const Interval &a, const Interval &b) {
    if (a.point(0.0) || b.point(0.0)) return Interval(0.0);
    if (a.point(1.0)) return b;
    if (b.point(1.0)) return a;
    const double products[] = {a.lower_ * b.lower_, a.lower_ * b.upper_,
                               a.upper_ * b.lower_, a.upper_ * b.upper_};
    for (double value : products)
      if (!std::isfinite(value)) throw std::domain_error("interval product overflow");
    const auto limits = std::minmax_element(std::begin(products), std::end(products));
    double lower = downward(*limits.first), upper = upward(*limits.second);
    if ((a.lower_ >= 0.0 && b.lower_ >= 0.0) ||
        (a.upper_ <= 0.0 && b.upper_ <= 0.0)) lower = std::max(0.0, lower);
    if ((a.lower_ >= 0.0 && b.upper_ <= 0.0) ||
        (a.upper_ <= 0.0 && b.lower_ >= 0.0)) upper = std::min(0.0, upper);
    return Interval(lower, upper);
  }
  friend Interval operator/(const Interval &a, const Interval &b) {
    if (b.lower_ <= 0.0 && b.upper_ >= 0.0)
      throw std::domain_error("interval divisor includes zero");
    if (b.point(1.0)) return a;
    return a * Interval(downward(1.0 / b.upper_), upward(1.0 / b.lower_));
  }

private:
  static double outward(double value, double direction) {
    if (!std::isfinite(value)) throw std::domain_error("interval arithmetic overflow");
    const double rounded = std::nextafter(value, direction);
    if (!std::isfinite(rounded)) throw std::domain_error("interval enclosure overflow");
    return rounded;
  }
  double lower_, upper_;
};

inline Interval square(const Interval &x) {
  if (x.point(0.0)) return Interval(0.0);
  const double a = x.lower() * x.lower(), b = x.upper() * x.upper();
  const double lower = x.lower() <= 0.0 && x.upper() >= 0.0
      ? 0.0 : std::max(0.0, Interval::downward(std::min(a, b)));
  return Interval(lower, Interval::upward(std::max(a, b)));
}

inline Interval sqrt(const Interval &x) {
  if (x.lower() < 0.0) throw std::domain_error("nonnegative square-root interval required");
  if (x.point(0.0)) return Interval(0.0);
  return Interval(x.lower() == 0.0 ? 0.0 : Interval::downward(std::sqrt(x.lower())),
                  Interval::upward(std::sqrt(x.upper())));
}

inline Interval unit_range(const Interval &x) {
  return Interval(std::max(-1.0, x.lower()), std::min(1.0, x.upper()));
}

namespace interval_internal {

inline std::pair<Interval, Interval> taylor_sin_cos(const Interval &x) {
  if (x.point(0.0)) return {Interval(0.0), Interval(1.0)};
  const Interval minus_square = -square(x);
  Interval sine = x, sine_term = x;
  Interval cosine(1.0), cosine_term(1.0);
  for (int k = 1; k <= 24; ++k) {
    cosine_term = cosine_term * minus_square / Interval((2 * k - 1) * 2 * k);
    cosine = cosine + cosine_term;
    if (k < 24) {
      sine_term = sine_term * minus_square / Interval(2 * k * (2 * k + 1));
      sine = sine + sine_term;
    }
  }
  // Both are Taylor polynomials through degree 48 (the sine coefficient at
  // degree 48 is zero). All derivatives have magnitude <=1, giving the common
  // Lagrange remainder |x|^49/49!, enclosed by the same outward arithmetic.
  const double radius = std::max(std::abs(x.lower()), std::abs(x.upper()));
  Interval remainder(1.0);
  for (int k = 1; k <= 49; ++k)
    remainder = remainder * Interval(radius) / Interval(k);
  const Interval error(-remainder.upper(), remainder.upper());
  return {unit_range(sine + error), unit_range(cosine + error)};
}

} // namespace interval_internal

inline std::pair<Interval, Interval> sin_cos(const Interval &x) {
  if (x.lower() < -4.0 || x.upper() > 4.0)
    throw std::domain_error("interval trigonometry supports angles within [-4,4]");
  if (x.point(0.0)) return {Interval(0.0), Interval(1.0)};
  const double midpoint = std::clamp(0.5 * x.lower() + 0.5 * x.upper(), x.lower(), x.upper());
  const auto center = interval_internal::taylor_sin_cos(Interval(midpoint));
  const auto delta = interval_internal::taylor_sin_cos(x - Interval(midpoint));
  return {unit_range(center.first * delta.second + center.second * delta.first),
          unit_range(center.second * delta.second - center.first * delta.first)};
}

} // namespace embodik::detail
