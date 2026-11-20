#include "bounded_interval.hpp"

#include <gtest/gtest.h>
#include <cstdlib>
#ifdef __SSE2__
#include <xmmintrin.h>
#endif

namespace embodik::detail {
namespace {

TEST(BoundedInterval, EnvironmentAndInvalidInputs) {
  EXPECT_NO_THROW(validate_interval_environment());
  EXPECT_THROW((void)Interval(2.0, 1.0), std::domain_error);
  EXPECT_THROW((void)Interval(NAN), std::domain_error);
  EXPECT_THROW((void)Interval(INFINITY), std::domain_error);
  EXPECT_THROW(Interval(1.0) / Interval(-1.0, 1.0), std::domain_error);
  EXPECT_THROW(sqrt(Interval(-1.0, 1.0)), std::domain_error);
  EXPECT_THROW(sin_cos(Interval(-4.1, 0.0)), std::domain_error);
  EXPECT_THROW(Interval(1e300) * Interval(1e300), std::domain_error);
}

TEST(BoundedInterval, RejectsUnsupportedRoundingWithoutChangingIt) {
  const int saved = std::fegetround();
  struct Restore {
    int mode;
    ~Restore() { std::fesetround(mode); }
  } restore{saved};
  ASSERT_EQ(std::fesetround(FE_UPWARD), 0);
  EXPECT_THROW(validate_interval_environment(), std::runtime_error);
  EXPECT_EQ(std::fegetround(), FE_UPWARD);
}

TEST(BoundedInterval, ScopedEnvironmentPreservesCallerFlags) {
  std::fenv_t saved;
  ASSERT_EQ(std::fegetenv(&saved), 0);
  struct Restore {
    std::fenv_t environment;
    ~Restore() { std::fesetenv(&environment); }
  } restore{saved};
  std::feraiseexcept(FE_INEXACT);
  {
    const ScopedIntervalEnvironment environment;
    EXPECT_EQ(std::fetestexcept(FE_ALL_EXCEPT), 0);
    EXPECT_THROW(Interval(1e300) * Interval(1e300), std::domain_error);
  }
  EXPECT_NE(std::fetestexcept(FE_INEXACT), 0);
}

#if defined(__GLIBC__) && defined(__USE_GNU)
TEST(BoundedInterval, ScopedEnvironmentMasksCallerTrapsDuringOwnedArithmetic) {
  EXPECT_EXIT({
    feenableexcept(FE_OVERFLOW);
    bool rejected = false;
    {
      const ScopedIntervalEnvironment environment;
      try { (void)(Interval(1e300) * Interval(1e300)); }
      catch (const std::domain_error &) { rejected = true; }
    }
    std::_Exit(rejected && (fegetexcept() & FE_OVERFLOW) ? 0 : 1);
  }, ::testing::ExitedWithCode(0), "");
}
#endif

TEST(BoundedInterval, ExactIdentitiesPreserveStructuralZeros) {
  const Interval x(-0.25, 0.5);
  EXPECT_EQ((Interval(0.0) + x).lower(), x.lower());
  EXPECT_EQ((Interval(1.0) * x).upper(), x.upper());
  EXPECT_TRUE((Interval(0.0) * x).point(0.0));
  EXPECT_TRUE((Interval(1.0) - Interval(1.0)).point(0.0));
  const auto trig = sin_cos(Interval(0.0));
  EXPECT_TRUE(trig.first.point(0.0));
  EXPECT_TRUE(trig.second.point(1.0));
}

#ifdef __SSE2__
TEST(BoundedInterval, RejectsFlushToZeroAndDenormalsAreZeroWithoutChangingThem) {
  const unsigned saved = _mm_getcsr();
  struct Restore {
    unsigned mode;
    ~Restore() { _mm_setcsr(mode); }
  } restore{saved};
  _mm_setcsr(saved | 0x8040U);
  EXPECT_THROW(validate_interval_environment(), std::runtime_error);
  EXPECT_EQ(_mm_getcsr() & 0x8040U, 0x8040U);
}
#endif

TEST(BoundedInterval, SignedProductsDivisionAndInteriorSquares) {
  const auto product = Interval(-2.0, 3.0) * Interval(-4.0, 5.0);
  EXPECT_LE(product.lower(), -12.0);
  EXPECT_GE(product.upper(), 15.0);
  const auto quotient = Interval(1.0, 2.0) / Interval(-4.0, -2.0);
  EXPECT_LE(quotient.lower(), -1.0);
  EXPECT_GE(quotient.upper(), -0.25);
  const auto squared = square(Interval(-2.0, 3.0));
  EXPECT_EQ(squared.lower(), 0.0);
  EXPECT_GE(squared.upper(), 9.0);
}

TEST(BoundedInterval, UnderflowIsEnclosedRatherThanTreatedAsExactZero) {
  const auto product = Interval(1e-300) * Interval(1e-100);
  EXPECT_EQ(product.lower(), 0.0);
  EXPECT_GT(product.upper(), 0.0);
  const auto negative = Interval(-1e-300) * Interval(1e-100);
  EXPECT_LT(negative.lower(), 0.0);
  EXPECT_EQ(negative.upper(), 0.0);
}

TEST(BoundedInterval, BoundedTrigonometryContainsSamplesAndStaysTightAtPoints) {
  for (double center : {-4.0, -3.1, -1.0, -0.01, 0.01, 1.0, 3.1, 4.0}) {
    const auto trig = sin_cos(Interval(center));
    EXPECT_LE(trig.first.lower(), std::sin(center));
    EXPECT_GE(trig.first.upper(), std::sin(center));
    EXPECT_LE(trig.second.lower(), std::cos(center));
    EXPECT_GE(trig.second.upper(), std::cos(center));
    EXPECT_LT(trig.first.upper() - trig.first.lower(), 1e-10);
    EXPECT_LT(trig.second.upper() - trig.second.lower(), 1e-10);
  }
  const auto wide = sin_cos(Interval(-4.0, 4.0));
  EXPECT_EQ(wide.first.lower(), -1.0);
  EXPECT_EQ(wide.first.upper(), 1.0);
  EXPECT_EQ(wide.second.lower(), -1.0);
  EXPECT_EQ(wide.second.upper(), 1.0);
}

} // namespace
} // namespace embodik::detail
