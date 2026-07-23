#pragma once

#include <Eigen/Core>

#include <string>
#include <utility>
#include <vector>

namespace embodik::detail {

enum class GeneralizedConstraintBoundSide { kBoth, kLower, kUpper };

struct GeneralizedConstraintSource {
  std::string family;
  std::string key;
  int component = -1;
  GeneralizedConstraintBoundSide side =
      GeneralizedConstraintBoundSide::kBoth;
};

struct GeneralizedConstraintBlock {
#ifdef EMBODIK_ENABLE_TEST_HOOKS
  std::string block_id;
#endif
  Eigen::MatrixXd coefficient_matrix;
  // Physical affine row contract:
  //   physical_lower <= coefficient_matrix*u + affine_bias <= physical_upper.
  Eigen::VectorXd affine_bias;
  Eigen::VectorXd physical_lower_bounds;
  Eigen::VectorXd physical_upper_bounds;
  // Empty means hard rows with a factor of one.
  Eigen::VectorXd max_softening_factors;
  // Empty means no source records were supplied for this block.
  std::vector<std::vector<GeneralizedConstraintSource>> row_sources;
};

#ifdef EMBODIK_ENABLE_TEST_HOOKS
struct GeneralizedConstraintBlockSpan {
  std::string block_id;
  Eigen::Index row_begin = 0;
  Eigen::Index row_count = 0;
};
#endif

// Derivative-neutral storage for lower <= C*u + b <= upper. The set validates
// structural shape and applies affine-bias shifting. Numerical sanitization,
// softening policy, and recovery remain caller responsibilities.
class GeneralizedConstraintSet {
public:
  GeneralizedConstraintSet(Eigen::Index variable_count,
                           Eigen::Index row_capacity,
                           bool track_sources = false)
      : variable_count_(variable_count), row_capacity_(row_capacity),
        coefficient_matrix_(0, variable_count >= 0 ? variable_count : 0),
        track_sources_(track_sources),
        storage_initialized_(variable_count >= 0 && row_capacity == 0) {}

  bool append_block(const GeneralizedConstraintBlock &block) {
    if (!can_append(block)) {
      return false;
    }
    const Eigen::Index rows = block.coefficient_matrix.rows();
    if (rows == 0) {
      return true;
    }
    initialize_storage();

    coefficient_matrix_.middleRows(row_count_, rows) =
        block.coefficient_matrix;
    lower_bounds_.segment(row_count_, rows) =
        block.physical_lower_bounds - block.affine_bias;
    upper_bounds_.segment(row_count_, rows) =
        block.physical_upper_bounds - block.affine_bias;
    if (block.max_softening_factors.size() == rows) {
      max_softening_factors_.segment(row_count_, rows) =
          block.max_softening_factors;
    }
    if (track_sources_ && !block.row_sources.empty()) {
      for (Eigen::Index row = 0; row < rows; ++row) {
        row_sources_[static_cast<std::size_t>(row_count_ + row)] =
            block.row_sources[static_cast<std::size_t>(row)];
      }
    }
#ifdef EMBODIK_ENABLE_TEST_HOOKS
    if (track_sources_) {
      block_spans_.push_back({block.block_id, row_count_, rows});
    }
#endif
    row_count_ += rows;
    return true;
  }

  bool append_block(GeneralizedConstraintBlock &&block) {
    if (!can_append(block)) {
      return false;
    }
    const Eigen::Index rows = block.coefficient_matrix.rows();
    if (rows == 0) {
      return true;
    }
    if (row_count_ != 0 || rows != row_capacity_) {
      return append_block(
          static_cast<const GeneralizedConstraintBlock &>(block));
    }

    coefficient_matrix_ = std::move(block.coefficient_matrix);
    lower_bounds_ =
        std::move(block.physical_lower_bounds) - block.affine_bias;
    upper_bounds_ =
        std::move(block.physical_upper_bounds) - block.affine_bias;
    if (block.max_softening_factors.size() == rows) {
      max_softening_factors_ = std::move(block.max_softening_factors);
    } else {
      max_softening_factors_ = Eigen::VectorXd::Ones(rows);
    }
    if (track_sources_ && !block.row_sources.empty()) {
      row_sources_ = std::move(block.row_sources);
    } else if (track_sources_) {
      row_sources_.resize(static_cast<std::size_t>(row_capacity_));
    }
#ifdef EMBODIK_ENABLE_TEST_HOOKS
    if (track_sources_) {
      block_spans_.push_back({std::move(block.block_id), 0, rows});
    }
#endif
    storage_initialized_ = true;
    row_count_ = rows;
    return true;
  }

  bool add_row_source(Eigen::Index row, GeneralizedConstraintSource source) {
    if (!track_sources_ || row < 0 || row >= row_count_) {
      return false;
    }
    row_sources_[static_cast<std::size_t>(row)].push_back(std::move(source));
    return true;
  }

  bool finalize() {
    return row_count_ == row_capacity_ && has_consistent_dimensions();
  }

  bool has_consistent_dimensions() const {
    return variable_count_ >= 0 && row_capacity_ >= 0 && row_count_ >= 0 &&
           row_count_ <= row_capacity_ && storage_initialized_ &&
           coefficient_matrix_.rows() == row_capacity_ &&
           coefficient_matrix_.cols() == variable_count_ &&
           lower_bounds_.size() == row_capacity_ &&
           upper_bounds_.size() == row_capacity_ &&
           max_softening_factors_.size() == row_capacity_ &&
           has_consistent_metadata();
  }

  Eigen::Index variable_count() const { return variable_count_; }
  Eigen::Index row_count() const { return row_count_; }
  Eigen::Index row_capacity() const { return row_capacity_; }
  bool tracks_sources() const { return track_sources_; }

  const Eigen::MatrixXd &coefficient_matrix() const {
    return coefficient_matrix_;
  }
  const Eigen::VectorXd &lower_bounds() const { return lower_bounds_; }
  const Eigen::VectorXd &upper_bounds() const { return upper_bounds_; }
  const Eigen::VectorXd &max_softening_factors() const {
    return max_softening_factors_;
  }
  const std::vector<std::vector<GeneralizedConstraintSource>> &
  row_sources() const {
    return row_sources_;
  }
#ifdef EMBODIK_ENABLE_TEST_HOOKS
  const std::vector<GeneralizedConstraintBlockSpan> &block_spans() const {
    return block_spans_;
  }
#endif

private:
  bool has_consistent_metadata() const {
    const bool sources_are_consistent =
        (!track_sources_ && row_sources_.empty()) ||
        (track_sources_ &&
         row_sources_.size() == static_cast<std::size_t>(row_capacity_));
    if (!sources_are_consistent) {
      return false;
    }
#ifdef EMBODIK_ENABLE_TEST_HOOKS
    return (!track_sources_ && block_spans_.empty()) ||
           (track_sources_ && block_spans_cover_rows());
#else
    return true;
#endif
  }

#ifdef EMBODIK_ENABLE_TEST_HOOKS
  bool block_spans_cover_rows() const {
    Eigen::Index next_row = 0;
    for (const auto &span : block_spans_) {
      if (span.row_begin != next_row || span.row_count <= 0) {
        return false;
      }
      next_row += span.row_count;
    }
    return next_row == row_count_;
  }
#endif

  void initialize_storage() {
    if (storage_initialized_) {
      return;
    }
    coefficient_matrix_ =
        Eigen::MatrixXd::Zero(row_capacity_, variable_count_);
    lower_bounds_ = Eigen::VectorXd::Zero(row_capacity_);
    upper_bounds_ = Eigen::VectorXd::Zero(row_capacity_);
    max_softening_factors_ = Eigen::VectorXd::Ones(row_capacity_);
    if (track_sources_) {
      row_sources_.resize(static_cast<std::size_t>(row_capacity_));
    }
    storage_initialized_ = true;
  }

  bool can_append(const GeneralizedConstraintBlock &block) const {
    const Eigen::Index rows = block.coefficient_matrix.rows();
    return variable_count_ >= 0 && row_capacity_ >= 0 && rows >= 0 &&
           block.coefficient_matrix.cols() == variable_count_ &&
           block.physical_lower_bounds.size() == rows &&
           block.physical_upper_bounds.size() == rows &&
           block.affine_bias.size() == rows &&
           (block.max_softening_factors.size() == 0 ||
            block.max_softening_factors.size() == rows) &&
           (block.row_sources.empty() ||
            block.row_sources.size() == static_cast<std::size_t>(rows)) &&
           row_count_ <= row_capacity_ - rows;
  }

  Eigen::Index variable_count_ = 0;
  Eigen::Index row_capacity_ = 0;
  Eigen::Index row_count_ = 0;
  Eigen::MatrixXd coefficient_matrix_;
  Eigen::VectorXd lower_bounds_;
  Eigen::VectorXd upper_bounds_;
  Eigen::VectorXd max_softening_factors_;
  bool track_sources_ = false;
  bool storage_initialized_ = false;
  std::vector<std::vector<GeneralizedConstraintSource>> row_sources_;
#ifdef EMBODIK_ENABLE_TEST_HOOKS
  std::vector<GeneralizedConstraintBlockSpan> block_spans_;
#endif
};

} // namespace embodik::detail
