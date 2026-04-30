/*
 * Copyright 2025-2026 Andy Park <andypark.purdue@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <Eigen/Core>
#include <limits>
#include <utility>

namespace embodik::constraints_utils {

inline constexpr std::pair<double, double> FindScaleFactor(double lower, double upper, double a) {
    double upper_scale = 1.0;
    double lower_scale = 0.0;
    if (1e-10 < std::fabs(a) && std::fabs(a) < 1e10) {
        if (a < 0.0 && lower < 0.0 && a <= upper) {
            upper_scale = lower / a;
            lower_scale = upper / a;
        } else if (a > 0.0 && upper > 0.0 && a >= lower) {
            upper_scale = upper / a;
            lower_scale = lower / a;
        }
    }
    return {std::min(1.0, upper_scale), std::max(0.0, lower_scale)};
}

inline double ComputeMinMaxScale(const Eigen::VectorXd& lower_margin,
                                 const Eigen::VectorXd& upper_margin,
                                 const Eigen::VectorXd& a_vec,
                                 const Eigen::MatrixXd& active_selections,
                                 Eigen::Index* min_index_out) {
    const Eigen::Index rows = a_vec.size();
    Eigen::VectorXd max_scale(rows);
    for (Eigen::Index j = 0; j < rows; ++j) {
        if (active_selections(j, j) == 1.0) {
            max_scale(j) = std::numeric_limits<double>::infinity();
        } else {
            auto [upper_s, lower_s] = FindScaleFactor(lower_margin(j), upper_margin(j), a_vec(j));
            (void)lower_s;  // only upper scale used here
            max_scale(j) = upper_s;
        }
    }
    return max_scale.minCoeff(min_index_out);
}

}  // namespace embodik::constraints_utils

