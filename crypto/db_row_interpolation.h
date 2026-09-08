// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CRYPTO_DB_ROW_INTERPOLATION_H_
#define CRYPTO_DB_ROW_INTERPOLATION_H_

#include <vector>

#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// Interprets given `db_row` as `t` polynomial evaluations over the plaintext
// ring, and interpolates into `t` polynomial coefficients.
//
// Let d = ring_degree, and t = interpolation_degree.
// The input vector `db_row` must have length d * t.
// Each chunk of length d represents a polynomial in Z_p[X] / (X^d + 1),
// where p = `plaintext_modulus` is odd. Let these evaluated polynomials be y_0,
// ..., y_{t-1}. The evaluations points are \omega^0, \omega^1, ...,
// \omega^{t-1}, where \omega = X^{2d/t}. Note that \omega^t = X^{2d} = 1 in the
// ring, so \omega is a primitive t-th root of unity.
//
// The output is a vector of length d * t representing the t polynomials
// c_0(X), ..., c_{t-1}(X) such that P(\omega^j) = y_j, where
// P(Z) = \sum c_i(X) * Z^i.
//
// The implementation performs a Cooley-Tukey Inverse Fast Fourier Transform.
//
// Supports uint16_t and uint32_t types. For uint32_t type, modulus must be at
// most 2^30 - 1.
template <typename T>
absl::StatusOr<std::vector<T>> Interpolate(const std::vector<T>& db_row,
                                           int ring_degree,
                                           int interpolation_degree,
                                           T plaintext_modulus);

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_DB_ROW_INTERPOLATION_H_
