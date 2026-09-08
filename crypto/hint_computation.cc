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

#include "crypto/hint_computation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "crypto/matrix.h"
#include "crypto/polynomial.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "crypto/status_macros.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename OutputType, typename S, typename T>
absl::StatusOr<Matrix<OutputType>> MulMatNegacyclic(const Matrix<S>& mat,
                                                    const Polynomial<T>& poly,
                                                    FftContext& ctx) {
  if (mat.Rows() == 0) {
    return absl::InvalidArgumentError("Matrix cannot be empty.");
  }
  if (mat.Cols() != poly.Len()) {
    return absl::InvalidArgumentError(
        "Matrix columns must match polynomial length.");
  }

  const size_t rows = mat.Rows();
  const size_t d = mat.Cols();

  std::vector<std::vector<OutputType>> result_data;
  result_data.reserve(rows);

  const std::vector<S>& mat_data = mat.Data();

  std::vector<OutputType> poly_coeffs_out(d);
  poly_coeffs_out[0] = static_cast<OutputType>(poly.Coeffs()[0]);
  for (size_t j = 1; j < d; ++j) {
    poly_coeffs_out[j] = static_cast<OutputType>(-poly.Coeffs()[d - j]);
  }
  ASSIGN_OR_RETURN(auto poly_out,
                   Polynomial<OutputType>::Create(std::move(poly_coeffs_out)));

  for (size_t i = 0; i < rows; ++i) {
    std::vector<OutputType> row_poly_coeffs(d);
    for (size_t j = 0; j < d; ++j) {
      row_poly_coeffs[j] = static_cast<OutputType>(mat_data[i * d + j]);
    }

    ASSIGN_OR_RETURN(auto row_poly, Polynomial<OutputType>::Create(
                                        std::move(row_poly_coeffs)));
    ASSIGN_OR_RETURN(auto mult_res, row_poly.MultFft(poly_out, ctx));

    result_data.push_back(mult_res.Coeffs());
  }

  return Matrix<OutputType>::Create(std::move(result_data));
}

template <typename OutputType, typename S, typename T>
absl::StatusOr<Matrix<OutputType>> ComputeHintMatrix(
    const Matrix<S>& database, const std::vector<Polynomial<T>>& a_components,
    FftContext& ctx) {
  if (database.Rows() == 0) {
    return absl::InvalidArgumentError("Database matrix cannot be empty.");
  }
  if (a_components.empty()) {
    return absl::InvalidArgumentError("a_components cannot be empty.");
  }

  const size_t n = a_components.size();
  const size_t d = a_components[0].Len();
  const size_t rows = database.Rows();
  const size_t cols = database.Cols();
  if (cols != d * n) {
    return absl::InvalidArgumentError(
        "Database columns must match d * n where d is the polynomial length "
        "and n is the number of polynomials.");
  }

  const int chunk_bits = 20;
  // The precision of a double-precision floating-point format is 53 bits.
  // We use this to compute how many polynomials we can safely process together
  // without losing precision in the lower bits.
  constexpr double kDoublePrecisionBits = 53.0;
  double bits_left = kDoublePrecisionBits - (std::log2(d) + 2.0 * chunk_bits);
  size_t safe_n =
      (bits_left >= 0.0) ? static_cast<size_t>(std::pow(2.0, bits_left)) : 1;
  if (safe_n < 1) safe_n = 1;

  std::vector<Polynomial<OutputType>> v_inverted;
  v_inverted.reserve(n);
  for (size_t k = 0; k < n; ++k) {
    std::vector<OutputType> poly_out(d);
    poly_out[0] = static_cast<OutputType>(a_components[k].Coeffs()[0]);
    for (size_t j = 1; j < d; ++j) {
      poly_out[j] = static_cast<OutputType>(-a_components[k].Coeffs()[d - j]);
    }
    ASSIGN_OR_RETURN(auto p,
                     Polynomial<OutputType>::Create(std::move(poly_out)));
    v_inverted.push_back(std::move(p));
  }

  std::vector<std::vector<OutputType>> result_data(
      rows, std::vector<OutputType>(d, 0));
  const std::vector<S>& mat_data = database.Data();

  for (size_t i = 0; i < rows; ++i) {
    size_t row_offset = i * cols;

    for (size_t batch_start = 0; batch_start < n; batch_start += safe_n) {
      size_t batch_size = std::min(safe_n, n - batch_start);
      std::vector<Polynomial<OutputType>> u_batch;
      u_batch.reserve(batch_size);
      std::vector<Polynomial<OutputType>> v_batch;
      v_batch.reserve(batch_size);

      for (size_t b = 0; b < batch_size; ++b) {
        size_t k = batch_start + b;
        std::vector<OutputType> block_coeffs(d);
        size_t block_offset = row_offset + k * d;
        for (size_t j = 0; j < d; ++j) {
          block_coeffs[j] = static_cast<OutputType>(mat_data[block_offset + j]);
        }
        ASSIGN_OR_RETURN(
            auto p, Polynomial<OutputType>::Create(std::move(block_coeffs)));
        u_batch.push_back(std::move(p));
        // We push a copy of v_inverted
        v_batch.push_back(v_inverted[k]);
      }

      ASSIGN_OR_RETURN(auto mult_res, Polynomial<OutputType>::InnerProductFft(
                                          u_batch, v_batch, ctx, 8 * sizeof(S),
                                          8 * sizeof(T), 20));
      for (size_t j = 0; j < d; ++j) {
        result_data[i][j] += mult_res.Coeffs()[j];
      }
    }
  }

  return Matrix<OutputType>::Create(std::move(result_data));
}

template absl::StatusOr<Matrix<uint32_t>>
MulMatNegacyclic<uint32_t, uint16_t, uint32_t>(const Matrix<uint16_t>& mat,
                                               const Polynomial<uint32_t>& poly,
                                               FftContext& ctx);
template absl::StatusOr<Matrix<uint32_t>>
MulMatNegacyclic<uint32_t, uint8_t, uint32_t>(const Matrix<uint8_t>& mat,
                                              const Polynomial<uint32_t>& poly,
                                              FftContext& ctx);
template absl::StatusOr<Matrix<uint64_t>>
MulMatNegacyclic<uint64_t, uint16_t, uint64_t>(const Matrix<uint16_t>& mat,
                                               const Polynomial<uint64_t>& poly,
                                               FftContext& ctx);
template absl::StatusOr<Matrix<uint64_t>>
MulMatNegacyclic<uint64_t, uint8_t, uint64_t>(const Matrix<uint8_t>& mat,
                                              const Polynomial<uint64_t>& poly,
                                              FftContext& ctx);
template absl::StatusOr<Matrix<uint64_t>>
MulMatNegacyclic<uint64_t, uint32_t, uint64_t>(const Matrix<uint32_t>& mat,
                                               const Polynomial<uint64_t>& poly,
                                               FftContext& ctx);

template absl::StatusOr<Matrix<uint32_t>>
ComputeHintMatrix<uint32_t, uint16_t, uint32_t>(
    const Matrix<uint16_t>& database,
    const std::vector<Polynomial<uint32_t>>& a_components, FftContext& ctx);
template absl::StatusOr<Matrix<uint32_t>>
ComputeHintMatrix<uint32_t, uint8_t, uint32_t>(
    const Matrix<uint8_t>& database,
    const std::vector<Polynomial<uint32_t>>& a_components, FftContext& ctx);
template absl::StatusOr<Matrix<uint64_t>>
ComputeHintMatrix<uint64_t, uint16_t, uint64_t>(
    const Matrix<uint16_t>& database,
    const std::vector<Polynomial<uint64_t>>& a_components, FftContext& ctx);
template absl::StatusOr<Matrix<uint64_t>>
ComputeHintMatrix<uint64_t, uint8_t, uint64_t>(
    const Matrix<uint8_t>& database,
    const std::vector<Polynomial<uint64_t>>& a_components, FftContext& ctx);
template absl::StatusOr<Matrix<uint64_t>>
ComputeHintMatrix<uint64_t, uint32_t, uint64_t>(
    const Matrix<uint32_t>& database,
    const std::vector<Polynomial<uint64_t>>& a_components, FftContext& ctx);

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
