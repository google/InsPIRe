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

#ifndef CRYPTO_MATRIX_H_
#define CRYPTO_MATRIX_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace proto {
class Matrix;
}  // namespace proto

// Represents a 2D matrix containing elements of type `CoeffType`.
//
// Supports basic operations such as matrix-vector multiplication. The elements
// are stored in row-major order internally.
template <typename CoeffType>
class Matrix {
 public:
  // Creates a Matrix from a 2D vector of data.
  //
  // Returns an InvalidArgumentError if the rows in the input data have
  // varying lengths (i.e., if the data is a jagged 2D array).
  // If the input data is empty, creates an empty 0x0 Matrix.
  static absl::StatusOr<Matrix> Create(
      std::vector<std::vector<CoeffType>> data);

  // Creates a default initialized Matrix of the specified dimensions.
  //
  // Returns an InvalidArgumentError if `rows` or `cols` is negative.
  static absl::StatusOr<Matrix> Create(int rows, int cols);

  // Creates a condensed Matrix from a 2D vector of original uncondensed data.
  //
  // This method is only supported when `CoeffType` is `uint8_t`, `uint16_t`, or
  // `int32_t`.
  static absl::StatusOr<Matrix> CreateCondensed(
      std::vector<std::vector<CoeffType>> data);

  // Creates a condensed Matrix from an uncondensed Matrix.
  //
  // This method is only supported when `CoeffType` is `uint8_t`, `uint16_t`, or
  // `int32_t`.
  static absl::StatusOr<Matrix> CreateCondensed(const Matrix& normal_matrix);

  // Vertically concatenates multiple matrices.
  //
  // Returns an InvalidArgumentError if the input vector is empty, or if the
  // matrices have different number of columns or condensed states.
  static absl::StatusOr<Matrix> VerticalConcatenate(
      const std::vector<Matrix>& matrices);

  Matrix() = default;

  Matrix(const Matrix& m) = default;
  Matrix& operator=(const Matrix& that) = default;

  Matrix(Matrix&& m) = default;
  Matrix& operator=(Matrix&& that) = default;

  // Multiplies the matrix by a given column vector.
  //
  // During computation, the elements of the matrix and the input vector are
  // cast to `OutputType`.
  //
  // Returns an error if the size of the input vector `vec` does not match
  // the number of columns in this matrix.
  template <typename OutputType, typename VecType>
  absl::StatusOr<std::vector<OutputType>> Multiply(
      const std::vector<VecType>& vec) const;

  // Returns the number of rows in the matrix.
  int Rows() const { return rows_; }

  // Returns the number of columns in the matrix.
  int Cols() const { return original_cols_; }

  // Returns whether the matrix is stored in condensed form.
  bool IsCondensed() const { return is_condensed_; }

  // Adds another matrix to this matrix in place.
  //
  // Returns an InvalidArgumentError if the dimensions of `that` matrix do not
  // match the dimensions of this matrix.
  absl::Status AddInPlace(const Matrix& that);

  // Extracts a submatrix starting at (row_offset, col_offset) with dimensions
  // `num_rows` and `num_cols`.
  //
  // Returns an InvalidArgumentError if the requested submatrix is out of
  // bounds.
  absl::StatusOr<Matrix> SubMatrix(int row_offset, int col_offset, int num_rows,
                                   int num_cols) const;

  // Returns the underlying 1D array representation of the matrix.
  // The matrix elements are stored in row-major order.
  const std::vector<CoeffType>& Data() const { return data_; }

  // Returns the underlying condensed 1D array representation.
  const std::vector<uint64_t>& CondensedData() const { return condensed_data_; }

  absl::StatusOr<proto::Matrix> ToProto() const;

  static absl::StatusOr<Matrix> FromProto(const proto::Matrix& proto);

  int PhysicalCols() const { return cols_; }
  int OriginalCols() const { return original_cols_; }
  const std::vector<CoeffType>& GetCondensedData() const { return data_; }

 public:
  explicit Matrix(std::vector<CoeffType> data, int rows, int cols,
                  bool is_condensed = false, int original_cols = 0)
      : data_(std::move(data)),
        rows_(rows),
        cols_(cols),
        is_condensed_(is_condensed),
        original_cols_(original_cols == 0 ? cols : original_cols) {}

  explicit Matrix(std::vector<CoeffType> data,
                  std::vector<uint64_t> condensed_data, int rows, int cols,
                  bool is_condensed, int original_cols)
      : data_(std::move(data)),
        condensed_data_(std::move(condensed_data)),
        rows_(rows),
        cols_(cols),
        is_condensed_(is_condensed),
        original_cols_(original_cols) {}

  absl::StatusOr<std::vector<uint64_t>> CondensedMultiply(
      const std::vector<uint64_t>& vec) const;

  std::vector<CoeffType> data_;
  std::vector<uint64_t> condensed_data_;
  int rows_ = 0;
  int cols_ = 0;
  bool is_condensed_ = false;
  int original_cols_ = 0;
};

extern template class Matrix<uint8_t>;
extern template class Matrix<uint16_t>;
extern template class Matrix<uint32_t>;
extern template class Matrix<uint64_t>;
extern template class Matrix<int16_t>;
extern template class Matrix<int32_t>;

template <typename CoeffType>
template <typename OutputType, typename VecType>
absl::StatusOr<std::vector<OutputType>> Matrix<CoeffType>::Multiply(
    const std::vector<VecType>& vec) const {
  if (vec.size() != original_cols_) {
    return absl::InvalidArgumentError(
        "Vector size must match the number of matrix columns.");
  }
  if (is_condensed_) {
    std::vector<uint64_t> vec_u64(vec.begin(), vec.end());
    auto result_u64_or = CondensedMultiply(vec_u64);
    if (!result_u64_or.ok()) {
      return result_u64_or.status();
    }
    return std::vector<OutputType>(result_u64_or->begin(),
                                   result_u64_or->end());
  }

  std::vector<OutputType> result(rows_, 0);
  const CoeffType* __restrict m_ptr = data_.data();
  const VecType* __restrict v_ptr = vec.data();
  OutputType* __restrict res_ptr = result.data();
  for (size_t i = 0; i < rows_; ++i) {
    for (size_t j = 0; j < cols_; ++j) {
      size_t flat_idx = i * static_cast<size_t>(cols_) + j;
      res_ptr[i] += static_cast<OutputType>(m_ptr[flat_idx]) *
                    static_cast<OutputType>(v_ptr[j]);
    }
  }
  return result;
}
}  // namespace v2

}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_MATRIX_H_
