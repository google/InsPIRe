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

#include "crypto/matrix.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "crypto/proto/pir_server.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "hwy/highway.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::Create(
    std::vector<std::vector<CoeffType>> data) {
  const int rows = data.size();
  if (rows == 0) {
    return Matrix<CoeffType>(std::vector<CoeffType>(), 0, 0);
  }
  const int cols = data[0].size();
  std::vector<CoeffType> flat_data;
  flat_data.reserve(static_cast<size_t>(rows) * cols);
  for (int i = 0; i < rows; ++i) {
    if (data[i].size() != cols) {
      return absl::InvalidArgumentError(
          "All rows must have the same number of columns.");
    }
    flat_data.insert(flat_data.end(), data[i].begin(), data[i].end());
  }
  return Matrix<CoeffType>(std::move(flat_data), rows, cols);
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::Create(int rows,
                                                            int cols) {
  if (rows < 0 || cols < 0) {
    return absl::InvalidArgumentError(
        "Matrix dimensions must be non-negative.");
  }
  return Matrix<CoeffType>(
      std::vector<CoeffType>(static_cast<size_t>(rows) * cols, 0), rows, cols);
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::CreateCondensed(
    std::vector<std::vector<CoeffType>> data) {
  auto normal_matrix_or = Create(std::move(data));
  if (!normal_matrix_or.ok()) {
    return normal_matrix_or.status();
  }
  return CreateCondensed(*normal_matrix_or);
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::CreateCondensed(
    const Matrix<CoeffType>& normal_matrix) {
  if (normal_matrix.IsCondensed()) {
    return absl::InvalidArgumentError(
        "Cannot condense a matrix that is already condensed.");
  }

  const int rows = normal_matrix.Rows();
  const int cols = normal_matrix.Cols();
  const std::vector<CoeffType>& data = normal_matrix.Data();

  std::vector<uint64_t> condensed_flat;

  if constexpr (std::is_same_v<CoeffType, uint8_t>) {
    if (cols % 8 != 0) {
      return absl::InvalidArgumentError(
          "Columns must be a multiple of 8 to condense uint8_t.");
    }
    condensed_flat.resize(static_cast<size_t>(rows) * (cols / 8), 0);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols / 8; ++j) {
        uint64_t condensed_val = 0;
        for (size_t k = 0; k < 8; ++k) {
          condensed_val |= static_cast<uint64_t>(
                               data[i * static_cast<size_t>(cols) + j * 8 + k])
                           << (k * 8);
        }
        condensed_flat[i * static_cast<size_t>(cols / 8) + j] = condensed_val;
      }
    }
  } else if constexpr (std::is_same_v<CoeffType, uint16_t>) {
    if (cols % 4 != 0) {
      return absl::InvalidArgumentError(
          "Columns must be a multiple of 4 to condense uint16_t.");
    }
    condensed_flat.resize(static_cast<size_t>(rows) * (cols / 4), 0);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols / 4; ++j) {
        uint64_t condensed_val = 0;
        for (size_t k = 0; k < 4; ++k) {
          condensed_val |= static_cast<uint64_t>(
                               data[i * static_cast<size_t>(cols) + j * 4 + k])
                           << (k * 16);
        }
        condensed_flat[i * static_cast<size_t>(cols / 4) + j] = condensed_val;
      }
    }
  } else if constexpr (std::is_same_v<CoeffType, int32_t>) {
    if (cols % 2 != 0) {
      return absl::InvalidArgumentError(
          "Columns must be a multiple of 2 to condense int32_t.");
    }
    condensed_flat.resize(static_cast<size_t>(rows) * (cols / 2), 0);
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols / 2; ++j) {
        uint64_t condensed_val = 0;
        for (size_t k = 0; k < 2; ++k) {
          uint64_t val = static_cast<uint64_t>(static_cast<uint32_t>(
              data[i * static_cast<size_t>(cols) + j * 2 + k]));
          condensed_val |= (val & 0xFFFFFFFFULL) << (k * 32);
        }
        condensed_flat[i * static_cast<size_t>(cols / 2) + j] = condensed_val;
      }
    }
  } else {
    return absl::FailedPreconditionError(
        "CreateCondensed is only supported for uint8_t, uint16_t, or int32_t "
        "matrices.");
  }

  int physical_cols = 0;
  if constexpr (std::is_same_v<CoeffType, uint8_t>) {
    physical_cols = cols / 8;
  } else if constexpr (std::is_same_v<CoeffType, uint16_t>) {
    physical_cols = cols / 4;
  } else if constexpr (std::is_same_v<CoeffType, int32_t>) {
    physical_cols = cols / 2;
  }

  Matrix<CoeffType> matrix(std::vector<CoeffType>(), rows, physical_cols,
                           /*is_condensed=*/true, /*original_cols=*/cols);
  matrix.condensed_data_ = std::move(condensed_flat);
  return matrix;
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::VerticalConcatenate(
    const std::vector<Matrix<CoeffType>>& matrices) {
  if (matrices.empty()) {
    return Matrix<CoeffType>();
  }
  const int cols = matrices[0].Cols();
  const bool is_condensed = matrices[0].IsCondensed();
  const int physical_cols = matrices[0].cols_;
  int total_rows = 0;

  for (const auto& m : matrices) {
    if (m.Cols() != cols) {
      return absl::InvalidArgumentError(
          "All matrices must have the same number of columns.");
    }
    if (m.IsCondensed() != is_condensed) {
      return absl::InvalidArgumentError(
          "All matrices must have the same condensed state.");
    }
    total_rows += m.Rows();
  }

  std::vector<CoeffType> combined_data;
  std::vector<uint64_t> combined_condensed_data;

  for (const auto& m : matrices) {
    combined_data.insert(combined_data.end(), m.Data().begin(), m.Data().end());
    if (is_condensed) {
      combined_condensed_data.insert(combined_condensed_data.end(),
                                     m.CondensedData().begin(),
                                     m.CondensedData().end());
    }
  }

  if (is_condensed) {
    Matrix<CoeffType> matrix(std::vector<CoeffType>(), total_rows,
                             physical_cols, /*is_condensed=*/true,
                             /*original_cols=*/cols);
    matrix.condensed_data_ = std::move(combined_condensed_data);
    return matrix;
  } else {
    return Matrix<CoeffType>(std::move(combined_data), total_rows, cols);
  }
}

template <typename CoeffType>
absl::Status Matrix<CoeffType>::AddInPlace(const Matrix& that) {
  if (is_condensed_) {
    return absl::FailedPreconditionError(
        "AddInPlace is not supported on a condensed matrix.");
  }
  if (rows_ != that.Rows() || cols_ != that.Cols()) {
    return absl::InvalidArgumentError(
        "Matrix dimensions must match for addition.");
  }
  const std::vector<CoeffType>& that_data = that.Data();
  for (int i = 0; i < data_.size(); ++i) {
    data_[i] += that_data[i];
  }
  return absl::OkStatus();
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::SubMatrix(
    int row_offset, int col_offset, int num_rows, int num_cols) const {
  if (is_condensed_) {
    return absl::FailedPreconditionError(
        "SubMatrix is not supported on a condensed matrix.");
  }
  if (row_offset < 0 || col_offset < 0 || num_rows < 0 || num_cols < 0) {
    return absl::InvalidArgumentError(
        "Offsets and dimensions must be non-negative.");
  }
  if (row_offset + num_rows > rows_ || col_offset + num_cols > cols_) {
    return absl::InvalidArgumentError("Submatrix is out of bounds.");
  }
  std::vector<CoeffType> sub_data(static_cast<size_t>(num_rows) * num_cols);
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      size_t flat_idx =
          (static_cast<size_t>(row_offset) + i) * cols_ + (col_offset + j);
      sub_data[i * static_cast<size_t>(num_cols) + j] = data_[flat_idx];
    }
  }
  return Matrix<CoeffType>(std::move(sub_data), num_rows, num_cols);
}

namespace {

template <typename CoeffType>
absl::StatusOr<std::vector<uint64_t>> CondensedMultiply(
    const Matrix<CoeffType>& condensed_matrix, absl::Span<const uint64_t> vec,
    int original_cols) {
  return absl::FailedPreconditionError(
      "CondensedMultiply is only supported for uint8_t, uint16_t, and "
      "int32_t matrices.");
}

template <>
absl::StatusOr<std::vector<uint64_t>> CondensedMultiply<uint16_t>(
    const Matrix<uint16_t>& condensed_matrix, absl::Span<const uint64_t> vec,
    int original_cols) {
  namespace hn = hwy::HWY_NAMESPACE;
  hn::ScalableTag<uint64_t> d;
  const size_t N = hn::Lanes(d);

  if (original_cols % (4 * N) != 0) {
    return absl::InvalidArgumentError(
        "Original columns must be a multiple of 4 * Lanes.");
  }
  if (vec.size() != original_cols) {
    return absl::InvalidArgumentError("Vector size mismatch.");
  }
  int rows = condensed_matrix.Rows();
  int condensed_cols = original_cols / 4;

  std::vector<uint64_t> result(rows, 0);
  const std::vector<uint64_t>& data = condensed_matrix.CondensedData();

  int num_blocks = original_cols / (4 * N);
  std::vector<hn::Vec<decltype(d)>> V_0(num_blocks);
  std::vector<hn::Vec<decltype(d)>> V_1(num_blocks);
  std::vector<hn::Vec<decltype(d)>> V_2(num_blocks);
  std::vector<hn::Vec<decltype(d)>> V_3(num_blocks);

  for (int b = 0; b < num_blocks; ++b) {
    int base = b * 4 * N;
    hn::LoadInterleaved4(d, &vec[base], V_0[b], V_1[b], V_2[b], V_3[b]);
  }

  auto mask = hn::Set(d, 0xFFFF);

  for (int i = 0; i < rows; ++i) {
    auto accum = hn::Zero(d);

    for (int b = 0; b < num_blocks; ++b) {
      auto m_packed = hn::LoadU(d, &data[i * condensed_cols + b * N]);

      auto m0 = hn::And(m_packed, mask);
      accum = hn::MulAdd(m0, V_0[b], accum);

      auto m1 = hn::And(hn::ShiftRight<16>(m_packed), mask);
      accum = hn::MulAdd(m1, V_1[b], accum);

      auto m2 = hn::And(hn::ShiftRight<32>(m_packed), mask);
      accum = hn::MulAdd(m2, V_2[b], accum);

      auto m3 = hn::ShiftRight<48>(m_packed);
      accum = hn::MulAdd(m3, V_3[b], accum);
    }

    result[i] = hn::ReduceSum(d, accum);
  }

  return result;
}

template <>
absl::StatusOr<std::vector<uint64_t>> CondensedMultiply<int32_t>(
    const Matrix<int32_t>& condensed_matrix, absl::Span<const uint64_t> vec,
    int original_cols) {
  namespace hn = hwy::HWY_NAMESPACE;
  hn::ScalableTag<uint64_t> d;
  const size_t N = hn::Lanes(d);

  if (original_cols % (2 * N) != 0) {
    return absl::InvalidArgumentError(
        "Original columns must be a multiple of 2 * Lanes.");
  }
  if (vec.size() != original_cols) {
    return absl::InvalidArgumentError("Vector size mismatch.");
  }
  int rows = condensed_matrix.Rows();
  int condensed_cols = original_cols / 2;

  std::vector<uint64_t> result(rows, 0);
  const std::vector<uint64_t>& data = condensed_matrix.CondensedData();

  int num_blocks = original_cols / (2 * N);
  std::vector<hn::Vec<decltype(d)>> V_0(num_blocks);
  std::vector<hn::Vec<decltype(d)>> V_1(num_blocks);

  for (int b = 0; b < num_blocks; ++b) {
    int base = b * 2 * N;
    hn::LoadInterleaved2(d, &vec[base], V_0[b], V_1[b]);
  }

  hn::ScalableTag<int64_t> d_signed;

  for (int i = 0; i < rows; ++i) {
    auto accum = hn::Zero(d);

    for (int b = 0; b < num_blocks; ++b) {
      auto m_condensed = hn::LoadU(d, &data[i * condensed_cols + b * N]);
      auto m_condensed_signed = hn::BitCast(d_signed, m_condensed);

      auto m1_signed = hn::ShiftRight<32>(m_condensed_signed);
      auto m0_signed =
          hn::ShiftRight<32>(hn::ShiftLeft<32>(m_condensed_signed));

      auto m0 = hn::BitCast(d, m0_signed);
      auto m1 = hn::BitCast(d, m1_signed);

      accum = hn::MulAdd(m0, V_0[b], accum);
      accum = hn::MulAdd(m1, V_1[b], accum);
    }

    result[i] = hn::ReduceSum(d, accum);
  }

  return result;
}

template <>
absl::StatusOr<std::vector<uint64_t>> CondensedMultiply<uint8_t>(
    const Matrix<uint8_t>& condensed_matrix, absl::Span<const uint64_t> vec,
    int original_cols) {
  namespace hn = hwy::HWY_NAMESPACE;
  hn::ScalableTag<uint64_t> d;
  const size_t N = hn::Lanes(d);

  if (original_cols % (8 * N) != 0) {
    return absl::InvalidArgumentError(
        "Original columns must be a multiple of 8 * Lanes.");
  }
  if (vec.size() != original_cols) {
    return absl::InvalidArgumentError("Vector size mismatch.");
  }
  int rows = condensed_matrix.Rows();
  int condensed_cols = original_cols / 8;

  std::vector<uint64_t> result(rows, 0);
  const std::vector<uint64_t>& data = condensed_matrix.CondensedData();

  int num_blocks = original_cols / (8 * N);

  // Rearrange the vector as suggested.
  std::vector<uint64_t> rearranged_vec(original_cols);
  for (int b = 0; b < num_blocks; ++b) {
    for (int k = 0; k < 8; ++k) {
      for (size_t l = 0; l < N; ++l) {
        rearranged_vec[b * 8 * N + k * N + l] = vec[(b * N + l) * 8 + k];
      }
    }
  }

  auto mask = hn::Set(d, 0xFF);

  for (int i = 0; i < rows; ++i) {
    auto accum = hn::Zero(d);

    for (int b = 0; b < num_blocks; ++b) {
      auto m_condensed = hn::LoadU(d, &data[i * condensed_cols + b * N]);

      // k = 0
      auto m0 = hn::And(m_condensed, mask);
      auto V_0 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 0 * N]);
      accum = hn::MulAdd(m0, V_0, accum);

      // k = 1
      auto m1 = hn::And(hn::ShiftRight<8>(m_condensed), mask);
      auto V_1 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 1 * N]);
      accum = hn::MulAdd(m1, V_1, accum);

      // k = 2
      auto m2 = hn::And(hn::ShiftRight<16>(m_condensed), mask);
      auto V_2 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 2 * N]);
      accum = hn::MulAdd(m2, V_2, accum);

      // k = 3
      auto m3 = hn::And(hn::ShiftRight<24>(m_condensed), mask);
      auto V_3 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 3 * N]);
      accum = hn::MulAdd(m3, V_3, accum);

      // k = 4
      auto m4 = hn::And(hn::ShiftRight<32>(m_condensed), mask);
      auto V_4 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 4 * N]);
      accum = hn::MulAdd(m4, V_4, accum);

      // k = 5
      auto m5 = hn::And(hn::ShiftRight<40>(m_condensed), mask);
      auto V_5 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 5 * N]);
      accum = hn::MulAdd(m5, V_5, accum);

      // k = 6
      auto m6 = hn::And(hn::ShiftRight<48>(m_condensed), mask);
      auto V_6 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 6 * N]);
      accum = hn::MulAdd(m6, V_6, accum);

      // k = 7
      auto m7 = hn::ShiftRight<56>(m_condensed);
      auto V_7 = hn::LoadU(d, &rearranged_vec[b * 8 * N + 7 * N]);
      accum = hn::MulAdd(m7, V_7, accum);
    }

    result[i] = hn::ReduceSum(d, accum);
  }

  return result;
}

}  // namespace

template <typename CoeffType>
absl::StatusOr<std::vector<uint64_t>> Matrix<CoeffType>::CondensedMultiply(
    const std::vector<uint64_t>& vec) const {
  if (!is_condensed_) {
    return absl::FailedPreconditionError(
        "CondensedMultiply can only be called on a condensed matrix.");
  }

  return ::private_membership::rlwe::v2::CondensedMultiply<CoeffType>(
      *this, vec, original_cols_);
}

template <typename CoeffType>
absl::StatusOr<proto::Matrix> Matrix<CoeffType>::ToProto() const {
  proto::Matrix proto;
  proto.set_rows(rows_);
  proto.set_cols(cols_);
  proto.set_is_condensed(is_condensed_);
  proto.set_original_cols(original_cols_);
  if (!data_.empty()) {
    proto.set_data(std::string(reinterpret_cast<const char*>(data_.data()),
                               data_.size() * sizeof(CoeffType)));
  }
  if (!condensed_data_.empty()) {
    proto.set_condensed_data(
        std::string(reinterpret_cast<const char*>(condensed_data_.data()),
                    condensed_data_.size() * sizeof(uint64_t)));
  }
  return proto;
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> Matrix<CoeffType>::FromProto(
    const proto::Matrix& proto) {
  if (proto.has_data() && proto.has_condensed_data()) {
    return absl::InvalidArgumentError(
        "Matrix cannot have both data and condensed_data set.");
  }
  if (proto.is_condensed() && proto.has_data()) {
    return absl::InvalidArgumentError(
        "Condensed matrix cannot have normal data set.");
  }
  if (!proto.is_condensed() && proto.has_condensed_data()) {
    return absl::InvalidArgumentError(
        "Normal matrix cannot have condensed data set.");
  }

  std::vector<CoeffType> data;
  if (!proto.data().empty()) {
    if (proto.data().size() % sizeof(CoeffType) != 0) {
      return absl::InvalidArgumentError("Invalid serialized data size.");
    }
    data.resize(proto.data().size() / sizeof(CoeffType));
    std::memcpy(data.data(), proto.data().data(), proto.data().size());
  }
  std::vector<uint64_t> condensed_data;
  if (!proto.condensed_data().empty()) {
    if (proto.condensed_data().size() % sizeof(uint64_t) != 0) {
      return absl::InvalidArgumentError(
          "Invalid serialized condensed_data size.");
    }
    condensed_data.resize(proto.condensed_data().size() / sizeof(uint64_t));
    std::memcpy(condensed_data.data(), proto.condensed_data().data(),
                proto.condensed_data().size());
  }
  return Matrix(std::move(data), std::move(condensed_data), proto.rows(),
                proto.cols(), proto.is_condensed(), proto.original_cols());
}

template class Matrix<uint8_t>;
template class Matrix<uint16_t>;
template class Matrix<uint32_t>;
template class Matrix<uint64_t>;
template class Matrix<int16_t>;
template class Matrix<int32_t>;

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
