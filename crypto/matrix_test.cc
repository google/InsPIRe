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

#include <cstdint>
#include <utility>
#include <vector>

#include "crypto/proto/pir_server.pb.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/status/status.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Test;

template <typename T>
class MatrixTest : public Test {};

using CoeffTypes = ::testing::Types<uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(MatrixTest, CoeffTypes);

TYPED_TEST(MatrixTest, CreateEmpty) {
  auto matrix_or = Matrix<TypeParam>::Create({});
  ASSERT_OK(matrix_or.status());
  EXPECT_EQ(matrix_or->Rows(), 0);
  EXPECT_EQ(matrix_or->Cols(), 0);
}

TYPED_TEST(MatrixTest, CreateValid) {
  std::vector<std::vector<TypeParam>> data = {{1, 2, 3}, {4, 5, 6}};
  auto matrix_or = Matrix<TypeParam>::Create(data);
  ASSERT_OK(matrix_or.status());
  EXPECT_EQ(matrix_or->Rows(), 2);
  EXPECT_EQ(matrix_or->Cols(), 3);
  std::vector<TypeParam> expected_data = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(matrix_or->Data(), expected_data);
}

TYPED_TEST(MatrixTest, CreateInvalidRowSizes) {
  std::vector<std::vector<TypeParam>> data = {{1, 2}, {3, 4, 5}};
  auto status = Matrix<TypeParam>::Create(data).status();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("same number of columns"));
}

TYPED_TEST(MatrixTest, CreateFromDimensionsValid) {
  auto matrix_or = Matrix<TypeParam>::Create(2, 3);
  ASSERT_OK(matrix_or.status());
  EXPECT_EQ(matrix_or->Rows(), 2);
  EXPECT_EQ(matrix_or->Cols(), 3);
  std::vector<TypeParam> expected_data(6, 0);
  EXPECT_EQ(matrix_or->Data(), expected_data);
}

TYPED_TEST(MatrixTest, CreateFromDimensionsInvalid) {
  EXPECT_FALSE(Matrix<TypeParam>::Create(-1, 3).status().ok());
  EXPECT_FALSE(Matrix<TypeParam>::Create(2, -1).status().ok());
}

TYPED_TEST(MatrixTest, MultiplyValid) {
  std::vector<std::vector<TypeParam>> data = {{1, 2}, {3, 4}};
  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<TypeParam>::Create(data));
  std::vector<TypeParam> vec = {5, 6};
  ASSERT_OK_AND_ASSIGN(auto result, matrix.template Multiply<TypeParam>(vec));
  EXPECT_THAT(result, ElementsAre(17, 39));
}

TEST(CondensedMultiplyHighwayU16U64Test, Correctness) {
  const int rows = 2;
  const int cols = 32;

  std::vector<std::vector<uint16_t>> matrix_data(rows,
                                                 std::vector<uint16_t>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] =
          (j % 4) + 1;  // Raw elements [1, 2, 3, 4] repeated cols/4 times
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix,
                       Matrix<uint16_t>::CreateCondensed(matrix_data));

  std::vector<uint64_t> vec(cols, 1);

  ASSERT_OK_AND_ASSIGN(auto result, matrix.template Multiply<uint64_t>(vec));

  EXPECT_THAT(result, ElementsAre(80, 80));
}

TEST(CondensedMultiplyHighwayI32U64Test, Correctness) {
  const int rows = 2;
  const int cols = 32;

  std::vector<std::vector<int32_t>> matrix_data(rows,
                                                std::vector<int32_t>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] =
          (j % 2 == 0) ? -1 : 2;  // Raw elements [-1, 2] repeated cols/2 times
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix,
                       Matrix<int32_t>::CreateCondensed(matrix_data));

  std::vector<uint64_t> vec(cols, 1);

  ASSERT_OK_AND_ASSIGN(auto result, matrix.template Multiply<uint64_t>(vec));

  EXPECT_THAT(result, ElementsAre(16, 16));
}

TEST(CondensedMultiplyHighwayU8U64Test, Correctness) {
  const int rows = 2;
  const int cols = 64;

  std::vector<std::vector<uint8_t>> matrix_data(rows,
                                                std::vector<uint8_t>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] =
          (j % 8) +
          1;  // Raw elements [1, 2, 3, 4, 5, 6, 7, 8] repeated cols/8 times
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix,
                       Matrix<uint8_t>::CreateCondensed(matrix_data));

  std::vector<uint64_t> vec(cols, 1);

  ASSERT_OK_AND_ASSIGN(auto result, matrix.template Multiply<uint64_t>(vec));

  EXPECT_THAT(result, ElementsAre(288, 288));
}

TEST(CondensedMultiplyHighwayU8U64Test, CorrectnessLargeScale) {
  const int rows = 2;
  const int cols = 1024;  // Dimensions matching typical benchmark setups

  std::vector<std::vector<uint8_t>> matrix_data(rows,
                                                std::vector<uint8_t>(cols));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] = (j % 8) + 1;
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix,
                       Matrix<uint8_t>::CreateCondensed(matrix_data));

  std::vector<uint64_t> vec(cols, 1);

  ASSERT_OK_AND_ASSIGN(auto result, matrix.template Multiply<uint64_t>(vec));

  // Each block of 8 elements sums to 36.
  // There are cols / 8 = 128 blocks.
  // Total sum per row = 128 * 36 = 4608.
  EXPECT_THAT(result, ElementsAre(4608, 4608));
}

TEST(CondensedMatrixErrorTest, StandardOperationsFailOnCondensedMatrix) {
  std::vector<std::vector<uint16_t>> matrix_data = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  ASSERT_OK_AND_ASSIGN(auto matrix,
                       Matrix<uint16_t>::CreateCondensed(matrix_data));

  EXPECT_TRUE(matrix.IsCondensed());

  // AddInPlace should fail
  std::vector<std::vector<uint16_t>> other_matrix_data = {{1}, {2}};
  ASSERT_OK_AND_ASSIGN(auto matrix2,
                       Matrix<uint16_t>::Create(other_matrix_data));
  auto add_status = matrix.AddInPlace(matrix2);
  EXPECT_FALSE(add_status.ok());
  EXPECT_EQ(add_status.code(), absl::StatusCode::kFailedPrecondition);

  // SubMatrix should fail
  auto submatrix_status = matrix.SubMatrix(0, 0, 1, 1).status();
  EXPECT_FALSE(submatrix_status.ok());
  EXPECT_EQ(submatrix_status.code(), absl::StatusCode::kFailedPrecondition);
}

TYPED_TEST(MatrixTest, MultiplyWithDifferentOutputAndVecType) {
  std::vector<std::vector<TypeParam>> data = {{1, 2}, {3, 4}};
  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<TypeParam>::Create(data));
  std::vector<uint8_t> vec = {5, 6};
  ASSERT_OK_AND_ASSIGN(auto result, matrix.template Multiply<uint64_t>(vec));
  EXPECT_THAT(result, ElementsAre(17, 39));
}

TYPED_TEST(MatrixTest, MultiplyInvalidVectorSize) {
  std::vector<std::vector<TypeParam>> data = {{1, 2}, {3, 4}};
  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<TypeParam>::Create(data));
  std::vector<TypeParam> vec = {5};
  auto status = matrix.template Multiply<TypeParam>(vec).status();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Vector size must match"));
}

TYPED_TEST(MatrixTest, AddInPlaceValid) {
  std::vector<std::vector<TypeParam>> data1 = {{1, 2}, {3, 4}};
  ASSERT_OK_AND_ASSIGN(auto matrix1, Matrix<TypeParam>::Create(data1));
  std::vector<std::vector<TypeParam>> data2 = {{5, 6}, {7, 8}};
  ASSERT_OK_AND_ASSIGN(auto matrix2, Matrix<TypeParam>::Create(data2));

  ASSERT_OK(matrix1.AddInPlace(matrix2));
  EXPECT_THAT(matrix1.Data(), ElementsAre(6, 8, 10, 12));
}

TYPED_TEST(MatrixTest, AddInPlaceInvalidDimensions) {
  std::vector<std::vector<TypeParam>> data1 = {{1, 2}, {3, 4}};
  ASSERT_OK_AND_ASSIGN(auto matrix1, Matrix<TypeParam>::Create(data1));
  std::vector<std::vector<TypeParam>> data2 = {{1, 2, 3}, {4, 5, 6}};
  ASSERT_OK_AND_ASSIGN(auto matrix2, Matrix<TypeParam>::Create(data2));

  auto status = matrix1.AddInPlace(matrix2);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Matrix dimensions must match"));
}

TYPED_TEST(MatrixTest, SubMatrixValid) {
  std::vector<std::vector<TypeParam>> data = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
  };
  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<TypeParam>::Create(data));
  ASSERT_OK_AND_ASSIGN(auto submatrix, matrix.SubMatrix(1, 1, 2, 2));

  EXPECT_EQ(submatrix.Rows(), 2);
  EXPECT_EQ(submatrix.Cols(), 2);
  EXPECT_THAT(submatrix.Data(), ElementsAre(6, 7, 10, 11));
}

TYPED_TEST(MatrixTest, SubMatrixInvalidOffsets) {
  std::vector<std::vector<TypeParam>> data = {
      {1, 2, 3},
      {4, 5, 6},
  };
  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<TypeParam>::Create(data));
  EXPECT_FALSE(matrix.SubMatrix(-1, 0, 1, 1).ok());
  EXPECT_FALSE(matrix.SubMatrix(0, -1, 1, 1).ok());
  EXPECT_FALSE(matrix.SubMatrix(0, 0, -1, 1).ok());
  EXPECT_FALSE(matrix.SubMatrix(0, 0, 1, -1).ok());
}

TYPED_TEST(MatrixTest, SubMatrixOutOfBounds) {
  std::vector<std::vector<TypeParam>> data = {
      {1, 2, 3},
      {4, 5, 6},
  };
  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<TypeParam>::Create(data));
  EXPECT_FALSE(matrix.SubMatrix(1, 0, 2, 1).ok());
  EXPECT_FALSE(matrix.SubMatrix(0, 2, 1, 2).ok());
}

TYPED_TEST(MatrixTest, ToFromProtoValid) {
  std::vector<std::vector<TypeParam>> data = {{1, 2}, {3, 4}};
  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<TypeParam>::Create(data));
  ASSERT_OK_AND_ASSIGN(auto proto, matrix.ToProto());
  ASSERT_OK_AND_ASSIGN(auto deserialized_matrix,
                       Matrix<TypeParam>::FromProto(proto));
  EXPECT_EQ(deserialized_matrix.Rows(), matrix.Rows());
  EXPECT_EQ(deserialized_matrix.Cols(), matrix.Cols());
  EXPECT_EQ(deserialized_matrix.IsCondensed(), matrix.IsCondensed());
  EXPECT_EQ(deserialized_matrix.Data(), matrix.Data());
}

TYPED_TEST(MatrixTest, ToFromProtoCondensedValid) {
  if constexpr (!std::is_same_v<TypeParam, uint16_t>) {
    return;
  }
  std::vector<std::vector<TypeParam>> data = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<TypeParam>::CreateCondensed(data));
  ASSERT_OK_AND_ASSIGN(auto proto, matrix.ToProto());
  ASSERT_OK_AND_ASSIGN(auto deserialized_matrix,
                       Matrix<TypeParam>::FromProto(proto));
  EXPECT_EQ(deserialized_matrix.Rows(), matrix.Rows());
  EXPECT_EQ(deserialized_matrix.Cols(), matrix.Cols());
  EXPECT_EQ(deserialized_matrix.IsCondensed(), matrix.IsCondensed());
  EXPECT_EQ(deserialized_matrix.CondensedData(), matrix.CondensedData());
}

TYPED_TEST(MatrixTest, FromProtoInvalidMatrixDataTypeChecks) {
  proto::Matrix proto;
  proto.set_rows(2);
  proto.set_cols(2);

  // Both set. Since it's proto oneof we can construct it or set the values.
  // Wait, setting one field in a oneof union clears the other. However, we can
  // construct the serialized bytes or set it programmatically if we want, or
  // just verify validation handles set fields when both present or mismatch
  // is_condensed. Let's test the mismatches with is_condensed.
  proto.set_data("data");
  proto.set_is_condensed(true);
  EXPECT_EQ(Matrix<TypeParam>::FromProto(proto).status().code(),
            absl::StatusCode::kInvalidArgument);

  proto.clear_data();
  proto.set_condensed_data("condensed");
  proto.set_is_condensed(false);
  EXPECT_EQ(Matrix<TypeParam>::FromProto(proto).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TYPED_TEST(MatrixTest, VerticalConcatenateEmpty) {
  std::vector<Matrix<TypeParam>> matrices;
  ASSERT_OK_AND_ASSIGN(auto result,
                       Matrix<TypeParam>::VerticalConcatenate(matrices));
  EXPECT_EQ(result.Rows(), 0);
  EXPECT_EQ(result.Cols(), 0);
}

TYPED_TEST(MatrixTest, VerticalConcatenateSingleMatrix) {
  std::vector<std::vector<TypeParam>> data = {{1, 2}, {3, 4}};
  ASSERT_OK_AND_ASSIGN(auto m, Matrix<TypeParam>::Create(data));

  std::vector<Matrix<TypeParam>> matrices = {m};
  ASSERT_OK_AND_ASSIGN(auto result,
                       Matrix<TypeParam>::VerticalConcatenate(matrices));

  EXPECT_EQ(result.Rows(), 2);
  EXPECT_EQ(result.Cols(), 2);
  EXPECT_EQ(result.Data(), m.Data());
}

TYPED_TEST(MatrixTest, VerticalConcatenateValidNormal) {
  std::vector<std::vector<TypeParam>> data1 = {{1, 2}, {3, 4}};
  std::vector<std::vector<TypeParam>> data2 = {{5, 6}};
  ASSERT_OK_AND_ASSIGN(auto m1, Matrix<TypeParam>::Create(data1));
  ASSERT_OK_AND_ASSIGN(auto m2, Matrix<TypeParam>::Create(data2));

  std::vector<Matrix<TypeParam>> matrices = {m1, m2};
  ASSERT_OK_AND_ASSIGN(auto result,
                       Matrix<TypeParam>::VerticalConcatenate(matrices));

  EXPECT_EQ(result.Rows(), 3);
  EXPECT_EQ(result.Cols(), 2);
  EXPECT_FALSE(result.IsCondensed());
  EXPECT_THAT(result.Data(), ElementsAre(1, 2, 3, 4, 5, 6));
}

TYPED_TEST(MatrixTest, VerticalConcatenateMismatchedCols) {
  std::vector<std::vector<TypeParam>> data1 = {{1, 2}, {3, 4}};
  std::vector<std::vector<TypeParam>> data2 = {{5, 6, 7}};
  ASSERT_OK_AND_ASSIGN(auto m1, Matrix<TypeParam>::Create(data1));
  ASSERT_OK_AND_ASSIGN(auto m2, Matrix<TypeParam>::Create(data2));

  std::vector<Matrix<TypeParam>> matrices = {m1, m2};
  auto status = Matrix<TypeParam>::VerticalConcatenate(matrices).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("same number of columns"));
}

TYPED_TEST(MatrixTest, VerticalConcatenateValidCondensed) {
  if constexpr (!std::is_same_v<TypeParam, uint16_t>) {
    return;
  }
  std::vector<std::vector<TypeParam>> data1 = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  std::vector<std::vector<TypeParam>> data2 = {{9, 10, 11, 12}};
  ASSERT_OK_AND_ASSIGN(auto m1, Matrix<TypeParam>::CreateCondensed(data1));
  ASSERT_OK_AND_ASSIGN(auto m2, Matrix<TypeParam>::CreateCondensed(data2));

  std::vector<Matrix<TypeParam>> matrices = {m1, m2};
  ASSERT_OK_AND_ASSIGN(auto result,
                       Matrix<TypeParam>::VerticalConcatenate(matrices));

  EXPECT_EQ(result.Rows(), 3);
  EXPECT_EQ(result.Cols(), 4);
  EXPECT_TRUE(result.IsCondensed());

  // Verify condensed data
  std::vector<uint64_t> expected_condensed;
  expected_condensed.insert(expected_condensed.end(),
                            m1.CondensedData().begin(),
                            m1.CondensedData().end());
  expected_condensed.insert(expected_condensed.end(),
                            m2.CondensedData().begin(),
                            m2.CondensedData().end());
  EXPECT_EQ(result.CondensedData(), expected_condensed);
}

TYPED_TEST(MatrixTest, VerticalConcatenateMismatchedCondensedState) {
  if constexpr (!std::is_same_v<TypeParam, uint16_t>) {
    return;
  }
  std::vector<std::vector<TypeParam>> data1 = {{1, 2, 3, 4}};
  std::vector<std::vector<TypeParam>> data2 = {{5, 6, 7, 8}};
  ASSERT_OK_AND_ASSIGN(auto m1, Matrix<TypeParam>::CreateCondensed(data1));
  ASSERT_OK_AND_ASSIGN(auto m2, Matrix<TypeParam>::Create(data2));

  std::vector<Matrix<TypeParam>> matrices = {m1, m2};
  auto status = Matrix<TypeParam>::VerticalConcatenate(matrices).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("same condensed state"));
}

}  // namespace

TEST(MatrixTestExtra, VerticalConcatenateCrashingCondensedU8) {
  const int rows = 16;
  const int cols = 1024;
  std::vector<std::vector<uint8_t>> data1(rows, std::vector<uint8_t>(cols, 0));
  std::vector<std::vector<uint8_t>> data2(rows, std::vector<uint8_t>(cols, 0));
  ASSERT_OK_AND_ASSIGN(auto m1, Matrix<uint8_t>::CreateCondensed(data1));
  ASSERT_OK_AND_ASSIGN(auto m2, Matrix<uint8_t>::CreateCondensed(data2));
  std::vector<Matrix<uint8_t>> matrices = {m1, m2};
  auto status_or = Matrix<uint8_t>::VerticalConcatenate(matrices);
  ASSERT_OK(status_or.status());
}

TEST(MatrixTestExtra, VerticalConcatenateCrashingPackI32) {
  const int rows = 16;
  const int cols = 512;
  const int num_matrices = 4;
  std::vector<Matrix<int32_t>> matrices;
  matrices.reserve(num_matrices);
  for (int i = 0; i < num_matrices; ++i) {
    ASSERT_OK_AND_ASSIGN(auto m, Matrix<int32_t>::Create(rows, cols));
    matrices.push_back(std::move(m));
  }
  auto status_or = Matrix<int32_t>::VerticalConcatenate(matrices);
  ASSERT_OK(status_or.status());
}

TEST(MatrixTestExtra, VerticalConcatenateZServerCondensedU8) {
  const int rows = 16;
  const int cols = 128;
  const int num_matrices = 4;
  std::vector<Matrix<uint8_t>> matrices;
  matrices.reserve(num_matrices);
  for (int i = 0; i < num_matrices; ++i) {
    std::vector<std::vector<uint8_t>> data(rows, std::vector<uint8_t>(cols, 0));
    ASSERT_OK_AND_ASSIGN(auto m, Matrix<uint8_t>::CreateCondensed(data));
    matrices.push_back(std::move(m));
  }
  auto status_or = Matrix<uint8_t>::VerticalConcatenate(matrices);
  ASSERT_OK(status_or.status());
}

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
