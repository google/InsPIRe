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

#include <cstdint>
#include <utility>
#include <vector>

#include "crypto/matrix.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/random/random.h"
#include "absl/status/status.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using ::rlwe::testing::StatusIs;

template <typename S, typename T>
std::vector<std::vector<T>> NaiveMulMatNegacyclic(
    const std::vector<std::vector<S>>& mat_data,
    const std::vector<T>& poly_data) {
  const int rows = mat_data.size();
  const int d = poly_data.size();

  // Construct the d x d negacyclic matrix from poly_data.
  std::vector<std::vector<T>> M(d, std::vector<T>(d));
  for (int i = 0; i < d; ++i) {
    for (int j = 0; j < d; ++j) {
      if (i >= j) {
        M[i][j] = poly_data[i - j];
      } else {
        M[i][j] = static_cast<T>(0) - poly_data[d + i - j];
      }
    }
  }

  std::vector<std::vector<T>> out(rows, std::vector<T>(d, 0));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < d; ++j) {
      for (int k = 0; k < d; ++k) {
        out[i][j] += static_cast<T>(mat_data[i][k]) * M[k][j];
      }
    }
  }
  return out;
}

template <typename TypePair>
class MulMatNegacyclicTest : public ::testing::Test {
 public:
  using S = typename TypePair::first_type;
  using T = typename TypePair::second_type;
};

using CoeffTypes = ::testing::Types<std::pair<uint16_t, uint32_t>,
                                    std::pair<uint16_t, uint64_t>,
                                    std::pair<uint32_t, uint64_t>>;
TYPED_TEST_SUITE(MulMatNegacyclicTest, CoeffTypes);

TYPED_TEST(MulMatNegacyclicTest, RandomMatrixMultiplication) {
  using S = typename TestFixture::S;
  using T = typename TestFixture::T;

  absl::BitGen bitgen;

  for (int log_d = 1; log_d <= 5; ++log_d) {
    int d = 1 << log_d;
    ASSERT_OK_AND_ASSIGN(auto ctx, FftContext::Create(log_d));

    for (int rows : {1, 2, d, d + 5}) {
      std::vector<std::vector<S>> mat_data(rows, std::vector<S>(d));
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < d; ++j) {
          mat_data[i][j] = absl::Uniform<S>(bitgen);
        }
      }

      std::vector<T> poly_data(d);
      for (int j = 0; j < d; ++j) {
        poly_data[j] = absl::Uniform<T>(bitgen);
      }

      ASSERT_OK_AND_ASSIGN(auto mat, Matrix<S>::Create(mat_data));
      ASSERT_OK_AND_ASSIGN(auto poly, Polynomial<T>::Create(poly_data));

      std::vector<std::vector<T>> expected_data =
          NaiveMulMatNegacyclic(mat_data, poly_data);

      ASSERT_OK_AND_ASSIGN(auto res, (MulMatNegacyclic<T>(mat, poly, *ctx)));

      const std::vector<T>& result_data = res.Data();

      ASSERT_EQ(res.Rows(), rows);
      ASSERT_EQ(res.Cols(), d);

      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < d; ++j) {
          EXPECT_EQ(result_data[i * d + j], expected_data[i][j]);
        }
      }
    }
  }
}

TYPED_TEST(MulMatNegacyclicTest, InvalidParameters) {
  using S = typename TestFixture::S;
  using T = typename TestFixture::T;

  ASSERT_OK_AND_ASSIGN(auto ctx, FftContext::Create(2));

  // Test 0x0 empty matrix
  std::vector<std::vector<S>> mat_data_empty;
  ASSERT_OK_AND_ASSIGN(auto mat_empty, Matrix<S>::Create(mat_data_empty));
  std::vector<T> poly_data = {1, 2, 3, 4};
  ASSERT_OK_AND_ASSIGN(auto poly, Polynomial<T>::Create(poly_data));
  auto res_empty = MulMatNegacyclic<T>(mat_empty, poly, *ctx);
  EXPECT_THAT(res_empty.status(), StatusIs(absl::StatusCode::kInvalidArgument,
                                           "Matrix cannot be empty."));

  // Test mismatched dimensions
  std::vector<std::vector<S>> mat_data_mismatch = {{1, 2}, {3, 4}};
  ASSERT_OK_AND_ASSIGN(auto mat_mismatch, Matrix<S>::Create(mat_data_mismatch));
  auto res_mismatch = MulMatNegacyclic<T>(mat_mismatch, poly, *ctx);
  EXPECT_THAT(res_mismatch.status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Matrix columns must match polynomial length."));
}

template <typename S, typename T>
std::vector<std::vector<T>> NaiveComputeHintMatrix(
    const std::vector<std::vector<S>>& database_data,
    const std::vector<std::vector<T>>& a_components_data) {
  const int rows = database_data.size();
  const int n = a_components_data.size();
  const int d = a_components_data[0].size();

  std::vector<std::vector<T>> out(rows, std::vector<T>(d, 0));

  for (int j = 0; j < n; ++j) {
    std::vector<std::vector<S>> submat_data(rows, std::vector<S>(d));
    for (int i = 0; i < rows; ++i) {
      for (int k = 0; k < d; ++k) {
        submat_data[i][k] = database_data[i][j * d + k];
      }
    }
    std::vector<std::vector<T>> submat_out =
        NaiveMulMatNegacyclic(submat_data, a_components_data[j]);

    for (int i = 0; i < rows; ++i) {
      for (int k = 0; k < d; ++k) {
        out[i][k] += submat_out[i][k];
      }
    }
  }

  return out;
}

TYPED_TEST(MulMatNegacyclicTest, ComputeHintMatrixRandomMatrix) {
  using S = typename TestFixture::S;
  using T = typename TestFixture::T;

  absl::BitGen bitgen;

  for (int log_d = 1; log_d <= 4; ++log_d) {
    int d = 1 << log_d;
    ASSERT_OK_AND_ASSIGN(auto ctx, FftContext::Create(log_d));

    for (int n : {1, 2, 4}) {
      for (int rows : {1, 2, d, d + 5}) {
        std::vector<std::vector<S>> database_data(rows, std::vector<S>(d * n));
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < d * n; ++j) {
            database_data[i][j] = absl::Uniform<S>(bitgen);
          }
        }

        std::vector<std::vector<T>> a_components_data(n, std::vector<T>(d));
        std::vector<Polynomial<T>> a_components;
        a_components.reserve(n);
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < d; ++j) {
            a_components_data[i][j] = absl::Uniform<T>(bitgen);
          }
          ASSERT_OK_AND_ASSIGN(auto a_component,
                               Polynomial<T>::Create(a_components_data[i]));
          a_components.push_back(std::move(a_component));
        }

        ASSERT_OK_AND_ASSIGN(auto database, Matrix<S>::Create(database_data));

        std::vector<std::vector<T>> expected_data =
            NaiveComputeHintMatrix(database_data, a_components_data);

        ASSERT_OK_AND_ASSIGN(
            auto res, (ComputeHintMatrix<T>(database, a_components, *ctx)));

        const std::vector<T>& result_data = res.Data();

        ASSERT_EQ(res.Rows(), rows);
        ASSERT_EQ(res.Cols(), d);

        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < d; ++j) {
            EXPECT_EQ(result_data[i * d + j], expected_data[i][j]);
          }
        }
      }
    }
  }
}

TYPED_TEST(MulMatNegacyclicTest, ComputeHintMatrixInvalidParameters) {
  using S = typename TestFixture::S;
  using T = typename TestFixture::T;

  ASSERT_OK_AND_ASSIGN(auto ctx, FftContext::Create(2));
  int d = 4;

  // Test empty database
  std::vector<std::vector<S>> db_data_empty;
  ASSERT_OK_AND_ASSIGN(auto db_empty, Matrix<S>::Create(db_data_empty));
  ASSERT_OK_AND_ASSIGN(auto poly_val,
                       Polynomial<T>::Create(std::vector<T>(d, 0)));
  std::vector<Polynomial<T>> a_components(1, poly_val);
  auto res_empty = ComputeHintMatrix<T>(db_empty, a_components, *ctx);
  EXPECT_THAT(res_empty.status(), StatusIs(absl::StatusCode::kInvalidArgument,
                                           "Database matrix cannot be empty."));

  // Test empty a_components
  std::vector<std::vector<S>> db_data_valid(2, std::vector<S>(d, 0));
  ASSERT_OK_AND_ASSIGN(auto db_valid, Matrix<S>::Create(db_data_valid));
  std::vector<Polynomial<T>> a_components_empty;
  auto res_empty_a = ComputeHintMatrix<T>(db_valid, a_components_empty, *ctx);
  EXPECT_THAT(res_empty_a.status(), StatusIs(absl::StatusCode::kInvalidArgument,
                                             "a_components cannot be empty."));

  // Test mismatched dimensions
  std::vector<std::vector<S>> db_data_mismatch(2, std::vector<S>(3, 0));
  ASSERT_OK_AND_ASSIGN(auto db_mismatch, Matrix<S>::Create(db_data_mismatch));
  auto res_mismatch = ComputeHintMatrix<T>(db_mismatch, a_components, *ctx);
  EXPECT_THAT(
      res_mismatch.status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "Database columns must match d * n where d is the polynomial length "
          "and n is the number of polynomials."));
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
