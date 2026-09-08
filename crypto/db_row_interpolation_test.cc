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

#include "crypto/db_row_interpolation.h"

#include <cstdint>
#include <vector>

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

// Evaluates a polynomial P(X) = \sum P_i X^i at X = val.
// The coefficients are elements of Z_p[X] / (X^d + 1).
// `val` is also an element of Z_p[X] / (X^d + 1).
// Here `val` is always of the form X^k, so we just use cyclic shifts.
template <typename T>
std::vector<T> EvaluatePolynomial(
    const std::vector<std::vector<T>>& poly_coeffs, int eval_x_power, int d,
    T modulus) {
  std::vector<T> result(d, 0);

  for (int i = 0; i < poly_coeffs.size(); ++i) {
    int power = (i * eval_x_power) % (2 * d);
    if (power < 0) power += 2 * d;

    // cyclic shift poly_coeffs[i] by power
    bool negate_all = false;
    if (power >= d) {
      negate_all = true;
      power -= d;
    }

    std::vector<T> term(d);
    for (int j = 0; j < d; ++j) {
      T val;
      bool negate_this = negate_all;
      if (j < power) {
        val = poly_coeffs[i][d - power + j];
        negate_this = !negate_this;
      } else {
        val = poly_coeffs[i][j - power];
      }

      if (negate_this) {
        term[j] = (val == 0) ? 0 : modulus - val;
      } else {
        term[j] = val;
      }
    }

    // add term to result
    for (int j = 0; j < d; ++j) {
      result[j] = (result[j] + term[j]) % modulus;
    }
  }

  return result;
}

}  // namespace

template <typename T>
class DbRowInterpolationTest : public ::testing::Test {};

using CoeffTypes = ::testing::Types<uint16_t, uint32_t>;
TYPED_TEST_SUITE(DbRowInterpolationTest, CoeffTypes);

TYPED_TEST(DbRowInterpolationTest, RandomizedInputs) {
  using T = TypeParam;

  struct Param {
    int d;
    int t;
    uint32_t modulus;
  };

  std::vector<Param> params = {
      {32, 16, 1031},
      {64, 32, 2031},
      {128, 64, (1 << 16) - 1},
  };

  absl::BitGen bitgen;

  for (const auto& param : params) {
    const T modulus = static_cast<T>(param.modulus);
    const int d = param.d;
    const int t = param.t;

    std::vector<std::vector<T>> P(t, std::vector<T>(d));
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < d; ++j) {
        P[i][j] =
            static_cast<T>(absl::Uniform<uint32_t>(bitgen, 0, param.modulus));
      }
    }

    std::vector<T> evals(d * t);
    for (int i = 0; i < t; ++i) {
      // \omega^i = X^{i * 2d / t}
      int eval_power = (i * 2 * d / t) % (2 * d);
      std::vector<T> eval_i = EvaluatePolynomial(P, eval_power, d, modulus);
      for (int j = 0; j < d; ++j) {
        evals[i * d + j] = eval_i[j];
      }
    }

    ASSERT_OK_AND_ASSIGN(auto interpolated, Interpolate(evals, d, t, modulus));

    EXPECT_EQ(interpolated.size(), d * t);
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < d; ++j) {
        EXPECT_EQ(interpolated[i * d + j], P[i][j])
            << "Mismatch at d=" << d << ", t=" << t << ", polynomial " << i
            << ", coeff " << j;
      }
    }
  }
}

TYPED_TEST(DbRowInterpolationTest, InvalidArguments) {
  using T = TypeParam;

  const T modulus = 1031;
  const int d = 32;
  const int t = 16;

  std::vector<T> db_row(d * t);
  for (int i = 0; i < d * t; ++i) {
    db_row[i] = i % modulus;
  }

  // Valid arguments
  EXPECT_OK(Interpolate(db_row, d, t, modulus));

  // Invalid db_row size.
  std::vector<T> invalid_db_row(d * t - 1);
  EXPECT_THAT(
      Interpolate(invalid_db_row, d, t, modulus),
      StatusIs(absl::StatusCode::kInvalidArgument,
               "Length of evals must be ring_degree * interpolation_degree."));

  // Invalid ring_degree.
  std::vector<T> db_row_d((d - 1) * t);
  EXPECT_THAT(Interpolate(db_row_d, d - 1, t, modulus),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "ring_degree must be a power of two."));

  // Invalid interpolation_degree.
  std::vector<T> db_row_t(d * (t - 1));
  EXPECT_THAT(Interpolate(db_row_t, d, t - 1, modulus),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "interpolation_degree must be a power of two."));

  // interpolation_degree > 2 * ring_degree.
  std::vector<T> db_row_t_large(32 * 128);
  EXPECT_THAT(Interpolate(db_row_t_large, 32, 128, modulus),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "interpolation_degree must be less than or equal to 2 * "
                       "ring_degree."));

  // Invalid plaintext_modulus.
  EXPECT_THAT(Interpolate(db_row, d, t, static_cast<T>(modulus - 1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "plaintext_modulus must be odd."));
}

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
