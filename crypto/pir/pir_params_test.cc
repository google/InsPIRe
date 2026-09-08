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

#include "crypto/pir/pir_params.h"

#include <cstdint>

#include "crypto/encryption.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using ::testing::HasSubstr;

constexpr int kDegree = 1024;
constexpr int kModulusBitSize = 9;

template <typename T>
class PirParamsTest : public testing::Test {
 protected:
  absl::StatusOr<RlweParams<T>> CreateRlweParams(int degree) {
    return RlweParams<T>::Create(degree, 1 << kModulusBitSize, 2);
  }

  GadgetParams CreateGadgetParams() {
    return GadgetParams{/*log_digit=*/2, /*num_digits=*/4};
  }
};

using MyTypes = ::testing::Types<uint32_t, uint64_t>;
TYPED_TEST_SUITE(PirParamsTest, MyTypes);

TYPED_TEST(PirParamsTest, ValidParams) {
  ASSERT_OK_AND_ASSIGN(auto rlwe_params, this->CreateRlweParams(kDegree));
  auto gadget_params = this->CreateGadgetParams();

  int num_entries = 4096;
  int interpolation_degree = 2;

  ASSERT_OK_AND_ASSIGN(
      auto pir_params,
      PirParams<TypeParam>::Create(rlwe_params, 5, gadget_params, gadget_params,
                                   num_entries, interpolation_degree));

  EXPECT_EQ(pir_params.NumEntries(), num_entries);
  EXPECT_EQ(pir_params.InterpolationDegree(), interpolation_degree);
  EXPECT_EQ(pir_params.RlweParameters().Degree(), kDegree);
  EXPECT_EQ(pir_params.PlaintextModulus(), 5);
}

TYPED_TEST(PirParamsTest, InvalidNumEntries) {
  ASSERT_OK_AND_ASSIGN(auto rlwe_params, this->CreateRlweParams(kDegree));
  auto gadget_params = this->CreateGadgetParams();

  int num_entries = 0;
  int interpolation_degree = 2;

  auto pir_params =
      PirParams<TypeParam>::Create(rlwe_params, 5, gadget_params, gadget_params,
                                   num_entries, interpolation_degree);

  EXPECT_FALSE(pir_params.ok());
  EXPECT_EQ(pir_params.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(pir_params.status().message(), HasSubstr("strictly positive"));
}

TYPED_TEST(PirParamsTest, InterpolationDegreeNotPowerOfTwo) {
  ASSERT_OK_AND_ASSIGN(auto rlwe_params, this->CreateRlweParams(kDegree));
  auto gadget_params = this->CreateGadgetParams();

  int num_entries = 4096;
  int interpolation_degree = 3;

  auto pir_params =
      PirParams<TypeParam>::Create(rlwe_params, 5, gadget_params, gadget_params,
                                   num_entries, interpolation_degree);

  EXPECT_FALSE(pir_params.ok());
  EXPECT_EQ(pir_params.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(pir_params.status().message(), HasSubstr("power of two"));
}

TYPED_TEST(PirParamsTest, InterpolationDegreeDoesNotDivideNumEntries) {
  ASSERT_OK_AND_ASSIGN(auto rlwe_params, this->CreateRlweParams(kDegree));
  auto gadget_params = this->CreateGadgetParams();

  int num_entries = 4097;
  int interpolation_degree = 2;

  auto pir_params =
      PirParams<TypeParam>::Create(rlwe_params, 5, gadget_params, gadget_params,
                                   num_entries, interpolation_degree);

  EXPECT_FALSE(pir_params.ok());
  EXPECT_EQ(pir_params.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(pir_params.status().message(),
              HasSubstr("must divide num_entries"));
}

TYPED_TEST(PirParamsTest, ValidParamsWhenRowsNotDivisibleByRingDegree) {
  ASSERT_OK_AND_ASSIGN(auto rlwe_params, this->CreateRlweParams(kDegree));
  auto gadget_params = this->CreateGadgetParams();

  int num_entries = 2048;
  int interpolation_degree = 4;  // Rows = 512, which is not divisible by 1024

  ASSERT_OK_AND_ASSIGN(
      auto pir_params,
      PirParams<TypeParam>::Create(rlwe_params, 5, gadget_params, gadget_params,
                                   num_entries, interpolation_degree));

  EXPECT_EQ(pir_params.NumEntries(), num_entries);
  EXPECT_EQ(pir_params.InterpolationDegree(), interpolation_degree);
  EXPECT_EQ(pir_params.InterpolatedRows(), 512);
  EXPECT_EQ(pir_params.NumFirstDimSamples(), 1);
  EXPECT_EQ(pir_params.PaddedCols(), 1024);

  // Test another case spanning multiple RLWE samples
  int num_entries_2 = 2500;
  int interpolation_degree_2 = 2;  // Rows = 1250

  ASSERT_OK_AND_ASSIGN(
      auto pir_params_2,
      PirParams<TypeParam>::Create(rlwe_params, 5, gadget_params, gadget_params,
                                   num_entries_2, interpolation_degree_2));

  EXPECT_EQ(pir_params_2.NumEntries(), num_entries_2);
  EXPECT_EQ(pir_params_2.InterpolationDegree(), interpolation_degree_2);
  EXPECT_EQ(pir_params_2.InterpolatedRows(), 1250);
  EXPECT_EQ(pir_params_2.NumFirstDimSamples(), 2);
  EXPECT_EQ(pir_params_2.PaddedCols(), 2048);
}

TYPED_TEST(PirParamsTest, InterpolationDegreeTooLarge) {
  ASSERT_OK_AND_ASSIGN(auto rlwe_params, this->CreateRlweParams(2));
  auto gadget_params = this->CreateGadgetParams();

  int num_entries = 16;
  // rows = 2, which divides d = 2, but degree = 8 > 2 * d = 4
  int interpolation_degree = 8;

  auto pir_params =
      PirParams<TypeParam>::Create(rlwe_params, 5, gadget_params, gadget_params,
                                   num_entries, interpolation_degree);

  EXPECT_FALSE(pir_params.ok());
  EXPECT_EQ(pir_params.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(pir_params.status().message(),
              HasSubstr("must be less than or equal to 2 * ring degree"));
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
