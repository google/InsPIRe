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

#include "crypto/vpir/vpir_params.h"

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
class VpirParamsTest : public testing::Test {
 protected:
  absl::StatusOr<RlweParams<T>> CreateRlweParams(int degree) {
    return RlweParams<T>::Create(degree, 1 << kModulusBitSize, 2);
  }

  GadgetParams CreateGadgetParams() {
    return GadgetParams{/*log_digit=*/2, /*num_digits=*/4};
  }
};

using MyTypes = ::testing::Types<uint32_t, uint64_t>;
TYPED_TEST_SUITE(VpirParamsTest, MyTypes);

TYPED_TEST(VpirParamsTest, ValidParams) {
  ASSERT_OK_AND_ASSIGN(auto rlwe_params, this->CreateRlweParams(kDegree));
  auto gadget_params = this->CreateGadgetParams();

  int num_entries = 4096;

  ASSERT_OK_AND_ASSIGN(auto vpir_params,
                       VpirParams<TypeParam>::Create(
                           rlwe_params, 5, gadget_params, num_entries));

  EXPECT_EQ(vpir_params.NumEntries(), num_entries);
  EXPECT_EQ(vpir_params.RlweParameters().Degree(), kDegree);
  EXPECT_EQ(vpir_params.PlaintextModulus(), 5);
}

TYPED_TEST(VpirParamsTest, InvalidNumEntries) {
  ASSERT_OK_AND_ASSIGN(auto rlwe_params, this->CreateRlweParams(kDegree));
  auto gadget_params = this->CreateGadgetParams();

  int num_entries = 0;

  auto vpir_params =
      VpirParams<TypeParam>::Create(rlwe_params, 5, gadget_params, num_entries);

  EXPECT_FALSE(vpir_params.ok());
  EXPECT_EQ(vpir_params.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(vpir_params.status().message(), HasSubstr("strictly positive"));
}

TYPED_TEST(VpirParamsTest, RowsNotDivisibleByRingDegree) {
  ASSERT_OK_AND_ASSIGN(auto rlwe_params, this->CreateRlweParams(kDegree));
  auto gadget_params = this->CreateGadgetParams();

  int num_entries = 1000;

  auto vpir_params =
      VpirParams<TypeParam>::Create(rlwe_params, 5, gadget_params, num_entries);

  EXPECT_FALSE(vpir_params.ok());
  EXPECT_EQ(vpir_params.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(vpir_params.status().message(),
              HasSubstr("divisible by the ring degree"));
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
