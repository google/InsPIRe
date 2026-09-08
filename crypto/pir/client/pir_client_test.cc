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

#include "crypto/pir/client/pir_client.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/pir/pir_params.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using ::testing::HasSubstr;
using ::testing::Test;
using ::rlwe::testing::StatusIs;

constexpr int kDegree = 1024;
constexpr int kVariance = 8;
constexpr int kLogDigit0 = 2;
constexpr int kNumDigits0 = 4;
constexpr int kLogDigit1 = 3;
constexpr int kNumDigits1 = 3;

template <typename CoeffType>
class PirClientTest : public Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        auto rlwe_params,
        RlweParams<CoeffType>::Create(kDegree, GetModulus(), kVariance));
    rlwe_params_ =
        std::make_unique<RlweParams<CoeffType>>(std::move(rlwe_params));
    ASSERT_OK_AND_ASSIGN(a_component_prng_, this->CreatePrng());
    ASSERT_OK_AND_ASSIGN(prng_, this->CreatePrng());
  }

  CoeffType GetModulus() const {
    if (sizeof(CoeffType) == 4) {
      return 0;  // uint32_t uses implicit modulus 2^32 (0)
    } else {
      return 0;  // uint64_t uses implicit modulus 2^64 (0)
    }
  }

  CoeffType GetPlaintextModulus() const { return 256; }

  GadgetParams GetGadgetParams0() const {
    return GadgetParams{/*log_digit=*/kLogDigit0, /*num_digits=*/kNumDigits0};
  }

  GadgetParams GetGadgetParams1() const {
    return GadgetParams{/*log_digit=*/kLogDigit1, /*num_digits=*/kNumDigits1};
  }

  absl::StatusOr<PirParams<CoeffType>> CreateValidPirParams(
      int interpolation_degree = 2) const {
    int num_entries = 4096;
    return PirParams<CoeffType>::Create(*rlwe_params_, GetPlaintextModulus(),
                                        GetGadgetParams0(), GetGadgetParams1(),
                                        num_entries, interpolation_degree);
  }

  absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() const {
    ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
    return ::rlwe::ChaChaPrng::Create(prng_seed);
  }

  std::unique_ptr<RlweParams<CoeffType>> rlwe_params_;
  std::unique_ptr<::rlwe::SecurePrng> a_component_prng_;
  std::unique_ptr<::rlwe::SecurePrng> prng_;
};

using CoeffTypes = ::testing::Types<uint32_t, uint64_t>;
TYPED_TEST_SUITE(PirClientTest, CoeffTypes);

TYPED_TEST(PirClientTest, CreateFailsWithNullPrng) {
  ASSERT_OK_AND_ASSIGN(auto params, this->CreateValidPirParams());
  EXPECT_THAT(PirClient<TypeParam>::Create(params, nullptr, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("a_component_prng must not be null")));

  EXPECT_THAT(PirClient<TypeParam>::Create(
                  params, std::move(this->a_component_prng_), nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("prng must not be null")));
}

TYPED_TEST(PirClientTest, CreateRequestFailsWithIndexOutOfBounds) {
  ASSERT_OK_AND_ASSIGN(auto params, this->CreateValidPirParams());
  ASSERT_OK_AND_ASSIGN(
      auto client,
      PirClient<TypeParam>::Create(params, std::move(this->a_component_prng_),
                                   std::move(this->prng_)));

  EXPECT_THAT(client->CreateRequest(-1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index out of bounds")));

  EXPECT_THAT(client->CreateRequest(params.NumEntries()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index out of bounds")));
}

TYPED_TEST(PirClientTest, CreateRequestGeneratesValidCiphertexts) {
  ASSERT_OK_AND_ASSIGN(auto params, this->CreateValidPirParams());

  std::vector<int> test_indices = {0, 1, 35, 100, params.NumEntries() - 1};

  for (int index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
    ASSERT_OK_AND_ASSIGN(auto a_component_prng,
                         ::rlwe::ChaChaPrng::Create(prng_seed));
    ASSERT_OK_AND_ASSIGN(auto a_component_prng_clone,
                         ::rlwe::ChaChaPrng::Create(prng_seed));
    ASSERT_OK_AND_ASSIGN(auto prng, this->CreatePrng());

    ASSERT_OK_AND_ASSIGN(
        auto client, PirClient<TypeParam>::Create(
                         params, std::move(a_component_prng), std::move(prng)));

    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(index));

    const auto& rlwe_params = params.RlweParameters();
    int d = rlwe_params.Degree();
    int t = params.InterpolationDegree();

    int expected_first_dim = params.InterpolatedRows();
    EXPECT_EQ(request.first_dimension_query.size(), expected_first_dim);

    int expected_second_dim = 2 * params.PolyEvalGadgetParams().num_digits;
    EXPECT_EQ(request.second_dimension_query.size(), expected_second_dim);

    int expected_packing = 2 * params.PackGadgetParams().num_digits;
    EXPECT_EQ(request.packing_key.size(), expected_packing);

    int num_samples = params.NumFirstDimSamples();
    ASSERT_OK_AND_ASSIGN(
        auto a_first,
        SampleAComponents(rlwe_params, num_samples, *a_component_prng_clone));

    int entries_per_row = t;
    int row_index = index / entries_per_row;
    int expected_i = row_index / d;
    int expected_j = row_index % d;

    for (int idx = 0; idx < num_samples; ++idx) {
      int count = std::min(d, expected_first_dim - idx * d);
      std::vector<TypeParam> b_coeffs(d, 0);
      for (int k = 0; k < count; ++k) {
        b_coeffs[k] = request.first_dimension_query[idx * d + k];
      }
      ASSERT_OK_AND_ASSIGN(auto b_poly,
                           Polynomial<TypeParam>::Create(std::move(b_coeffs)));
      RlweCiphertext<TypeParam> ciphertext{std::move(a_first[idx]),
                                           std::move(b_poly)};
      ASSERT_OK_AND_ASSIGN(auto plaintext,
                           Decrypt(rlwe_params, params.PlaintextModulus(),
                                   ciphertext, client->GetSecretKey()));

      std::vector<TypeParam> expected_message(d, 0);
      if (idx == expected_i) {
        expected_message[expected_j] = 1;
      }
      for (int k = 0; k < count; ++k) {
        EXPECT_EQ(plaintext[k], expected_message[k]);
      }
    }
  }
}

TYPED_TEST(PirClientTest,
           CreateRequestForVectorFailsForInvalidInterpolationDegree) {
  ASSERT_OK_AND_ASSIGN(auto params,
                       this->CreateValidPirParams(/*interpolation_degree=*/2));
  ASSERT_OK_AND_ASSIGN(
      auto client,
      PirClient<TypeParam>::Create(params, std::move(this->a_component_prng_),
                                   std::move(this->prng_)));
  std::vector<TypeParam> vec(kDegree, 1);
  EXPECT_THAT(client->CreateRequestForVector(vec),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("interpolation degree 1")));
}

TYPED_TEST(PirClientTest, CreateRequestForVectorFailsForInvalidVectorSize) {
  ASSERT_OK_AND_ASSIGN(auto params,
                       this->CreateValidPirParams(/*interpolation_degree=*/1));
  ASSERT_OK_AND_ASSIGN(
      auto client,
      PirClient<TypeParam>::Create(params, std::move(this->a_component_prng_),
                                   std::move(this->prng_)));

  EXPECT_THAT(client->CreateRequestForVector({}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("non-zero multiple of ring degree d")));

  std::vector<TypeParam> vec_invalid_size(kDegree - 1, 1);
  EXPECT_THAT(client->CreateRequestForVector(vec_invalid_size),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("non-zero multiple of ring degree d")));
}

TYPED_TEST(PirClientTest, CreateRequestForVectorSucceedsForMultiplesOfDegree) {
  ASSERT_OK_AND_ASSIGN(auto params,
                       this->CreateValidPirParams(/*interpolation_degree=*/1));

  for (int num_blocks : {1, 2, 3}) {
    ASSERT_OK_AND_ASSIGN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
    ASSERT_OK_AND_ASSIGN(auto a_component_prng,
                         ::rlwe::ChaChaPrng::Create(prng_seed));
    ASSERT_OK_AND_ASSIGN(auto a_component_prng_clone,
                         ::rlwe::ChaChaPrng::Create(prng_seed));
    ASSERT_OK_AND_ASSIGN(auto prng, this->CreatePrng());

    ASSERT_OK_AND_ASSIGN(
        auto client, PirClient<TypeParam>::Create(
                         params, std::move(a_component_prng), std::move(prng)));

    std::vector<TypeParam> vec(num_blocks * kDegree);
    for (size_t i = 0; i < vec.size(); ++i) {
      vec[i] = (i + 1) % params.PlaintextModulus();
    }

    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequestForVector(vec));

    const auto& rlwe_params = params.RlweParameters();
    EXPECT_EQ(request.first_dimension_query.size(), num_blocks * kDegree);
    EXPECT_TRUE(request.second_dimension_query.empty());

    ASSERT_OK_AND_ASSIGN(
        auto a_first,
        SampleAComponents(rlwe_params, num_blocks, *a_component_prng_clone));

    for (int idx = 0; idx < num_blocks; ++idx) {
      std::vector<TypeParam> b_chunk(
          request.first_dimension_query.begin() + idx * kDegree,
          request.first_dimension_query.begin() + (idx + 1) * kDegree);
      ASSERT_OK_AND_ASSIGN(auto b_poly,
                           Polynomial<TypeParam>::Create(std::move(b_chunk)));
      RlweCiphertext<TypeParam> ciphertext{std::move(a_first[idx]),
                                           std::move(b_poly)};
      ASSERT_OK_AND_ASSIGN(auto plaintext,
                           Decrypt(rlwe_params, params.PlaintextModulus(),
                                   ciphertext, client->GetSecretKey()));

      std::vector<TypeParam> expected_chunk(vec.begin() + idx * kDegree,
                                            vec.begin() + (idx + 1) * kDegree);
      EXPECT_EQ(plaintext, expected_chunk);
    }
  }
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
