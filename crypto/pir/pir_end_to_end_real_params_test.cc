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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_server.h"
#include "crypto/pir/server/preprocessing_server.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "crypto/status_macros.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace pir {
namespace {

using CoeffType = uint64_t;
using MatCoeffType = int32_t;
using DbDataType = uint16_t;

struct PirTestParams {
  // Database parameters
  int num_entries = 4096;
  int entry_size_multiple = 1;
  int degree = 2048;
  // Database size = num_entries * degree * entry_size_multiple * 2 bytes

  // Crypto parameters
  int variance = 6;
  int plaintext_modulus = 65535;
  uint64_t modulus = 1ULL << 56;

  // Tunable parameters for performance
  int pack_log_digit = 19;
  int pack_num_digits = 2;
  int eval_log_digit = 19;
  int eval_num_digits = 3;
  int interpolation_degree = 2;

  uint64_t mod1_after_switch;
  uint64_t mod2_after_switch;

  PirTestParams() {
    int log_p = absl::bit_width(static_cast<uint32_t>(plaintext_modulus));
    int log_d = absl::bit_width(static_cast<uint32_t>(degree)) - 1;
    mod1_after_switch = 1ULL << (log_p + log_d + 2);
    mod2_after_switch = 1ULL << (log_p + 2);
  }
};

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng(
    absl::string_view prng_seed) {
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() {
  ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

class PirEndToEndTest : public ::testing::Test {
 protected:
  absl::StatusOr<std::unique_ptr<PirParams<CoeffType>>> CreateParams(
      const PirTestParams& p) {
    ASSIGN_OR_RETURN(
        auto rlwe_params,
        RlweParams<CoeffType>::Create(p.degree, p.modulus, p.mod1_after_switch,
                                      p.mod2_after_switch, p.variance));

    GadgetParams pack_gadget{p.pack_log_digit, p.pack_num_digits};
    GadgetParams poly_eval_gadget{p.eval_log_digit, p.eval_num_digits};

    ASSIGN_OR_RETURN(
        auto params,
        PirParams<CoeffType>::Create(
            rlwe_params, p.plaintext_modulus, pack_gadget, poly_eval_gadget,
            p.num_entries, p.interpolation_degree, p.entry_size_multiple));
    return std::make_unique<PirParams<CoeffType>>(params);
  }
};

TEST_F(PirEndToEndTest, ExecutesSuccessfullyAndMatchesExpectedOutput) {
  PirTestParams p;
  ASSERT_OK_AND_ASSIGN(auto params, CreateParams(p));

  // 1. Generate a random database
  std::vector<DbDataType> db_raw(static_cast<size_t>(p.num_entries) * p.degree *
                                 p.entry_size_multiple);
  ASSERT_OK_AND_ASSIGN(auto db_prng, CreatePrng());
  for (size_t i = 0; i < db_raw.size(); ++i) {
    db_raw[i] = (*db_prng->Rand64()) % p.plaintext_modulus;
  }

  // 2. Create server and preprocess the database
  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params.get())));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data, prep_server->Preprocess(db_raw));
  auto prng_seed = prep_server->GetPrngSeed();

  // Directly pass to online server (in memory)
  ASSERT_OK_AND_ASSIGN(auto server,
                       (PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
                           *params, std::move(preprocessed_data))));

  // 3. Client setup
  ASSERT_OK_AND_ASSIGN(auto client_a_prng, CreatePrng(prng_seed));
  ASSERT_OK_AND_ASSIGN(auto client_prng, CreatePrng());
  ASSERT_OK_AND_ASSIGN(auto client, PirClient<CoeffType>::Create(
                                        *params, std::move(client_a_prng),
                                        std::move(client_prng)));

  // Try out a random index
  int target_index = (*db_prng->Rand8()) % p.num_entries;

  // 4. Request / Response Phase
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(target_index));
  ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));
  ASSERT_OK_AND_ASSIGN(auto extracted_data, client->ProcessResponse(response));

  // 5. Verification Phase
  ASSERT_EQ(extracted_data.size(),
            static_cast<size_t>(p.degree) * p.entry_size_multiple);
  size_t offset = target_index * p.degree * p.entry_size_multiple;
  for (size_t i = 0; i < extracted_data.size(); ++i) {
    EXPECT_EQ(extracted_data[i], db_raw[offset + i])
        << "Mismatch at fetched index " << i << " (DB offset: " << offset + i
        << ")";
  }
}

TEST_F(PirEndToEndTest,
       ExecutesSuccessfullyWithNonRingDegreeMultipleNumEntries) {
  PirTestParams p;
  // 1000 is divisible by interpolation_degree = 2, but 500 is not divisible by
  // degree = 2048.
  p.num_entries = 1000;
  ASSERT_OK_AND_ASSIGN(auto params, CreateParams(p));

  // 1. Generate a random database
  std::vector<DbDataType> db_raw(static_cast<size_t>(p.num_entries) * p.degree *
                                 p.entry_size_multiple);
  ASSERT_OK_AND_ASSIGN(auto db_prng, CreatePrng());
  for (size_t i = 0; i < db_raw.size(); ++i) {
    db_raw[i] = (*db_prng->Rand64()) % p.plaintext_modulus;
  }

  // 2. Create server and preprocess the database
  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params.get())));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data, prep_server->Preprocess(db_raw));
  auto prng_seed = prep_server->GetPrngSeed();

  // Directly pass to online server (in memory)
  ASSERT_OK_AND_ASSIGN(auto server,
                       (PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
                           *params, std::move(preprocessed_data))));

  // Test boundaries and random index
  std::vector<int> test_indices = {0, 1, 499, p.num_entries - 1};
  for (int target_index : test_indices) {
    // 3. Client setup
    ASSERT_OK_AND_ASSIGN(auto client_a_prng, CreatePrng(prng_seed));
    ASSERT_OK_AND_ASSIGN(auto client_prng, CreatePrng());
    ASSERT_OK_AND_ASSIGN(auto client, PirClient<CoeffType>::Create(
                                          *params, std::move(client_a_prng),
                                          std::move(client_prng)));

    // 4. Request / Response Phase
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(target_index));
    EXPECT_EQ(request.first_dimension_query.size(),
              p.num_entries / p.interpolation_degree);
    ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));
    ASSERT_OK_AND_ASSIGN(auto extracted_data,
                         client->ProcessResponse(response));

    // 5. Verification Phase
    ASSERT_EQ(extracted_data.size(),
              static_cast<size_t>(p.degree) * p.entry_size_multiple);
    size_t offset = target_index * p.degree * p.entry_size_multiple;
    for (size_t i = 0; i < extracted_data.size(); ++i) {
      EXPECT_EQ(extracted_data[i], db_raw[offset + i])
          << "Mismatch at fetched index " << i << " (DB offset: " << offset + i
          << ")";
    }
  }
}

}  // namespace
}  // namespace pir
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
