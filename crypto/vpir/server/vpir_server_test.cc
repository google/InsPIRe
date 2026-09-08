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

#include "crypto/vpir/server/vpir_server.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/vpir/client/vpir_client.h"
#include "crypto/vpir/server/vpir_preprocessing_server.h"
#include "crypto/vpir/vpir_params.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using ::testing::HasSubstr;
using ::testing::Test;
using ::rlwe::testing::StatusIs;

constexpr int kDegree = 32;
constexpr int kVariance = 4;
constexpr int kLogDigit = 2;
constexpr int kNumEntries = 1024;

struct Uint32Mod0 {
  using DbDataType = uint16_t;
  using CoeffType = uint32_t;
  using MatCoeffType = int16_t;
  static constexpr uint32_t kModulus = 0;
  static constexpr uint32_t kModulus1AfterSwitch = 1U << 31;
  static constexpr uint32_t kModulus2AfterSwitch = 1U << 30;
};

struct Uint32Mod30 {
  using DbDataType = uint16_t;
  using CoeffType = uint32_t;
  using MatCoeffType = int16_t;
  static constexpr uint32_t kModulus = 1U << 30;
  static constexpr uint32_t kModulus1AfterSwitch = 1U << 29;
  static constexpr uint32_t kModulus2AfterSwitch = 1U << 28;
};

struct Uint64Mod0 {
  using DbDataType = uint16_t;
  using CoeffType = uint64_t;
  using MatCoeffType = int16_t;
  static constexpr uint64_t kModulus = 0;
  static constexpr uint64_t kModulus1AfterSwitch = 1ULL << 61;
  static constexpr uint64_t kModulus2AfterSwitch = 1ULL << 60;
};

struct Uint64Mod62 {
  using DbDataType = uint16_t;
  using CoeffType = uint64_t;
  using MatCoeffType = int32_t;
  static constexpr uint64_t kModulus = 1ULL << 62;
  static constexpr uint64_t kModulus1AfterSwitch = 1ULL << 61;
  static constexpr uint64_t kModulus2AfterSwitch = 1ULL << 60;
};

template <typename PirTestType>
class VpirServerTest : public Test {
 public:
  using DbDataType = typename PirTestType::DbDataType;
  using CoeffType = typename PirTestType::CoeffType;
  using MatCoeffType = typename PirTestType::MatCoeffType;

 protected:
  VpirServerTest() = default;

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        auto rlwe_params,
        RlweParams<CoeffType>::Create(
            kDegree, PirTestType::kModulus, PirTestType::kModulus1AfterSwitch,
            PirTestType::kModulus2AfterSwitch, kVariance));
    rlwe_params_ =
        std::make_unique<RlweParams<CoeffType>>(std::move(rlwe_params));
    ASSERT_OK_AND_ASSIGN(auto params, this->CreateValidVpirParams());
    params_ = std::make_unique<VpirParams<CoeffType>>(std::move(params));

    const int d = params_->RlweParameters().Degree();
    const int n = params_->NumEntries();

    ASSERT_OK_AND_ASSIGN(auto db_prng, this->CreatePrng());
    db_raw_.resize(n * d);
    for (int i = 0; i < n * d; ++i) {
      ASSERT_OK_AND_ASSIGN(auto rand_val, db_prng->Rand8());
      db_raw_[i] = rand_val % params_->PlaintextModulus();
    }

    ASSERT_OK_AND_ASSIGN(
        auto preprocessing_server,
        (VpirPreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
            params_.get())));
    prng_seed_ = preprocessing_server->GetPrngSeed();

    ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                         preprocessing_server->Preprocess(db_raw_));
    ASSERT_OK_AND_ASSIGN(
        auto server, (VpirServer<DbDataType, CoeffType, MatCoeffType>::Create(
                         *params_, std::move(preprocessed_data))));
    server_ = std::move(server);
  }

  CoeffType GetPlaintextModulus() const { return 257; }

  GadgetParams GetPackGadgetParams() const {
    int log_q = PirTestType::kModulus == 0
                    ? sizeof(CoeffType) * 8
                    : absl::bit_width(PirTestType::kModulus - 1);
    return GadgetParams{/*log_digit=*/kLogDigit,
                        /*num_digits=*/(log_q + kLogDigit - 1) / kLogDigit - 1};
  }

  absl::StatusOr<VpirParams<CoeffType>> CreateValidVpirParams(
      bool local_finalize = false) const {
    return VpirParams<CoeffType>::Create(
        *rlwe_params_, GetPlaintextModulus(), GetPackGadgetParams(),
        kNumEntries, /*entry_size_multiple=*/1,
        /*z_interpolation_degree=*/1, /*z_pack_gadget_params=*/std::nullopt,
        /*z_poly_eval_gadget_params=*/std::nullopt,
        /*z_plaintext_modulus=*/std::nullopt,
        /*z_rlwe_params=*/std::nullopt, local_finalize);
  }

  absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() const {
    ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
    return ::rlwe::ChaChaPrng::Create(prng_seed);
  }

  absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng(
      absl::string_view prng_seed) const {
    return ::rlwe::ChaChaPrng::Create(prng_seed);
  }

  absl::StatusOr<std::unique_ptr<VpirClient<CoeffType>>> CreateVpirClient(
      const VpirParams<CoeffType>* params = nullptr) const {
    const auto& p = (params != nullptr) ? *params : *params_;
    ASSIGN_OR_RETURN(auto client_a_prng, CreatePrng(prng_seed_));
    ASSIGN_OR_RETURN(auto client_prng, CreatePrng());
    ASSIGN_OR_RETURN(auto client, VpirClient<CoeffType>::Create(
                                      p, std::move(client_a_prng),
                                      std::move(client_prng)));
    if (server_ != nullptr) {
      std::vector<PreprocessMatrixPackOutput<CoeffType>> prep_outputs;
      prep_outputs.reserve(server_->GetPreprocessedOutputs().size());
      for (const auto& out : server_->GetPreprocessedOutputs()) {
        prep_outputs.push_back(PreprocessMatrixPackOutput<CoeffType>{
            out.a_tilde_agg, Matrix<CoeffType>(), out.t_vec_h});
      }
      client->SetPreprocessedOutputs(std::move(prep_outputs));
    }
    return client;
  }

  std::unique_ptr<RlweParams<CoeffType>> rlwe_params_;
  std::unique_ptr<VpirParams<CoeffType>> params_;
  std::string prng_seed_;
  std::unique_ptr<VpirServer<DbDataType, CoeffType, MatCoeffType>> server_;
  std::vector<DbDataType> db_raw_;
};

using PirTestTypes =
    ::testing::Types<Uint32Mod0, Uint32Mod30, Uint64Mod0, Uint64Mod62>;
TYPED_TEST_SUITE(VpirServerTest, PirTestTypes);

TYPED_TEST(VpirServerTest, EndToEndTest) {
  using CoeffType = typename TypeParam::CoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int i = 0; i < num_entries; i += 53) {
    test_indices.push_back(i);
  }
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client, this->CreateVpirClient());
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));

    ASSERT_OK_AND_ASSIGN(auto response,
                         this->server_->ProcessResponse(request));

    ASSERT_OK_AND_ASSIGN(
        auto recovered_message,
        client->ProcessResponse(response, request, test_index));

    std::vector<CoeffType> expected_message(
        this->db_raw_.begin() + test_index * d,
        this->db_raw_.begin() + (test_index + 1) * d);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(VpirServerTest, EndToEndTestWithLargeEntrySizeMultiple) {
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();
  int entry_size_multiple = 3;

  ASSERT_OK_AND_ASSIGN(
      auto params,
      VpirParams<CoeffType>::Create(
          this->params_->RlweParameters(), this->GetPlaintextModulus(),
          this->GetPackGadgetParams(), num_entries, entry_size_multiple));

  ASSERT_OK_AND_ASSIGN(auto db_prng, this->CreatePrng());
  std::vector<typename TypeParam::DbDataType> db_raw(num_entries * d *
                                                     entry_size_multiple);
  for (int i = 0; i < num_entries * d * entry_size_multiple; ++i) {
    ASSERT_OK_AND_ASSIGN(auto rand_val, db_prng->Rand8());
    db_raw[i] = rand_val % params.PlaintextModulus();
  }

  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (VpirPreprocessingServer<typename TypeParam::DbDataType, CoeffType,
                               MatCoeffType>::Create(&params)));

  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       preprocessing_server->Preprocess(db_raw));

  ASSERT_OK_AND_ASSIGN(
      auto server,
      (VpirServer<typename TypeParam::DbDataType, CoeffType,
                  MatCoeffType>::Create(params, std::move(preprocessed_data))));

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int i = 0; i < num_entries; i += 53) {
    test_indices.push_back(i);
  }
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                         this->CreatePrng(preprocessing_server->GetPrngSeed()));
    ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
    ASSERT_OK_AND_ASSIGN(auto client, VpirClient<CoeffType>::Create(
                                          params, std::move(client_a_prng),
                                          std::move(client_prng)));
    std::vector<PreprocessMatrixPackOutput<CoeffType>> prep_outputs;
    prep_outputs.reserve(server->GetPreprocessedOutputs().size());
    for (const auto& out : server->GetPreprocessedOutputs()) {
      prep_outputs.push_back(PreprocessMatrixPackOutput<CoeffType>{
          out.a_tilde_agg, Matrix<CoeffType>(), {}});
    }
    client->SetPreprocessedOutputs(std::move(prep_outputs));
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));

    ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));

    ASSERT_OK_AND_ASSIGN(
        auto recovered_message,
        client->ProcessResponse(response, request, test_index));

    std::vector<CoeffType> expected_message(
        db_raw.begin() + test_index * d * entry_size_multiple,
        db_raw.begin() + (test_index + 1) * d * entry_size_multiple);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(VpirServerTest, EndToEndTestWithMatMul) {
  using CoeffType = typename TypeParam::CoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int i = 0; i < num_entries; i += 53) {
    test_indices.push_back(i);
  }
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client, this->CreateVpirClient());
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));

    ASSERT_OK_AND_ASSIGN(auto response,
                         this->server_->ProcessResponseWithMatMul(request));

    ASSERT_OK_AND_ASSIGN(
        auto recovered_message,
        client->ProcessResponse(response, request, test_index));

    std::vector<CoeffType> expected_message(
        this->db_raw_.begin() + test_index * d,
        this->db_raw_.begin() + (test_index + 1) * d);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(VpirServerTest, CompareProcessResponseVersions) {
  int num_entries = this->params_->NumEntries();
  std::vector<int> test_indices = {0, 10, num_entries - 1};

  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client, this->CreateVpirClient());
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));

    ASSERT_OK_AND_ASSIGN(auto response_default,
                         this->server_->ProcessResponse(request));
    ASSERT_OK_AND_ASSIGN(auto response_matmul,
                         this->server_->ProcessResponseWithMatMul(request));

    ASSERT_EQ(response_default.b_responses.size(),
              response_matmul.b_responses.size());
    for (size_t i = 0; i < response_default.b_responses.size(); ++i) {
      EXPECT_EQ(response_default.b_responses[i].Coeffs(),
                response_matmul.b_responses[i].Coeffs());
    }
  }
}

TYPED_TEST(VpirServerTest, GetProofPreprocessMaterialWorks) {
  const int kappa = 5;
  ASSERT_OK_AND_ASSIGN(auto client, this->CreateVpirClient());
  ASSERT_OK(client->GetProofPreprocessMaterial(kappa, *this->server_));
  EXPECT_EQ(client->GetPreprocessedOutputs().size(),
            static_cast<size_t>(this->params_->Ell()));
  EXPECT_EQ(client->GetProofZ().Rows(), kappa);
  EXPECT_EQ(client->GetProofZ().Cols(), this->params_->NumEntries());
}

TYPED_TEST(VpirServerTest, VerifiablePirEndToEndTest) {
  using CoeffType = typename TypeParam::CoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();
  const int kappa = 5;

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client, this->CreateVpirClient());
    ASSERT_OK(client->GetProofPreprocessMaterial(kappa, *this->server_));
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));

    ASSERT_OK_AND_ASSIGN(auto response,
                         this->server_->ProcessResponse(request));

    ASSERT_OK_AND_ASSIGN(
        auto recovered_message,
        client->ProcessResponse(response, request, test_index));

    std::vector<CoeffType> expected_message(
        this->db_raw_.begin() + test_index * d,
        this->db_raw_.begin() + (test_index + 1) * d);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(VpirServerTest, VerifiablePirEndToEndTestWithMatMul) {
  using CoeffType = typename TypeParam::CoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();
  const int kappa = 5;

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client, this->CreateVpirClient());
    ASSERT_OK(client->GetProofPreprocessMaterial(kappa, *this->server_));
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));

    ASSERT_OK_AND_ASSIGN(auto response,
                         this->server_->ProcessResponseWithMatMul(request));

    ASSERT_OK_AND_ASSIGN(
        auto recovered_message,
        client->ProcessResponse(response, request, test_index));

    std::vector<CoeffType> expected_message(
        this->db_raw_.begin() + test_index * d,
        this->db_raw_.begin() + (test_index + 1) * d);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(VpirServerTest, VerifiablePirDetectsCheatingServer) {
  using DbDataType = typename TypeParam::DbDataType;
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  ASSERT_OK_AND_ASSIGN(auto client, this->CreateVpirClient());
  const int kappa = 5;
  // The client fetches preprocessing proof material from the correct server.
  ASSERT_OK(client->GetProofPreprocessMaterial(kappa, *this->server_));

  std::vector<DbDataType> db_cheating = this->db_raw_;
  // Modify one entry in the database by a small amount.
  db_cheating[0] = (db_cheating[0] + 1) % this->params_->PlaintextModulus();

  ASSERT_OK_AND_ASSIGN(
      auto prep_server_cheating,
      (VpirPreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          this->params_.get())));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data_cheating,
                       prep_server_cheating->Preprocess(db_cheating));
  ASSERT_OK_AND_ASSIGN(
      auto server_cheating,
      (VpirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          *this->params_, std::move(preprocessed_data_cheating))));

  const int test_index = 0;
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));

  // The server cheats by responding using a different database.
  ASSERT_OK_AND_ASSIGN(auto response_cheating,
                       server_cheating->ProcessResponseWithMatMul(request));

  // The client should detect that the response doesn't match the preprocessed
  // material.
  EXPECT_THAT(client->ProcessResponse(response_cheating, request, test_index),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("[Z Z'] u != C x verification failed.")));
}

TEST(VpirHeterogeneousTest, Main32BitZ64BitVerificationSucceeds) {
  using MainDbDataType = uint8_t;
  using MainCoeffType = uint32_t;
  using MainMatCoeffType = int16_t;

  using ZDbDataType = uint8_t;
  using ZCoeffType = uint64_t;
  using ZMatCoeffType = int32_t;

  constexpr int d = 32;
  constexpr int n = 64;
  constexpr int ell = 2;
  constexpr int kappa = 5;

  constexpr uint32_t kMod = 1U << 30;
  constexpr uint64_t kZMod = 1ULL << 56;
  constexpr uint64_t kZPlaintextModulus = 1ULL << 19;

  ASSERT_OK_AND_ASSIGN(
      auto rlwe_params,
      RlweParams<MainCoeffType>::Create(d, kMod, 1U << 29, 1U << 28,
                                        kVariance));

  int log_p = absl::bit_width(static_cast<uint32_t>(kZPlaintextModulus));
  int log_d = absl::bit_width(static_cast<uint32_t>(d)) - 1;
  uint64_t mod1_after_switch = 1ULL << (log_p + log_d + 2);
  uint64_t mod2_after_switch = 1ULL << (log_p + 2);

  ASSERT_OK_AND_ASSIGN(
      auto z_rlwe_params,
      RlweParams<ZCoeffType>::Create(d, kZMod, mod1_after_switch,
                                     mod2_after_switch, kVariance));

  GadgetParams pack_gadget{/*log_digit=*/8, /*num_digits=*/3};
  GadgetParams z_pack_gadget{/*log_digit=*/19, /*num_digits=*/3};

  ASSERT_OK_AND_ASSIGN(
      auto params,
      (VpirParams<MainCoeffType, ZCoeffType>::Create(
          rlwe_params, /*plaintext_modulus=*/256, pack_gadget, n, ell,
          /*z_interpolation_degree=*/1, z_pack_gadget,
          /*z_poly_eval_gadget_params=*/std::nullopt, kZPlaintextModulus,
          z_rlwe_params)));

  std::vector<MainDbDataType> db_raw(n * d * ell);
  ASSERT_OK_AND_ASSIGN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(auto prng, ::rlwe::ChaChaPrng::Create(prng_seed));
  for (size_t i = 0; i < db_raw.size(); ++i) {
    ASSERT_OK_AND_ASSIGN(auto rand_val, prng->Rand8());
    db_raw[i] = rand_val % 256;
  }

  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (VpirPreprocessingServer<MainDbDataType, MainCoeffType, MainMatCoeffType,
                               ZDbDataType, ZCoeffType, ZMatCoeffType>::
           Create(&params)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       prep_server->Preprocess(db_raw));

  ASSERT_OK_AND_ASSIGN(
      auto server,
      (VpirServer<MainDbDataType, MainCoeffType, MainMatCoeffType, ZDbDataType,
                  ZCoeffType, ZMatCoeffType>::Create(params,
                                                     std::move(
                                                         preprocessed_data))));

  ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                       ::rlwe::ChaChaPrng::Create(server->GetPrngSeed()));
  ASSERT_OK_AND_ASSIGN(auto client_prng_seed,
                       ::rlwe::ChaChaPrng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(auto client_prng,
                       ::rlwe::ChaChaPrng::Create(client_prng_seed));
  ASSERT_OK_AND_ASSIGN(
      auto client,
      (VpirClient<MainCoeffType, ZCoeffType>::Create(params,
                                                     std::move(client_a_prng),
                                                     std::move(client_prng))));

  ASSERT_OK(client->GetProofPreprocessMaterial(kappa, *server));

  const int target_index = 5;
  ASSERT_OK_AND_ASSIGN(auto req, client->CreateRequest(target_index));
  ASSERT_OK_AND_ASSIGN(auto resp, server->ProcessResponse(req));
  ASSERT_OK_AND_ASSIGN(auto decrypted,
                       client->ProcessResponse(resp, req, target_index));

  std::vector<MainCoeffType> expected_message(
      db_raw.begin() + target_index * d * ell,
      db_raw.begin() + (target_index + 1) * d * ell);
  EXPECT_EQ(decrypted, expected_message);
}

TYPED_TEST(VpirServerTest, EndToEndTestLocalFinalize) {
  using DbDataType = typename TypeParam::DbDataType;
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();

  ASSERT_OK_AND_ASSIGN(auto params,
                       this->CreateValidVpirParams(/*local_finalize=*/true));

  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (VpirPreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          &params)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       prep_server->Preprocess(this->db_raw_));
  ASSERT_OK_AND_ASSIGN(
      auto server,
      (VpirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params, std::move(preprocessed_data))));

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int i = 0; i < num_entries; i += 53) {
    test_indices.push_back(i);
  }
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                         this->CreatePrng(prep_server->GetPrngSeed()));
    ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
    ASSERT_OK_AND_ASSIGN(auto client, VpirClient<CoeffType>::Create(
                                          params, std::move(client_a_prng),
                                          std::move(client_prng)));
    std::vector<PreprocessMatrixPackOutput<CoeffType>> prep_outputs;
    prep_outputs.reserve(server->GetPreprocessedOutputs().size());
    for (const auto& out : server->GetPreprocessedOutputs()) {
      prep_outputs.push_back(PreprocessMatrixPackOutput<CoeffType>{
          out.a_tilde_agg, Matrix<CoeffType>(), out.t_vec_h});
    }
    client->SetPreprocessedOutputs(std::move(prep_outputs));

    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
    EXPECT_EQ(request.packing_key.size(),
              static_cast<size_t>(params.PackGadgetParams().num_digits));

    ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));

    ASSERT_OK_AND_ASSIGN(
        auto recovered_message,
        client->ProcessResponse(response, request, test_index));

    std::vector<CoeffType> expected_message(
        this->db_raw_.begin() + test_index * d,
        this->db_raw_.begin() + (test_index + 1) * d);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(VpirServerTest, EndToEndTestWithMatMulLocalFinalize) {
  using DbDataType = typename TypeParam::DbDataType;
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();

  ASSERT_OK_AND_ASSIGN(auto params,
                       this->CreateValidVpirParams(/*local_finalize=*/true));

  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (VpirPreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          &params)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       prep_server->Preprocess(this->db_raw_));
  ASSERT_OK_AND_ASSIGN(
      auto server,
      (VpirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params, std::move(preprocessed_data))));

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int i = 0; i < num_entries; i += 53) {
    test_indices.push_back(i);
  }
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                         this->CreatePrng(prep_server->GetPrngSeed()));
    ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
    ASSERT_OK_AND_ASSIGN(auto client, VpirClient<CoeffType>::Create(
                                          params, std::move(client_a_prng),
                                          std::move(client_prng)));
    std::vector<PreprocessMatrixPackOutput<CoeffType>> prep_outputs;
    prep_outputs.reserve(server->GetPreprocessedOutputs().size());
    for (const auto& out : server->GetPreprocessedOutputs()) {
      prep_outputs.push_back(PreprocessMatrixPackOutput<CoeffType>{
          out.a_tilde_agg, Matrix<CoeffType>(), out.t_vec_h});
    }
    client->SetPreprocessedOutputs(std::move(prep_outputs));

    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
    EXPECT_EQ(request.packing_key.size(),
              static_cast<size_t>(params.PackGadgetParams().num_digits));

    ASSERT_OK_AND_ASSIGN(auto response,
                         server->ProcessResponseWithMatMul(request));

    ASSERT_OK_AND_ASSIGN(
        auto recovered_message,
        client->ProcessResponse(response, request, test_index));

    std::vector<CoeffType> expected_message(
        this->db_raw_.begin() + test_index * d,
        this->db_raw_.begin() + (test_index + 1) * d);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(VpirServerTest, VerifiablePirEndToEndTestLocalFinalize) {
  using DbDataType = typename TypeParam::DbDataType;
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();
  const int kappa = 5;

  ASSERT_OK_AND_ASSIGN(auto params,
                       this->CreateValidVpirParams(/*local_finalize=*/true));

  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (VpirPreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          &params)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       prep_server->Preprocess(this->db_raw_));
  ASSERT_OK_AND_ASSIGN(
      auto server,
      (VpirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params, std::move(preprocessed_data))));

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                         this->CreatePrng(prep_server->GetPrngSeed()));
    ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
    ASSERT_OK_AND_ASSIGN(auto client, VpirClient<CoeffType>::Create(
                                          params, std::move(client_a_prng),
                                          std::move(client_prng)));

    ASSERT_OK(client->GetProofPreprocessMaterial(kappa, *server));
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
    EXPECT_EQ(request.packing_key.size(),
              static_cast<size_t>(params.PackGadgetParams().num_digits));

    ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));

    ASSERT_OK_AND_ASSIGN(
        auto recovered_message,
        client->ProcessResponse(response, request, test_index));

    std::vector<CoeffType> expected_message(
        this->db_raw_.begin() + test_index * d,
        this->db_raw_.begin() + (test_index + 1) * d);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(VpirServerTest, VerifiablePirEndToEndTestWithMatMulLocalFinalize) {
  using DbDataType = typename TypeParam::DbDataType;
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();
  const int kappa = 5;

  ASSERT_OK_AND_ASSIGN(auto params,
                       this->CreateValidVpirParams(/*local_finalize=*/true));

  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (VpirPreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          &params)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       prep_server->Preprocess(this->db_raw_));
  ASSERT_OK_AND_ASSIGN(
      auto server,
      (VpirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params, std::move(preprocessed_data))));

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                         this->CreatePrng(prep_server->GetPrngSeed()));
    ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
    ASSERT_OK_AND_ASSIGN(auto client, VpirClient<CoeffType>::Create(
                                          params, std::move(client_a_prng),
                                          std::move(client_prng)));

    ASSERT_OK(client->GetProofPreprocessMaterial(kappa, *server));
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
    EXPECT_EQ(request.packing_key.size(),
              static_cast<size_t>(params.PackGadgetParams().num_digits));

    ASSERT_OK_AND_ASSIGN(auto response,
                         server->ProcessResponseWithMatMul(request));

    ASSERT_OK_AND_ASSIGN(
        auto recovered_message,
        client->ProcessResponse(response, request, test_index));

    std::vector<CoeffType> expected_message(
        this->db_raw_.begin() + test_index * d,
        this->db_raw_.begin() + (test_index + 1) * d);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(VpirServerTest, VerifiablePirDetectsCheatingServerLocalFinalize) {
  using DbDataType = typename TypeParam::DbDataType;
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;

  ASSERT_OK_AND_ASSIGN(auto params,
                       this->CreateValidVpirParams(/*local_finalize=*/true));

  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (VpirPreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          &params)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       prep_server->Preprocess(this->db_raw_));
  ASSERT_OK_AND_ASSIGN(
      auto server,
      (VpirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params, std::move(preprocessed_data))));

  ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                       this->CreatePrng(prep_server->GetPrngSeed()));
  ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
  ASSERT_OK_AND_ASSIGN(auto client, VpirClient<CoeffType>::Create(
                                        params, std::move(client_a_prng),
                                        std::move(client_prng)));
  const int kappa = 5;
  ASSERT_OK(client->GetProofPreprocessMaterial(kappa, *server));

  std::vector<DbDataType> db_cheating = this->db_raw_;
  db_cheating[0] = (db_cheating[0] + 1) % params.PlaintextModulus();

  ASSERT_OK_AND_ASSIGN(
      auto prep_server_cheating,
      (VpirPreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          &params)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data_cheating,
                       prep_server_cheating->Preprocess(db_cheating));
  ASSERT_OK_AND_ASSIGN(
      auto server_cheating,
      (VpirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params, std::move(preprocessed_data_cheating))));

  const int test_index = 0;
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));

  ASSERT_OK_AND_ASSIGN(auto response_cheating,
                       server_cheating->ProcessResponseWithMatMul(request));

  EXPECT_THAT(client->ProcessResponse(response_cheating, request, test_index),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("[Z Z'] u != C x verification failed.")));
}

TEST(VpirHeterogeneousTest, Main32BitZ64BitLocalFinalizeVerificationSucceeds) {
  using MainDbDataType = uint8_t;
  using MainCoeffType = uint32_t;
  using MainMatCoeffType = int16_t;

  using ZDbDataType = uint8_t;
  using ZCoeffType = uint64_t;
  using ZMatCoeffType = int32_t;

  constexpr int d = 32;
  constexpr int n = 64;
  constexpr int ell = 2;
  constexpr int kappa = 5;

  constexpr uint32_t kMod = 1U << 30;
  constexpr uint64_t kZMod = 1ULL << 56;
  constexpr uint64_t kZPlaintextModulus = 1ULL << 19;

  ASSERT_OK_AND_ASSIGN(
      auto rlwe_params,
      RlweParams<MainCoeffType>::Create(d, kMod, 1U << 29, 1U << 28,
                                        kVariance));

  int log_p = absl::bit_width(static_cast<uint32_t>(kZPlaintextModulus));
  int log_d = absl::bit_width(static_cast<uint32_t>(d)) - 1;
  uint64_t mod1_after_switch = 1ULL << (log_p + log_d + 2);
  uint64_t mod2_after_switch = 1ULL << (log_p + 2);

  ASSERT_OK_AND_ASSIGN(
      auto z_rlwe_params,
      RlweParams<ZCoeffType>::Create(d, kZMod, mod1_after_switch,
                                     mod2_after_switch, kVariance));

  GadgetParams pack_gadget{/*log_digit=*/8, /*num_digits=*/3};
  GadgetParams z_pack_gadget{/*log_digit=*/19, /*num_digits=*/3};

  ASSERT_OK_AND_ASSIGN(
      auto params,
      (VpirParams<MainCoeffType, ZCoeffType>::Create(
          rlwe_params, /*plaintext_modulus=*/256, pack_gadget, n, ell,
          /*z_interpolation_degree=*/1, z_pack_gadget,
          /*z_poly_eval_gadget_params=*/std::nullopt, kZPlaintextModulus,
          z_rlwe_params, /*local_finalize=*/true)));

  std::vector<MainDbDataType> db_raw(n * d * ell);
  ASSERT_OK_AND_ASSIGN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(auto prng, ::rlwe::ChaChaPrng::Create(prng_seed));
  for (size_t i = 0; i < db_raw.size(); ++i) {
    ASSERT_OK_AND_ASSIGN(auto rand_val, prng->Rand8());
    db_raw[i] = rand_val % 256;
  }

  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (VpirPreprocessingServer<MainDbDataType, MainCoeffType, MainMatCoeffType,
                               ZDbDataType, ZCoeffType, ZMatCoeffType>::
           Create(&params)));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       prep_server->Preprocess(db_raw));

  ASSERT_OK_AND_ASSIGN(
      auto server,
      (VpirServer<MainDbDataType, MainCoeffType, MainMatCoeffType, ZDbDataType,
                  ZCoeffType, ZMatCoeffType>::Create(params,
                                                     std::move(
                                                         preprocessed_data))));

  ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                       ::rlwe::ChaChaPrng::Create(server->GetPrngSeed()));
  ASSERT_OK_AND_ASSIGN(auto client_prng_seed,
                       ::rlwe::ChaChaPrng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(auto client_prng,
                       ::rlwe::ChaChaPrng::Create(client_prng_seed));
  ASSERT_OK_AND_ASSIGN(
      auto client,
      (VpirClient<MainCoeffType, ZCoeffType>::Create(params,
                                                     std::move(client_a_prng),
                                                     std::move(client_prng))));

  ASSERT_OK(client->GetProofPreprocessMaterial(kappa, *server));

  const int target_index = 5;
  ASSERT_OK_AND_ASSIGN(auto req, client->CreateRequest(target_index));
  EXPECT_EQ(req.packing_key.size(),
            static_cast<size_t>(params.PackGadgetParams().num_digits));

  ASSERT_OK_AND_ASSIGN(auto resp, server->ProcessResponse(req));
  ASSERT_OK_AND_ASSIGN(auto decrypted,
                       client->ProcessResponse(resp, req, target_index));

  std::vector<MainCoeffType> expected_message(
      db_raw.begin() + target_index * d * ell,
      db_raw.begin() + (target_index + 1) * d * ell);
  EXPECT_EQ(decrypted, expected_message);
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
