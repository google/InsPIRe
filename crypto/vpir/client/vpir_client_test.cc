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

#include "crypto/vpir/client/vpir_client.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/matrix.h"
#include "crypto/vpir/server/vpir_preprocessing_server.h"
#include "crypto/vpir/server/vpir_server.h"
#include "crypto/vpir/vpir_params.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/log/log.h"
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

constexpr int kDegree = 2048;
constexpr int kVariance = 6;

template <typename CoeffType>
class VpirClientTest : public Test {
 protected:
  void SetUp() override {
    auto rlwe_params_or = []() -> absl::StatusOr<RlweParams<CoeffType>> {
      if constexpr (sizeof(CoeffType) == 8) {
        return RlweParams<CoeffType>::Create(kDegree, 1ULL << 62, 1ULL << 61,
                                             1ULL << 60, kVariance);
      } else {
        return RlweParams<CoeffType>::Create(kDegree, 0, kVariance);
      }
    }();
    ASSERT_OK_AND_ASSIGN(auto rlwe_params, rlwe_params_or);
    rlwe_params_ =
        std::make_unique<RlweParams<CoeffType>>(std::move(rlwe_params));
    ASSERT_OK_AND_ASSIGN(a_component_prng_, this->CreatePrng());
    ASSERT_OK_AND_ASSIGN(prng_, this->CreatePrng());
  }

  CoeffType GetPlaintextModulus() const { return 256; }

  GadgetParams GetPackGadgetParams() const {
    if (sizeof(CoeffType) == 4) {
      return GadgetParams{/*log_digit=*/4, /*num_digits=*/8};  // 32 bits
    } else {
      return GadgetParams{/*log_digit=*/8, /*num_digits=*/8};  // 64 bits
    }
  }

  absl::StatusOr<VpirParams<CoeffType>> CreateValidVpirParams(
      bool local_finalize = false) const {
    int num_entries = 4096;
    return VpirParams<CoeffType>::Create(
        *rlwe_params_, GetPlaintextModulus(), GetPackGadgetParams(),
        num_entries, /*entry_size_multiple=*/1,
        /*z_interpolation_degree=*/1, /*z_pack_gadget_params=*/std::nullopt,
        /*z_poly_eval_gadget_params=*/std::nullopt,
        /*z_plaintext_modulus=*/std::nullopt,
        /*z_rlwe_params=*/std::nullopt, local_finalize);
  }

  absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() const {
    ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
    return ::rlwe::ChaChaPrng::Create(prng_seed);
  }

  std::unique_ptr<RlweParams<CoeffType>> rlwe_params_;
  std::unique_ptr<::rlwe::SecurePrng> a_component_prng_;
  std::unique_ptr<::rlwe::SecurePrng> prng_;
};

using CoeffTypes = ::testing::Types<uint64_t>;
TYPED_TEST_SUITE(VpirClientTest, CoeffTypes);

TYPED_TEST(VpirClientTest, CreateFailsWithNullPrng) {
  ASSERT_OK_AND_ASSIGN(auto params, this->CreateValidVpirParams());
  EXPECT_THAT(VpirClient<TypeParam>::Create(params, nullptr, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("a_component_prng must not be null")));

  EXPECT_THAT(VpirClient<TypeParam>::Create(
                  params, std::move(this->a_component_prng_), nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("prng must not be null")));
}

TYPED_TEST(VpirClientTest, CreateRequestFailsWithIndexOutOfBounds) {
  ASSERT_OK_AND_ASSIGN(auto params, this->CreateValidVpirParams());
  ASSERT_OK_AND_ASSIGN(
      auto client,
      VpirClient<TypeParam>::Create(params, std::move(this->a_component_prng_),
                                    std::move(this->prng_)));

  EXPECT_THAT(client->CreateRequest(-1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index out of bounds")));

  EXPECT_THAT(client->CreateRequest(params.NumEntries()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Index out of bounds")));
}

TYPED_TEST(VpirClientTest, CreateRequestGeneratesValidCiphertexts) {
  ASSERT_OK_AND_ASSIGN(auto params, this->CreateValidVpirParams());

  std::vector<int> test_indices = {0, 1, 35, 100, params.NumEntries() - 1};

  for (int index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
    ASSERT_OK_AND_ASSIGN(auto a_component_prng,
                         ::rlwe::ChaChaPrng::Create(prng_seed));
    ASSERT_OK_AND_ASSIGN(auto a_component_prng_clone,
                         ::rlwe::ChaChaPrng::Create(prng_seed));
    ASSERT_OK_AND_ASSIGN(auto prng, this->CreatePrng());

    ASSERT_OK_AND_ASSIGN(
        auto client, VpirClient<TypeParam>::Create(
                         params, std::move(a_component_prng), std::move(prng)));

    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(index));

    const auto& rlwe_params = params.RlweParameters();
    int d = rlwe_params.Degree();

    int expected_first_dim = params.NumEntries() / d;
    EXPECT_EQ(request.first_dimension_query.size(), expected_first_dim);

    int expected_packing = 2 * params.PackGadgetParams().num_digits;
    EXPECT_EQ(request.packing_key.size(), expected_packing);

    int num_samples = params.NumEntries() / d;
    ASSERT_OK_AND_ASSIGN(
        auto a_first,
        SampleAComponents(rlwe_params, num_samples, *a_component_prng_clone));

    int row_index = index;
    int expected_i = row_index / d;
    int expected_j = row_index % d;

    for (int idx = 0; idx < request.first_dimension_query.size(); ++idx) {
      RlweCiphertext<TypeParam> ciphertext{
          std::move(a_first[idx]),
          std::move(request.first_dimension_query[idx])};
      ASSERT_OK_AND_ASSIGN(auto plaintext,
                           Decrypt(rlwe_params, params.PlaintextModulus(),
                                   ciphertext, client->GetSecretKey()));

      std::vector<TypeParam> expected_message(d, 0);
      if (idx == expected_i) {
        expected_message[expected_j] = 1;
      }
      EXPECT_EQ(plaintext, expected_message);
    }
  }
}

TYPED_TEST(VpirClientTest, GetProofPreprocessMaterialWorks) {
  ASSERT_OK_AND_ASSIGN(auto params, this->CreateValidVpirParams());
  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (VpirPreprocessingServer<uint16_t, TypeParam, TypeParam>::Create(
          &params)));
  std::vector<uint16_t> db(params.NumEntries() *
                           params.RlweParameters().Degree() *
                           params.EntrySizeMultiple());
  std::mt19937 rng(42);
  for (auto& val : db) {
    val = rng() % 256;
  }
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data, prep_server->Preprocess(db));

  ASSERT_OK_AND_ASSIGN(auto server,
                       (VpirServer<uint16_t, TypeParam, TypeParam>::Create(
                           params, std::move(preprocessed_data))));

  ASSERT_OK_AND_ASSIGN(auto a_prng,
                       ::rlwe::ChaChaPrng::Create(server->GetPrngSeed()));
  ASSERT_OK_AND_ASSIGN(auto prng, this->CreatePrng());
  ASSERT_OK_AND_ASSIGN(auto client,
                       VpirClient<TypeParam>::Create(params, std::move(a_prng),
                                                     std::move(prng)));

  const int kappa = 10;
  absl::Status status = client->GetProofPreprocessMaterial(kappa, *server);
  if (client->GetProofZ().Rows() > 0) {
    const auto& c_mat = client->GetProofC();
    const int db_cols = params.NumEntries();
    std::vector<TypeParam> expected_z_data(c_mat.Rows() * db_cols, 0);
    for (int i = 0; i < c_mat.Rows(); ++i) {
      for (int j = 0; j < db_cols; ++j) {
        TypeParam sum = 0;
        for (int k = 0; k < c_mat.Cols(); ++k) {
          sum += c_mat.Data()[i * c_mat.Cols() + k] *
                 static_cast<TypeParam>(db[j * c_mat.Cols() + k]);
        }
        expected_z_data[i * db_cols + j] = sum;
      }
    }
    Matrix<TypeParam> expected_z(std::move(expected_z_data), c_mat.Rows(),
                                 db_cols);
    const TypeParam p_z = params.ZPirParams().PlaintextModulus();
    const auto& z_mat = client->GetProofZ();
    ASSERT_EQ(z_mat.Rows(), expected_z.Rows());
    ASSERT_EQ(z_mat.Cols(), expected_z.Cols());
    for (int i = 0; i < z_mat.Rows(); ++i) {
      for (int j = 0; j < z_mat.Cols(); ++j) {
        TypeParam z_val = z_mat.Data()[i * z_mat.Cols() + j];
        TypeParam exp_val = expected_z.Data()[i * expected_z.Cols() + j] % p_z;
        if (z_val != exp_val) {
          LOG(ERROR) << "Z mismatch at i=" << i << ", j=" << j
                     << ": decrypted=" << z_val << ", expected=" << exp_val;
          FAIL() << "Z decryption mismatch!";
        }
      }
    }
  } else {
    LOG(ERROR) << "GetProofZ() is empty! status: " << status;
  }
  ASSERT_OK(status);
  EXPECT_EQ(client->GetProofC().Rows(), kappa);
  EXPECT_EQ(client->GetProofC().Cols(),
            params.Ell() * params.RlweParameters().Degree());
  for (const auto& val : client->GetProofC().Data()) {
    EXPECT_TRUE(val == 0 || val == 1);
  }
  EXPECT_EQ(client->GetProofZ().Rows(), kappa);
  EXPECT_EQ(client->GetProofZ().Cols(), params.NumEntries());
  EXPECT_EQ(client->GetZPrimeMatrix().Rows(), kappa);
  EXPECT_EQ(client->GetZPrimeMatrix().Cols(),
            2 * params.PackGadgetParams().num_digits *
                params.RlweParameters().Degree());

  const int test_index = 42;
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
  ASSERT_OK_AND_ASSIGN(auto response,
                       server->ProcessResponseWithMatMul(request));
  ASSERT_OK_AND_ASSIGN(auto result,
                       client->ProcessResponse(response, request, test_index));
  std::vector<TypeParam> expected_message(
      db.begin() + test_index * params.RlweParameters().Degree(),
      db.begin() + (test_index + 1) * params.RlweParameters().Degree());
  EXPECT_EQ(result, expected_message);
}

TYPED_TEST(VpirClientTest, GetProofPreprocessMaterialAndLocalFinalizeWorks) {
  ASSERT_OK_AND_ASSIGN(auto params,
                       this->CreateValidVpirParams(/*local_finalize=*/true));

  ASSERT_OK_AND_ASSIGN(
      auto prep_server,
      (VpirPreprocessingServer<uint16_t, TypeParam, TypeParam>::Create(
          &params)));
  std::vector<uint16_t> db(params.NumEntries() *
                           params.RlweParameters().Degree() *
                           params.EntrySizeMultiple());
  std::mt19937 rng(42);
  for (auto& val : db) {
    val = rng() % 256;
  }
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data, prep_server->Preprocess(db));

  ASSERT_OK_AND_ASSIGN(auto server,
                       (VpirServer<uint16_t, TypeParam, TypeParam>::Create(
                           params, std::move(preprocessed_data))));

  ASSERT_OK_AND_ASSIGN(auto a_prng,
                       ::rlwe::ChaChaPrng::Create(server->GetPrngSeed()));
  ASSERT_OK_AND_ASSIGN(auto prng, this->CreatePrng());
  ASSERT_OK_AND_ASSIGN(auto client,
                       VpirClient<TypeParam>::Create(params, std::move(a_prng),
                                                     std::move(prng)));

  const int kappa = 10;
  ASSERT_OK(client->GetProofPreprocessMaterial(kappa, *server));
  EXPECT_EQ(client->GetProofC().Rows(), kappa);
  EXPECT_EQ(client->GetProofC().Cols(),
            params.Ell() * params.RlweParameters().Degree());
  EXPECT_EQ(client->GetProofZ().Rows(), kappa);
  EXPECT_EQ(client->GetProofZ().Cols(), params.NumEntries());
  EXPECT_EQ(client->GetZPrimeMatrix().Rows(), kappa);
  EXPECT_EQ(client->GetZPrimeMatrix().Cols(),
            params.PackGadgetParams().num_digits *
                params.RlweParameters().Degree());

  const int test_index = 42;
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
  EXPECT_EQ(request.packing_key.size(),
            static_cast<size_t>(params.PackGadgetParams().num_digits));

  ASSERT_OK_AND_ASSIGN(auto response,
                       server->ProcessResponseWithMatMul(request));
  ASSERT_OK_AND_ASSIGN(auto result,
                       client->ProcessResponse(response, request, test_index));
  std::vector<TypeParam> expected_message(
      db.begin() + test_index * params.RlweParameters().Degree(),
      db.begin() + (test_index + 1) * params.RlweParameters().Degree());
  EXPECT_EQ(result, expected_message);
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
