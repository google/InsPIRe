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
#include "crypto/matrix.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/pir_params.h"
#include "crypto/vpir/client/vpir_client.h"
#include "crypto/vpir/server/vpir_preprocessing_server.h"
#include "crypto/vpir/server/vpir_server.h"
#include "crypto/vpir/vpir_params.h"
#include <benchmark/benchmark.h>
#include <gmock/gmock.h>
#include "crypto/status_macros.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/flags/parse.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

ABSL_FLAG(int, degree, 2048, "The degree of the RLWE polynomial.");
ABSL_FLAG(int, num_entries, 2048, "The number of entries in the database.");
ABSL_FLAG(int, kappa, 40, "Number of verifiable PIR proof queries.");

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

constexpr int kVariance = 6;
constexpr int kLogDigit = 19;
constexpr int kNumDigits = 3;
constexpr int kTargetIndex = 0;

struct Uint64Mod44Pt256 {
  using DbDataType = uint8_t;
  using CoeffType = uint64_t;
  using MatCoeffType = int32_t;
  static constexpr uint64_t kModulus = 1ULL << 44;
  static constexpr uint64_t kModulus1AfterSwitch = 1ULL << 43;
  static constexpr uint64_t kModulus2AfterSwitch = 1ULL << 42;
  static constexpr int kPlaintextModulus = 256;
};

template <typename TypeParam>
struct BenchmarkState {
  using DbDataType = typename TypeParam::DbDataType;
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;

  std::unique_ptr<VpirParams<CoeffType>> params;
  std::vector<DbDataType> db_raw;
  std::unique_ptr<VpirServer<DbDataType, CoeffType, MatCoeffType>> server;
  std::unique_ptr<VpirClient<CoeffType>> client;
  std::unique_ptr<VpirRequest<CoeffType>> request;
  std::vector<PirRequest<CoeffType>> z_requests;
};

BenchmarkState<Uint64Mod44Pt256>* benchmark_state_ptr = nullptr;

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng(
    absl::string_view prng_seed) {
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() {
  ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

template <typename TypeParam>
absl::StatusOr<VpirParams<typename TypeParam::CoeffType>> CreateValidVpirParams(
    const RlweParams<typename TypeParam::CoeffType>& rlwe_params) {
  GadgetParams pack_gadget_params{/*log_digit=*/kLogDigit,
                                  /*num_digits=*/kNumDigits};

  return VpirParams<typename TypeParam::CoeffType>::Create(
      rlwe_params, /*plaintext_modulus=*/TypeParam::kPlaintextModulus,
      pack_gadget_params, absl::GetFlag(FLAGS_num_entries));
}

template <typename TypeParam>
void SetUpBenchmark() {
  using DbDataType = typename TypeParam::DbDataType;
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;

  benchmark_state_ptr = new BenchmarkState<TypeParam>();

  ASSERT_OK_AND_ASSIGN(auto rlwe_params,
                       RlweParams<CoeffType>::Create(
                           absl::GetFlag(FLAGS_degree), TypeParam::kModulus,
                           TypeParam::kModulus1AfterSwitch,
                           TypeParam::kModulus2AfterSwitch, kVariance));
  ASSERT_OK_AND_ASSIGN(auto params,
                       CreateValidVpirParams<TypeParam>(rlwe_params));
  benchmark_state_ptr->params =
      std::make_unique<VpirParams<CoeffType>>(std::move(params));

  const int d = benchmark_state_ptr->params->RlweParameters().Degree();
  const int n = benchmark_state_ptr->params->NumEntries();
  const int ell = benchmark_state_ptr->params->Ell();

  ASSERT_OK_AND_ASSIGN(auto db_prng, CreatePrng());

  LOG(INFO) << "VPIR Database size: "
            << static_cast<size_t>(n) * d * ell / 1024 / 1024 << " MB";

  benchmark_state_ptr->db_raw.resize(static_cast<size_t>(n) * d * ell);
  for (size_t i = 0; i < benchmark_state_ptr->db_raw.size(); ++i) {
    ASSERT_OK_AND_ASSIGN(auto rand_val, db_prng->Rand8());
    benchmark_state_ptr->db_raw[i] =
        rand_val % benchmark_state_ptr->params->PlaintextModulus();
  }

  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (VpirPreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          benchmark_state_ptr->params.get())));

  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_data,
      preprocessing_server->Preprocess(benchmark_state_ptr->db_raw));

  ASSERT_OK_AND_ASSIGN(
      benchmark_state_ptr->server,
      (VpirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          *benchmark_state_ptr->params, std::move(preprocessed_data))));

  ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                       CreatePrng(benchmark_state_ptr->server->GetPrngSeed()));
  ASSERT_OK_AND_ASSIGN(auto client_prng, CreatePrng());
  ASSERT_OK_AND_ASSIGN(benchmark_state_ptr->client,
                       (VpirClient<CoeffType>::Create(
                           *benchmark_state_ptr->params,
                           std::move(client_a_prng), std::move(client_prng))));

  ASSERT_OK_AND_ASSIGN(
      auto request, benchmark_state_ptr->client->CreateRequest(kTargetIndex));
  benchmark_state_ptr->request =
      std::make_unique<VpirRequest<CoeffType>>(std::move(request));

  ASSERT_OK_AND_ASSIGN(auto c_prng, CreatePrng());
  const int kappa = absl::GetFlag(FLAGS_kappa);
  const int total_entry_coeffs = ell * d;

  std::vector<std::vector<CoeffType>> c_data(
      kappa, std::vector<CoeffType>(total_entry_coeffs));
  for (int i = 0; i < kappa; ++i) {
    for (int j = 0; j < total_entry_coeffs; ++j) {
      ASSERT_OK_AND_ASSIGN(auto r, c_prng->Rand8());
      c_data[i][j] = r % 2;
    }
  }
  ASSERT_OK_AND_ASSIGN(auto c_mat,
                       Matrix<CoeffType>::Create(std::move(c_data)));

  ASSERT_OK_AND_ASSIGN(auto z_client_a_prng,
                       CreatePrng(benchmark_state_ptr->server->GetPrngSeed()));
  ASSERT_OK_AND_ASSIGN(auto z_client_prng, CreatePrng());
  ASSERT_OK_AND_ASSIGN(
      auto z_pir_client,
      PirClient<CoeffType>::Create(benchmark_state_ptr->params->ZPirParams(),
                                   std::move(z_client_a_prng),
                                   std::move(z_client_prng)));

  benchmark_state_ptr->z_requests.reserve(kappa);
  for (int i = 0; i < kappa; ++i) {
    std::vector<CoeffType> row_vec(total_entry_coeffs);
    for (int j = 0; j < total_entry_coeffs; ++j) {
      row_vec[j] = c_mat.Data()[i * total_entry_coeffs + j];
    }
    ASSERT_OK_AND_ASSIGN(auto req,
                         z_pir_client->CreateRequestForVector(row_vec));
    benchmark_state_ptr->z_requests.push_back(std::move(req));
  }

  ASSERT_OK_AND_ASSIGN(auto test_res,
                       benchmark_state_ptr->server->ProcessResponse(
                           *benchmark_state_ptr->request));
  ASSERT_OK_AND_ASSIGN(
      auto recovered_msg,
      benchmark_state_ptr->client->ProcessResponse(
          test_res, *benchmark_state_ptr->request, kTargetIndex));
  std::vector<CoeffType> expected_msg(
      benchmark_state_ptr->db_raw.begin() + kTargetIndex * d,
      benchmark_state_ptr->db_raw.begin() + (kTargetIndex + 1) * d);
  if (recovered_msg != expected_msg) {
    LOG(FATAL) << "VPIR online query decryption verification failed!";
  }
}

template <typename TypeParam>
void TearDownBenchmark() {
  delete benchmark_state_ptr;
  benchmark_state_ptr = nullptr;
}

template <typename TypeParam>
void BM_VpirServerProcessResponse(benchmark::State& state) {
  auto server = benchmark_state_ptr->server.get();
  auto request = benchmark_state_ptr->request.get();

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(*request));
    benchmark::DoNotOptimize(response);
  }
}

void BM_VpirServerProcessResponse_Uint64Mod44Pt256(benchmark::State& state) {
  BM_VpirServerProcessResponse<Uint64Mod44Pt256>(state);
}

template <typename TypeParam>
void BM_VpirServerGetProofPreprocessMaterial(benchmark::State& state) {
  auto server = benchmark_state_ptr->server.get();
  const auto& z_requests = benchmark_state_ptr->z_requests;

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto material,
                         server->GetProofPreprocessMaterial(z_requests));
    benchmark::DoNotOptimize(material);
  }
}

void BM_VpirServerGetProofPreprocessMaterial_Uint64Mod44Pt256(
    benchmark::State& state) {
  BM_VpirServerGetProofPreprocessMaterial<Uint64Mod44Pt256>(state);
}

BENCHMARK(BM_VpirServerProcessResponse_Uint64Mod44Pt256);
BENCHMARK(BM_VpirServerGetProofPreprocessMaterial_Uint64Mod44Pt256);

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  private_membership::rlwe::v2::SetUpBenchmark<
      private_membership::rlwe::v2::Uint64Mod44Pt256>();
  ::benchmark::RunSpecifiedBenchmarks();
  private_membership::rlwe::v2::TearDownBenchmark<
      private_membership::rlwe::v2::Uint64Mod44Pt256>();
  return 0;
}
