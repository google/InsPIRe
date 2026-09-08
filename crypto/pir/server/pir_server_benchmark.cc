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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/external_product.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_server.h"
#include "crypto/pir/server/preprocessing_server.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include <benchmark/benchmark.h>
#include <gmock/gmock.h>
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/flags/parse.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

ABSL_FLAG(int, degree, 1024, "The degree of the RLWE polynomial.");
ABSL_FLAG(int, num_entries, 1 << 14, "The number of entries in the database.");
ABSL_FLAG(int, interpolation_degree, 16,
          "The degree used for polynomial interpolation.");
ABSL_FLAG(int, entry_size_multiple, 1,
          "The number of shards (entry_size_multiple).");

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

constexpr int kVariance = 6;
constexpr int kLogDigit = 19;
constexpr int kNumDigits = 2;
constexpr int kTargetIndex = 123;

struct Uint64Mod56Pt255 {
  using DbValueType = uint8_t;
  using CoeffType = uint64_t;
  using MatCoeffType = int32_t;
  static constexpr uint64_t kModulus = 1ULL << 56;
  static constexpr int kPlaintextModulus = 255;
};

// Global state for benchmark
template <typename TypeParam>
struct BenchmarkState {
  using DbValueType = typename TypeParam::DbValueType;
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;

  std::unique_ptr<PirParams<CoeffType>> params;
  std::vector<DbValueType> db_raw;
  std::vector<Matrix<DbValueType>> db_matrices;
  std::unique_ptr<PirServer<DbValueType, CoeffType, MatCoeffType>> server;
  std::unique_ptr<PirClient<CoeffType>> client;
  std::unique_ptr<PirRequest<CoeffType>> request;
  std::vector<CoeffType> flattened_first_dim;
  std::vector<CoeffType> b_prime;
  std::vector<std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
      preprocessed_outputs;
  std::vector<RlweCiphertext<CoeffType>> packed_ciphertexts;
  std::vector<Polynomial<CoeffType>> second_dim_query_a;
  std::unique_ptr<Context> ctx;
  std::unique_ptr<FftContext> fft_ctx;
};

BenchmarkState<Uint64Mod56Pt255>* benchmark_state_ptr;

// Helper to create PRNG.
absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng(
    absl::string_view prng_seed) {
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() {
  ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

// Helper to create valid PIR parameters.
template <typename TypeParam>
absl::StatusOr<PirParams<typename TypeParam::CoeffType>> CreateValidPirParams(
    const RlweParams<typename TypeParam::CoeffType>& rlwe_params) {
  GadgetParams pack_gadget_params{/*log_digit=*/kLogDigit,
                                  /*num_digits=*/kNumDigits};
  GadgetParams poly_eval_gadget_params{/*log_digit=*/kLogDigit,
                                       /*num_digits=*/kNumDigits};

  return PirParams<typename TypeParam::CoeffType>::Create(
      rlwe_params, /*plaintext_modulus=*/TypeParam::kPlaintextModulus,
      pack_gadget_params, poly_eval_gadget_params,
      absl::GetFlag(FLAGS_num_entries),
      absl::GetFlag(FLAGS_interpolation_degree),
      absl::GetFlag(FLAGS_entry_size_multiple));
}

// SetUp function for benchmark.
template <typename TypeParam>
void SetUpBenchmark() {
  using DbValueType = typename TypeParam::DbValueType;
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;

  benchmark_state_ptr = new BenchmarkState<TypeParam>();

  // Setup PIR server (offline phase).
  ASSERT_OK_AND_ASSIGN(auto rlwe_params, RlweParams<CoeffType>::Create(
                                             absl::GetFlag(FLAGS_degree),
                                             TypeParam::kModulus, kVariance));
  ASSERT_OK_AND_ASSIGN(auto params,
                       CreateValidPirParams<TypeParam>(rlwe_params));
  benchmark_state_ptr->params =
      std::make_unique<PirParams<CoeffType>>(std::move(params));

  const int d = benchmark_state_ptr->params->RlweParameters().Degree();
  // const int t = benchmark_state_ptr->params->InterpolationDegree();
  const int n = benchmark_state_ptr->params->NumEntries();

  ASSERT_OK_AND_ASSIGN(auto db_prng, CreatePrng());

  // print the size of the database in MB
  // Each entry is modulo 257 or 8 bits (1 byte)
  LOG(INFO) << "Database size: " << n * d * 1 / 1024 / 1024 << " MB";

  const int entry_size_multiple =
      benchmark_state_ptr->params->EntrySizeMultiple();

  // Database creation
  benchmark_state_ptr->db_raw.resize(n * d * entry_size_multiple);
  for (int i = 0; i < n * d * entry_size_multiple; ++i) {
    ASSERT_OK_AND_ASSIGN(auto rand_val, db_prng->Rand8());
    benchmark_state_ptr->db_raw[i] =
        rand_val % benchmark_state_ptr->params->PlaintextModulus();
  }

  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (PreprocessingServer<DbValueType, CoeffType, MatCoeffType>::Create(
          benchmark_state_ptr->params.get())));
  std::string prng_seed = preprocessing_server->GetPrngSeed();

  ASSERT_OK_AND_ASSIGN(
      auto preprocessed_data,
      preprocessing_server->Preprocess(benchmark_state_ptr->db_raw));
  benchmark_state_ptr->db_matrices = preprocessed_data.db_matrices;
  benchmark_state_ptr->preprocessed_outputs =
      preprocessed_data.preprocessed_outputs;
  benchmark_state_ptr->second_dim_query_a = preprocessed_data.second_dim_a;

  {
    ASSERT_OK_AND_ASSIGN(
        auto ctx,
        Context::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
    benchmark_state_ptr->ctx = std::make_unique<Context>(std::move(ctx));
  }

  ASSERT_OK_AND_ASSIGN(
      benchmark_state_ptr->server,
      (PirServer<DbValueType, CoeffType, MatCoeffType>::Create(
          *benchmark_state_ptr->params, std::move(preprocessed_data))));

  ASSERT_OK_AND_ASSIGN(auto client_a_prng, CreatePrng(prng_seed));
  ASSERT_OK_AND_ASSIGN(auto client_prng, CreatePrng());
  ASSERT_OK_AND_ASSIGN(benchmark_state_ptr->client,
                       (PirClient<CoeffType>::Create(
                           *benchmark_state_ptr->params,
                           std::move(client_a_prng), std::move(client_prng))));

  ASSERT_OK_AND_ASSIGN(
      auto request, benchmark_state_ptr->client->CreateRequest(kTargetIndex));
  benchmark_state_ptr->request =
      std::make_unique<PirRequest<CoeffType>>(request);

  // First dimension query for matrix multiplication benchmark.
  benchmark_state_ptr->flattened_first_dim =
      benchmark_state_ptr->request->first_dimension_query;
  if (benchmark_state_ptr->flattened_first_dim.size() <
      static_cast<size_t>(benchmark_state_ptr->db_matrices[0].Cols())) {
    benchmark_state_ptr->flattened_first_dim.resize(
        benchmark_state_ptr->db_matrices[0].Cols(), 0);
  }
  ASSERT_OK_AND_ASSIGN(
      benchmark_state_ptr->b_prime,
      benchmark_state_ptr->db_matrices[0].template Multiply<CoeffType>(
          benchmark_state_ptr->flattened_first_dim));

  ASSERT_OK_AND_ASSIGN(
      auto fft_ctx,
      FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
  benchmark_state_ptr->fft_ctx = std::move(fft_ctx);

  // Compute packed ciphertexts for EvalPoly benchmark
  auto params_ptr = benchmark_state_ptr->params.get();
  const int d_for_pack = params_ptr->RlweParameters().Degree();
  const int t_for_pack = params_ptr->InterpolationDegree();
  const int num_digits_for_pack = params_ptr->PackGadgetParams().num_digits;
  auto pack_begin_for_pack = benchmark_state_ptr->request->packing_key.begin();
  std::vector<Polynomial<CoeffType>> y_vec_g_for_pack(
      pack_begin_for_pack, pack_begin_for_pack + num_digits_for_pack);
  std::vector<Polynomial<CoeffType>> y_vec_h_for_pack(
      pack_begin_for_pack + num_digits_for_pack,
      pack_begin_for_pack + 2 * num_digits_for_pack);

  const int k_for_pack = y_vec_g_for_pack.size();
  std::vector<CoeffType> y_vec_g_vec_for_pack(k_for_pack * d_for_pack);
  for (int l = 0; l < k_for_pack; ++l) {
    for (int m = 0; m < d_for_pack; ++m) {
      y_vec_g_vec_for_pack[l * d_for_pack + m] =
          y_vec_g_for_pack[l].Coeffs()[m];
    }
  }

  benchmark_state_ptr->packed_ciphertexts.reserve(t_for_pack);
  for (int i = 0; i < t_for_pack; ++i) {
    std::vector<CoeffType> chunk(
        benchmark_state_ptr->b_prime.begin() + i * d_for_pack,
        benchmark_state_ptr->b_prime.begin() + (i + 1) * d_for_pack);
    ASSERT_OK_AND_ASSIGN(
        std::vector<CoeffType> b_agg_partial,
        benchmark_state_ptr->preprocessed_outputs[0][i]
            .matrix.template Multiply<CoeffType>(y_vec_g_vec_for_pack));

    ASSERT_OK_AND_ASSIGN(
        (RlweCiphertext<CoeffType> pack_result),
        FinalizeMatrixPack(
            params_ptr->RlweParameters(), chunk, b_agg_partial,
            y_vec_h_for_pack,
            benchmark_state_ptr->preprocessed_outputs[0][i].t_vec_h,
            benchmark_state_ptr->preprocessed_outputs[0][i].a_tilde_agg,
            params_ptr->PackGadgetParams(), *benchmark_state_ptr->fft_ctx));
    benchmark_state_ptr->packed_ciphertexts.push_back(std::move(pack_result));
  }
}

template <typename TypeParam>
void TearDownBenchmark() {
  delete benchmark_state_ptr;
  benchmark_state_ptr = nullptr;
}

// Benchmark function.
template <typename TypeParam>
void BM_PirServerProcessResponse(benchmark::State& state) {
  auto params = benchmark_state_ptr->params.get();
  auto server = benchmark_state_ptr->server.get();
  auto client = benchmark_state_ptr->client.get();
  auto request = benchmark_state_ptr->request.get();
  const auto& db_raw = benchmark_state_ptr->db_raw;
  const int entry_size_multiple = params->EntrySizeMultiple();
  const int d = params->RlweParameters().Degree();
  std::vector<typename TypeParam::CoeffType> expected_message(
      db_raw.begin() + kTargetIndex * d * entry_size_multiple,
      db_raw.begin() + (kTargetIndex + 1) * d * entry_size_multiple);

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(*request));

    state.PauseTiming();

    // Verification check
    ASSERT_OK_AND_ASSIGN(auto recovered_message,
                         client->ProcessResponse(response));
    ASSERT_EQ(recovered_message, expected_message);

    state.ResumeTiming();
  }
}

void BM_PirServerProcessResponse_Uint64Mod56Pt255(benchmark::State& state) {
  BM_PirServerProcessResponse<Uint64Mod56Pt255>(state);
}

// Benchmark for matrix-vector multiplication in ProcessResponse.
template <typename TypeParam>
void BM_PirServerProcessResponse_MatrixVectorMul(benchmark::State& state) {
  using DbValueType = typename TypeParam::DbValueType;
  using CoeffType = typename TypeParam::CoeffType;
  const std::vector<CoeffType>& flattened_first_dim =
      benchmark_state_ptr->flattened_first_dim;

  const Matrix<DbValueType>& db_matrix = benchmark_state_ptr->db_matrices[0];
  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto result, db_matrix.template Multiply<CoeffType>(
                                          flattened_first_dim));
    benchmark::DoNotOptimize(result);
  }
}

void BM_PirServerProcessResponse_MatrixVectorMul_Uint64Mod56Pt255(
    benchmark::State& state) {
  BM_PirServerProcessResponse_MatrixVectorMul<Uint64Mod56Pt255>(state);
}

// Benchmark for packing in ProcessResponse.
template <typename TypeParam>
void BM_PirServerProcessResponse_Packing(benchmark::State& state) {
  using CoeffType = typename TypeParam::CoeffType;
  auto params = benchmark_state_ptr->params.get();
  const int d = params->RlweParameters().Degree();
  const int t = params->InterpolationDegree();
  const int num_digits = params->PackGadgetParams().num_digits;
  auto pack_begin = benchmark_state_ptr->request->packing_key.begin();
  std::vector<Polynomial<CoeffType>> y_vec_g(pack_begin,
                                             pack_begin + num_digits);
  std::vector<Polynomial<CoeffType>> y_vec_h(pack_begin + num_digits,
                                             pack_begin + 2 * num_digits);
  const std::vector<CoeffType>& b_prime = benchmark_state_ptr->b_prime;
  const auto& preprocessed_outputs = benchmark_state_ptr->preprocessed_outputs;
  FftContext* fft_ctx = benchmark_state_ptr->fft_ctx.get();

  for (auto s : state) {
    std::vector<RlweCiphertext<CoeffType>> packed_ciphertexts;
    packed_ciphertexts.reserve(t);
    for (int i = 0; i < t; ++i) {
      std::vector<CoeffType> chunk(b_prime.begin() + i * d,
                                   b_prime.begin() + (i + 1) * d);
      std::vector<CoeffType> y_vec_g_vec(num_digits * d);
      for (int l = 0; l < num_digits; ++l) {
        for (int m = 0; m < d; ++m) {
          y_vec_g_vec[l * d + m] = y_vec_g[l].Coeffs()[m];
        }
      }
      ASSERT_OK_AND_ASSIGN(
          std::vector<CoeffType> b_agg_partial,
          preprocessed_outputs[0][i].matrix.template Multiply<CoeffType>(
              y_vec_g_vec));

      ASSERT_OK_AND_ASSIGN(
          (RlweCiphertext<CoeffType> pack_result),
          FinalizeMatrixPack(params->RlweParameters(), chunk, b_agg_partial,
                                 y_vec_h,
                                 preprocessed_outputs[0][i].t_vec_h,
                                 preprocessed_outputs[0][i].a_tilde_agg,
                                 params->PackGadgetParams(), *fft_ctx));
      packed_ciphertexts.push_back(std::move(pack_result));
    }
    benchmark::DoNotOptimize(packed_ciphertexts);
  }
}

void BM_PirServerProcessResponse_Packing_Uint64Mod56Pt255(
    benchmark::State& state) {
  BM_PirServerProcessResponse_Packing<Uint64Mod56Pt255>(state);
}

// Benchmark for EvalPoly in ProcessResponse.
template <typename TypeParam>
void BM_PirServerProcessResponse_EvalPoly(benchmark::State& state) {
  using CoeffType = typename TypeParam::CoeffType;
  auto params = benchmark_state_ptr->params.get();
  const auto& packed_ciphertexts = benchmark_state_ptr->packed_ciphertexts;
  const auto& second_dim_query_a = benchmark_state_ptr->second_dim_query_a;
  const auto& second_dimension_query =
      benchmark_state_ptr->request->second_dimension_query;

  const int eval_num_digits = params->PolyEvalGadgetParams().num_digits;
  RgswCiphertext<CoeffType> rgsw_ct;
  rgsw_ct.u.reserve(eval_num_digits);
  rgsw_ct.v.reserve(eval_num_digits);
  for (int i = 0; i < eval_num_digits; ++i) {
    rgsw_ct.u.push_back({second_dim_query_a[i], second_dimension_query[i]});
  }
  for (int i = eval_num_digits; i < 2 * eval_num_digits; ++i) {
    rgsw_ct.v.push_back({second_dim_query_a[i], second_dimension_query[i]});
  }

  for (auto s : state) {
    RlweCiphertext<CoeffType> result = packed_ciphertexts.back();
    const auto& rlwe_params = params->RlweParameters();
    for (int i = packed_ciphertexts.size() - 2; i >= 0; --i) {
      ASSERT_OK_AND_ASSIGN(
          RlweCiphertext<CoeffType> product,
          ExternalProduct(rlwe_params, params->PolyEvalGadgetParams(), result,
                          rgsw_ct, *benchmark_state_ptr->fft_ctx));
      ASSERT_OK_AND_ASSIGN(Polynomial<CoeffType> next_a,
                           product.a.Add(packed_ciphertexts[i].a));
      ASSERT_OK_AND_ASSIGN(Polynomial<CoeffType> next_b,
                           product.b.Add(packed_ciphertexts[i].b));
      result = {std::move(next_a), std::move(next_b)};
    }
    benchmark::DoNotOptimize(result);
  }
}

void BM_PirServerProcessResponse_EvalPoly_Uint64Mod56Pt255(
    benchmark::State& state) {
  BM_PirServerProcessResponse_EvalPoly<Uint64Mod56Pt255>(state);
}

// To run the benchmark:
// blaze run -c opt --dynamic_mode=off
// //privacy/private_membership/rlwe/v2/crypto/pir:pir_server_benchmark --
// --benchmark_filter=all

// Benchmarks entire online server computation of PIR
BENCHMARK(BM_PirServerProcessResponse_Uint64Mod56Pt255);

// Benchmark for first-dimension matrix-vector multiplication
BENCHMARK(BM_PirServerProcessResponse_MatrixVectorMul_Uint64Mod56Pt255);

// Benchmark for packing in ProcessResponse
BENCHMARK(BM_PirServerProcessResponse_Packing_Uint64Mod56Pt255);

// Benchmark for polynomial evaluation
BENCHMARK(BM_PirServerProcessResponse_EvalPoly_Uint64Mod56Pt255);

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  private_membership::rlwe::v2::SetUpBenchmark<
      private_membership::rlwe::v2::Uint64Mod56Pt255>();
  ::benchmark::RunSpecifiedBenchmarks();
  private_membership::rlwe::v2::TearDownBenchmark<
      private_membership::rlwe::v2::Uint64Mod56Pt255>();
  return 0;
}
