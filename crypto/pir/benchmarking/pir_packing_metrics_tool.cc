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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_preprocessed_data.h"
#include "crypto/pir/server/preprocessing_server.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "crypto/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/flags/parse.h"
#include "crypto/status_macros.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

ABSL_FLAG(int, num_entries, 2048, "Number of entries in the database (n).");
ABSL_FLAG(int, entry_size_multiple, 1, "Number of database shards (ell).");
ABSL_FLAG(int, degree, 2048, "RLWE ring degree (d).");
ABSL_FLAG(int, pack_log_digit, 19, "Gadget log digit for packing (log2(z)).");
ABSL_FLAG(int, pack_num_digits, 2, "Number of gadget digits for packing.");
ABSL_FLAG(int, iterations, 50,
          "Number of timing iterations for online phase.");
ABSL_FLAG(int, warmup_iterations, 5, "Number of warmup iterations.");
ABSL_FLAG(std::string, output_file, "", "File to append metrics to.");
ABSL_FLAG(std::string, label, "", "Benchmark run label.");
ABSL_FLAG(bool, skip_verification, false,
          "If true, skip correctness verification check.");

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using DbDataType = uint16_t;
using CoeffType = uint64_t;
using MatCoeffType = int32_t;

// General RLWE parameters
constexpr uint64_t kModulus = 1ULL << 56;
constexpr int kVariance = 11;

// Simplified PIR parameters
constexpr uint64_t kPlaintextModulus = 1ULL << 16;  // 65536 (2^16)
constexpr int kInterpolationDegree = 1; // Set to 1 for packing metrics

// Dummy parameters for unused polynomial evaluation gadget (since
// interpolation_degree = 1)
constexpr int kEvalLogDigit = 19;
constexpr int kEvalNumDigits = 3;

constexpr int kTargetIndex = 0;

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng(
    absl::string_view prng_seed) {
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() {
  ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

absl::Status RunPackingMetrics() {
  const int n = absl::GetFlag(FLAGS_num_entries);
  const int entry_size_multiple = absl::GetFlag(FLAGS_entry_size_multiple);
  const int d = absl::GetFlag(FLAGS_degree);
  const int pack_log_digit = absl::GetFlag(FLAGS_pack_log_digit);
  const int pack_num_digits = absl::GetFlag(FLAGS_pack_num_digits);
  const int iterations = absl::GetFlag(FLAGS_iterations);
  const int warmup_iterations = absl::GetFlag(FLAGS_warmup_iterations);

  // No modulus switching: mod1 and mod2 after switch equal kModulus
  const uint64_t mod1_after_switch = kModulus;
  const uint64_t mod2_after_switch = kModulus;

  LOG(INFO) << "Starting PIR Packing Metrics Run: d=" << d << ", n=" << n
            << ", entry_size_multiple=" << entry_size_multiple
            << ", interpolation_degree=" << kInterpolationDegree
            << ", plaintext_modulus=" << kPlaintextModulus
            << ", pack_log_digit=" << pack_log_digit
            << ", pack_num_digits=" << pack_num_digits
            << ", warmup_iterations=" << warmup_iterations
            << ", iterations=" << iterations;

  ASSIGN_OR_RETURN(
      auto rlwe_params,
      RlweParams<CoeffType>::Create(d, kModulus, mod1_after_switch,
                                    mod2_after_switch, kVariance));

  GadgetParams pack_gadget{pack_log_digit, pack_num_digits};
  GadgetParams poly_eval_gadget{kEvalLogDigit, kEvalNumDigits};

  ASSIGN_OR_RETURN(
      auto params,
      PirParams<CoeffType>::Create(rlwe_params, kPlaintextModulus, pack_gadget,
                                   poly_eval_gadget, n, kInterpolationDegree,
                                   entry_size_multiple));

  // Generate random DB
  const size_t total_db_entries =
      static_cast<size_t>(n) * d * entry_size_multiple;
  const size_t total_db_bytes = total_db_entries * sizeof(DbDataType);
  LOG(INFO) << "Database size: "
            << total_db_bytes / (1024.0 * 1024.0) << " MB";
  std::vector<DbDataType> db_raw(total_db_entries);
  ASSIGN_OR_RETURN(auto db_prng, CreatePrng());
  for (size_t i = 0; i < db_raw.size(); ++i) {
    ASSIGN_OR_RETURN(auto rand_val, db_prng->Rand64());
    db_raw[i] = static_cast<DbDataType>(rand_val % kPlaintextModulus);
  }

  // 1. Preprocessing Server
  LOG(INFO) << "[Step 1/2] Running Preprocessing Server on DB...";
  ASSIGN_OR_RETURN(
      auto prep_server,
      (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          &params)));
  absl::Time start_time = absl::Now();
  ASSIGN_OR_RETURN(auto preprocessed_data, prep_server->Preprocess(db_raw));
  absl::Duration prep_server_time = absl::Now() - start_time;

  // Combine DB matrices
  ASSIGN_OR_RETURN(
      auto combined_db_matrix,
      Matrix<DbDataType>::VerticalConcatenate(preprocessed_data.db_matrices));

  // Combine pack matrices
  std::vector<Matrix<MatCoeffType>> pack_matrices;
  pack_matrices.reserve(entry_size_multiple * kInterpolationDegree);
  for (int k = 0; k < entry_size_multiple; ++k) {
    for (int i = 0; i < kInterpolationDegree; ++i) {
      pack_matrices.push_back(
          std::move(preprocessed_data.preprocessed_outputs[k][i].matrix));
      preprocessed_data.preprocessed_outputs[k][i].matrix =
          Matrix<MatCoeffType>();
    }
  }
  ASSIGN_OR_RETURN(auto combined_pack_matrix,
                   Matrix<MatCoeffType>::VerticalConcatenate(pack_matrices));

  auto preprocessed_outputs =
      std::move(preprocessed_data.preprocessed_outputs);

  ASSIGN_OR_RETURN(
      auto fft_ctx,
      FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));

  // 2. Online Phase with Detailed Packing Timings
  LOG(INFO) << "[Step 2/2] Running Online Phase timing iterations (warmup="
            << warmup_iterations << ", measured=" << iterations << ")...";

  absl::Duration client_query_gen_time = absl::ZeroDuration();
  absl::Duration server_proc_resp_time = absl::ZeroDuration();
  absl::Duration server_db_mult_time = absl::ZeroDuration();
  absl::Duration server_packing_total_time = absl::ZeroDuration();
  absl::Duration server_pack_matrix_mult_time = absl::ZeroDuration();
  absl::Duration server_pack_finalize_time = absl::ZeroDuration();
  absl::Duration client_proc_resp_time = absl::ZeroDuration();

  std::vector<CoeffType> rec_msg;

  const int total_runs = warmup_iterations + iterations;
  for (int run = 0; run < total_runs; ++run) {
    const bool is_warmup = (run < warmup_iterations);

    ASSIGN_OR_RETURN(auto client_a_prng,
                     CreatePrng(prep_server->GetPrngSeed()));
    ASSIGN_OR_RETURN(auto client_prng, CreatePrng());
    ASSIGN_OR_RETURN(auto client,
                     PirClient<CoeffType>::Create(
                         params, std::move(client_a_prng),
                         std::move(client_prng)));

    // Client query generation
    auto t_client_gen_start = absl::Now();
    ASSIGN_OR_RETURN(auto req, client->CreateRequest(kTargetIndex));
    auto cur_client_gen_time = absl::Now() - t_client_gen_start;

    // Server processing
    // Extract packing keys
    const int num_digits = params.PackGadgetParams().num_digits;
    if (req.packing_key.size() != static_cast<size_t>(2 * num_digits)) {
      return absl::InvalidArgumentError(
          "Packing key must contain 2 * num_digits polynomials.");
    }
    auto pack_begin = req.packing_key.begin();
    std::vector<Polynomial<CoeffType>> y_vec_g(pack_begin,
                                               pack_begin + num_digits);
    std::vector<Polynomial<CoeffType>> y_vec_h(pack_begin + num_digits,
                                               pack_begin + 2 * num_digits);

    // First dimension query
    std::vector<CoeffType> flattened_first_dim = req.first_dimension_query;
    if (flattened_first_dim.size() <
        static_cast<size_t>(combined_db_matrix.Cols())) {
      flattened_first_dim.resize(combined_db_matrix.Cols(), 0);
    }

    // Step 2.1: DB Matrix Multiply
    auto t_server_start = absl::Now();
    auto t_db_start = absl::Now();
    ASSIGN_OR_RETURN(
        auto b_prime_all,
        combined_db_matrix.template Multiply<CoeffType>(flattened_first_dim));
    auto cur_db_mult_time = absl::Now() - t_db_start;

    // Step 2.2: Vectorize Y_g
    auto t_pack_start = absl::Now();
    const int k_gadget = y_vec_g.size();
    std::vector<CoeffType> y_vec_g_vec(k_gadget * d);
    for (int l = 0; l < k_gadget; ++l) {
      for (int m = 0; m < d; ++m) {
        y_vec_g_vec[l * d + m] = y_vec_g[l].Coeffs()[m];
      }
    }

    // Step 2.3: Packing Matrix Multiply
    auto t_pack_mat_start = absl::Now();
    ASSIGN_OR_RETURN(
        auto b_agg_partial_all,
        combined_pack_matrix.template Multiply<CoeffType>(y_vec_g_vec));
    auto cur_pack_mat_mult_time = absl::Now() - t_pack_mat_start;

    // Step 2.4: Finalize Matrix Pack across shards
    auto t_pack_finalize_start = absl::Now();
    std::vector<RlweCiphertext<CoeffType>> shard_responses;
    shard_responses.reserve(entry_size_multiple);

    for (int k = 0; k < entry_size_multiple; ++k) {
      const size_t offset = static_cast<size_t>(k) * d;
      std::vector<CoeffType> chunk(b_prime_all.begin() + offset,
                                   b_prime_all.begin() + offset + d);
      std::vector<CoeffType> b_agg_partial(
          b_agg_partial_all.begin() + offset,
          b_agg_partial_all.begin() + offset + d);

      ASSIGN_OR_RETURN(
          auto pack_result,
          FinalizeMatrixPack(params.RlweParameters(), chunk, b_agg_partial,
                             y_vec_h, preprocessed_outputs[k][0].t_vec_h,
                             preprocessed_outputs[k][0].a_tilde_agg,
                             params.PackGadgetParams(), *fft_ctx));
      shard_responses.push_back(std::move(pack_result));
    }
    auto cur_pack_finalize_time = absl::Now() - t_pack_finalize_start;

    auto cur_pack_total_time = absl::Now() - t_pack_start;
    auto cur_server_proc_resp_time = absl::Now() - t_server_start;

    // Construct PIR response
    PirResponse<CoeffType> resp{std::move(shard_responses)};

    // Client process response
    auto t_client_proc_start = absl::Now();
    ASSIGN_OR_RETURN(auto rec_msg_iter, client->ProcessResponse(resp));
    auto cur_client_proc_time = absl::Now() - t_client_proc_start;

    rec_msg = std::move(rec_msg_iter);

    if (!is_warmup) {
      client_query_gen_time += cur_client_gen_time;
      server_proc_resp_time += cur_server_proc_resp_time;
      server_db_mult_time += cur_db_mult_time;
      server_packing_total_time += cur_pack_total_time;
      server_pack_matrix_mult_time += cur_pack_mat_mult_time;
      server_pack_finalize_time += cur_pack_finalize_time;
      client_proc_resp_time += cur_client_proc_time;
    }
  }

  // Average over measured iterations
  client_query_gen_time /= iterations;
  server_proc_resp_time /= iterations;
  server_db_mult_time /= iterations;
  server_packing_total_time /= iterations;
  server_pack_matrix_mult_time /= iterations;
  server_pack_finalize_time /= iterations;
  client_proc_resp_time /= iterations;

  // Verification check
  bool verification_success = true;
  size_t mismatches = 0;
  if (!absl::GetFlag(FLAGS_skip_verification)) {
    const size_t target_offset =
        static_cast<size_t>(kTargetIndex) * d * entry_size_multiple;
    for (size_t i = 0; i < static_cast<size_t>(d * entry_size_multiple); ++i) {
      if (rec_msg[i] != db_raw[target_offset + i]) {
        verification_success = false;
        mismatches++;
      }
    }
  }

  std::string label = absl::GetFlag(FLAGS_label);
  if (label.empty()) {
    label = absl::StrCat("PACKING_BENCH_ell", entry_size_multiple, "_z",
                         pack_log_digit);
  }

  std::ostream* out_stream = &std::cout;
  std::ofstream file_stream;
  if (!absl::GetFlag(FLAGS_output_file).empty()) {
    file_stream.open(absl::GetFlag(FLAGS_output_file), std::ios::app);
    if (file_stream.is_open()) {
      out_stream = &file_stream;
    }
  }

  *out_stream << "=== BENCHMARK_RESULT: " << label << " ===\n";
  *out_stream << "num_entries=" << n << "\n";
  *out_stream << "degree=" << d << "\n";
  *out_stream << "entry_size_multiple=" << entry_size_multiple << "\n";
  *out_stream << "interpolation_degree=" << kInterpolationDegree << "\n";
  *out_stream << "plaintext_modulus=" << kPlaintextModulus << "\n";
  *out_stream << "pack_log_digit=" << pack_log_digit << "\n";
  *out_stream << "pack_num_digits=" << pack_num_digits << "\n";
  *out_stream << "warmup_iterations=" << warmup_iterations << "\n";
  *out_stream << "iterations=" << iterations << "\n";
  if (absl::GetFlag(FLAGS_skip_verification)) {
    *out_stream << "verification_status=SKIPPED\n";
    *out_stream << "verification_mismatches=0\n";
  } else {
    *out_stream << "verification_status="
                << (verification_success ? "SUCCESS" : "FAILED") << "\n";
    *out_stream << "verification_mismatches=" << mismatches << "\n";
  }

  // Preprocessing metrics
  *out_stream << std::fixed << std::setprecision(4);
  *out_stream << "prep_server_time_ms="
              << absl::ToDoubleMilliseconds(prep_server_time) << "\n";
  *out_stream << "prep_server_time_us="
              << absl::ToDoubleMicroseconds(prep_server_time) << "\n";
  *out_stream << "prep_server_pack_time_ms="
              << absl::ToDoubleMilliseconds(
                     prep_server->GetPreprocessPackTime())
              << "\n";
  *out_stream << "prep_server_pack_time_us="
              << absl::ToDoubleMicroseconds(
                     prep_server->GetPreprocessPackTime())
              << "\n";

  // Online metrics
  *out_stream << "client_query_gen_time_ms="
              << absl::ToDoubleMilliseconds(client_query_gen_time) << "\n";
  *out_stream << "client_query_gen_time_us="
              << absl::ToDoubleMicroseconds(client_query_gen_time) << "\n";

  *out_stream << "server_proc_resp_time_ms="
              << absl::ToDoubleMilliseconds(server_proc_resp_time) << "\n";
  *out_stream << "server_proc_resp_time_us="
              << absl::ToDoubleMicroseconds(server_proc_resp_time) << "\n";

  *out_stream << "server_db_mult_time_ms="
              << absl::ToDoubleMilliseconds(server_db_mult_time) << "\n";
  *out_stream << "server_db_mult_time_us="
              << absl::ToDoubleMicroseconds(server_db_mult_time) << "\n";

  *out_stream << "server_packing_total_time_ms="
              << absl::ToDoubleMilliseconds(server_packing_total_time) << "\n";
  *out_stream << "server_packing_total_time_us="
              << absl::ToDoubleMicroseconds(server_packing_total_time) << "\n";

  *out_stream << "server_pack_matrix_mult_time_ms="
              << absl::ToDoubleMilliseconds(server_pack_matrix_mult_time)
              << "\n";
  *out_stream << "server_pack_matrix_mult_time_us="
              << absl::ToDoubleMicroseconds(server_pack_matrix_mult_time)
              << "\n";

  *out_stream << "server_pack_finalize_time_ms="
              << absl::ToDoubleMilliseconds(server_pack_finalize_time) << "\n";
  *out_stream << "server_pack_finalize_time_us="
              << absl::ToDoubleMicroseconds(server_pack_finalize_time) << "\n";

  *out_stream << "client_proc_resp_time_ms="
              << absl::ToDoubleMilliseconds(client_proc_resp_time) << "\n";
  *out_stream << "client_proc_resp_time_us="
              << absl::ToDoubleMicroseconds(client_proc_resp_time) << "\n";

  *out_stream << "========================================\n\n";
  out_stream->flush();
  return absl::OkStatus();
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (auto status = private_membership::rlwe::v2::RunPackingMetrics();
      !status.ok()) {
    LOG(FATAL) << "PIR packing metrics tool failed: " << status;
  }
  return 0;
}
