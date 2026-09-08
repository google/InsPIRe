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
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/matrix.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/pir_params.h"
#include "crypto/polynomial.h"
#include "crypto/vpir/client/vpir_client.h"
#include "crypto/vpir/server/vpir_preprocessing_server.h"
#include "crypto/vpir/server/vpir_server.h"
#include "crypto/vpir/vpir_params.h"
#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "crypto/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/flags/parse.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

ABSL_FLAG(int, num_entries, 2048, "Number of entries per database shard (n).");
ABSL_FLAG(int, entry_size_multiple, 256, "Number of database shards.");
ABSL_FLAG(int, degree, 2048, "RLWE ring degree (d).");
ABSL_FLAG(int, kappa, 40, "Number of verifiable proof queries.");
ABSL_FLAG(int, iterations, 5, "Number of timing iterations for online phase.");
ABSL_FLAG(std::string, output_file, "", "File to append metrics to.");
ABSL_FLAG(std::string, label, "", "Benchmark run label.");
ABSL_FLAG(bool, skip_verification, false,
          "If true, skip correctness verification check.");
ABSL_FLAG(bool, local_finalize, false,
          "If true, activate the local_finalize variant of VPIR.");

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

// --- Main PIR (Online Phase) Type Parameters ---
using MainDbDataType = uint8_t;
using MainCoeffType = uint32_t;
using MainMatCoeffType = int32_t;

// --- Z-PIR (Offline Proof Phase) Type Parameters ---
using ZDbDataType = uint8_t;
using ZCoeffType = uint64_t;
using ZMatCoeffType = int32_t;

// --- Offline Phase Parameters (Proof Generation / Z-PIR) ---
constexpr uint64_t kZModulus = (1ULL << 54);
constexpr int kZVariance = 40;
// Z-PIR Packing gadget parameters
constexpr uint64_t kZPlaintextModulus = 1ULL << 24;
constexpr int kZPackLogDigit = 11;
constexpr int kZPackNumDigits = 5;

// --- Online Phase Parameters (Query Response / Main PIR) ---
constexpr uint64_t kModulus = 0;
constexpr int kVariance = 11;
constexpr int kPlaintextModulus = 256;

// Main Packing gadget parameters (used to pack database elements)
constexpr int kPackLogDigit = 8;
constexpr int kPackNumDigits = 4;

constexpr int kTargetIndex = 123;

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng(
    absl::string_view prng_seed) {
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() {
  ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

absl::Status RunMetrics() {
  const int d = absl::GetFlag(FLAGS_degree);
  const int n = absl::GetFlag(FLAGS_num_entries);
  const int entry_size_multiple = absl::GetFlag(FLAGS_entry_size_multiple);
  const int kappa = absl::GetFlag(FLAGS_kappa);
  const int iterations = absl::GetFlag(FLAGS_iterations);

  LOG(INFO) << "Starting VPIR Metrics Run: d=" << d << ", n=" << n
            << ", entry_size_multiple=" << entry_size_multiple
            << ", kappa=" << kappa;

  int log_p = absl::bit_width(static_cast<uint32_t>(kZPlaintextModulus));
  int log_d = absl::bit_width(static_cast<uint32_t>(d)) - 1;
  uint64_t mod1_after_switch = 1ULL << (log_p + log_d + 2);
  uint64_t mod2_after_switch = 1ULL << (log_p + 2);

  ASSIGN_OR_RETURN(auto rlwe_params,
                   RlweParams<MainCoeffType>::Create(d, kModulus, kModulus,
                                                     kModulus, kVariance));

  ASSIGN_OR_RETURN(auto z_rlwe_params, RlweParams<ZCoeffType>::Create(
                                           d, kZModulus, mod1_after_switch,
                                           mod2_after_switch, kZVariance));

  GadgetParams pack_gadget{kPackLogDigit, kPackNumDigits};
  GadgetParams z_pack_gadget{kZPackLogDigit, kZPackNumDigits};

  ASSIGN_OR_RETURN(
      auto params,
      (VpirParams<MainCoeffType, ZCoeffType>::Create(
          rlwe_params, kPlaintextModulus, pack_gadget, n, entry_size_multiple,
          /*z_interpolation_degree=*/1, z_pack_gadget,
          /*z_poly_eval_gadget_params=*/std::nullopt, kZPlaintextModulus,
          z_rlwe_params, absl::GetFlag(FLAGS_local_finalize))));

  // Generate random DB
  const size_t total_db_bytes = static_cast<size_t>(n) * d *
                                entry_size_multiple * sizeof(MainDbDataType);
  LOG(INFO) << "Database size: "
            << total_db_bytes / (1024.0 * 1024.0) << " MB";
  std::vector<MainDbDataType> db_raw(static_cast<size_t>(n) * d *
                                     entry_size_multiple);
  ASSIGN_OR_RETURN(auto db_prng, CreatePrng());
  for (size_t i = 0; i < db_raw.size(); ++i) {
    ASSIGN_OR_RETURN(auto rand_val, db_prng->Rand8());
    db_raw[i] = rand_val % kPlaintextModulus;
  }

  // 1. Preprocessing Server Runtime
  LOG(INFO) << "[Step 1/6] Running Preprocessing Server on DB...";
  ASSIGN_OR_RETURN(
      auto prep_server,
      (VpirPreprocessingServer<MainDbDataType, MainCoeffType, MainMatCoeffType,
                               ZDbDataType, ZCoeffType,
                               ZMatCoeffType>::Create(&params)));
  absl::Time start_time = absl::Now();
  ASSIGN_OR_RETURN(auto preprocessed_data, prep_server->Preprocess(db_raw));
  absl::Duration prep_server_time = absl::Now() - start_time;

  ASSIGN_OR_RETURN(
      auto server,
      (VpirServer<MainDbDataType, MainCoeffType, MainMatCoeffType, ZDbDataType,
                  ZCoeffType, ZMatCoeffType>::Create(params,
                                                     std::move(
                                                         preprocessed_data))));

  ASSIGN_OR_RETURN(auto client_a_prng, CreatePrng(server->GetPrngSeed()));
  ASSIGN_OR_RETURN(auto client_prng, CreatePrng());
  ASSIGN_OR_RETURN(auto client, (VpirClient<MainCoeffType, ZCoeffType>::Create(
                                    params, std::move(client_a_prng),
                                    std::move(client_prng))));

  // Generate offline requests from client
  LOG(INFO) << "[Step 2/6] Generating offline requests from client (" << kappa
            << " queries for vector length " << entry_size_multiple * d
            << ")...";
  ASSIGN_OR_RETURN(auto z_client_a_prng, CreatePrng(server->GetPrngSeed()));
  ASSIGN_OR_RETURN(auto z_client_prng, CreatePrng());
  ASSIGN_OR_RETURN(auto z_pir_client,
                   PirClient<ZCoeffType>::Create(params.ZPirParams(),
                                                 std::move(z_client_a_prng),
                                                 std::move(z_client_prng)));

  ASSIGN_OR_RETURN(auto c_prng, CreatePrng());
  const int total_entry_coeffs = entry_size_multiple * d;
  std::vector<std::vector<MainCoeffType>> c_data(
      kappa, std::vector<MainCoeffType>(total_entry_coeffs));
  for (int i = 0; i < kappa; ++i) {
    for (int j = 0; j < total_entry_coeffs; ++j) {
      ASSIGN_OR_RETURN(auto rand_val, c_prng->Rand8());
      c_data[i][j] = rand_val % 2;
    }
  }
  ASSIGN_OR_RETURN(auto c_matrix,
                   Matrix<MainCoeffType>::Create(std::move(c_data)));

  absl::Time client_off_req_start = absl::Now();
  std::vector<PirRequest<ZCoeffType>> z_requests;
  z_requests.reserve(kappa);
  for (int i = 0; i < kappa; ++i) {
    std::vector<ZCoeffType> row_vec(total_entry_coeffs);
    for (int j = 0; j < total_entry_coeffs; ++j) {
      row_vec[j] = static_cast<ZCoeffType>(
          c_matrix.Data()[i * total_entry_coeffs + j]);
    }
    ASSIGN_OR_RETURN(auto z_req,
                     z_pir_client->CreateRequestForVector(row_vec));
    z_requests.push_back(std::move(z_req));
  }
  absl::Duration client_off_req_time = absl::Now() - client_off_req_start;

  // 2. Offline Server Proof Gen Runtime
  LOG(INFO) << "[Step 3/6] Running Offline Server Proof Gen ("
            << kappa << " queries)...";
  start_time = absl::Now();
  ASSIGN_OR_RETURN(auto proof_material,
                   server->GetProofPreprocessMaterial(z_requests));
  absl::Duration server_off_proof_time = absl::Now() - start_time;

  // 3. Offline Client Proof Verification Runtime
  LOG(INFO)
      << "[Step 4/6] Running Offline Client Proof Verification & Processing...";
  start_time = absl::Now();
  RETURN_IF_ERROR(client->GetProofPreprocessMaterial(kappa, *server));
  absl::Duration client_off_time =
      (absl::Now() - start_time) + client_off_req_time;

  // Helper to robustly determine bit width of a modulus (where 0 means full
  // ring 2^bitwidth)
  const int bits_modulus = params.RlweParameters().LogModulus();
  const int bits_z_modulus = params.ZPirParams().RlweParameters().LogModulus();
  const int bits_q1 =
      params.ZPirParams().RlweParameters().LogModulus1AfterSwitch();
  const int bits_q2 =
      params.ZPirParams().RlweParameters().LogModulus2AfterSwitch();

  // 4. Client to server offline message size (z_requests)
  LOG(INFO) << "[Step 5/6] Measuring message sizes and storage footprints...";
  const uint64_t c2s_off_bits =
      static_cast<uint64_t>(entry_size_multiple) * kappa * d * bits_z_modulus;
  const size_t c2s_off_formula = static_cast<size_t>((c2s_off_bits + 7) / 8);

  // 5. Server to client offline message size (VpirProofPreprocessMaterial)
  // 5a. Static client-independent hint portion, only hint_matrix
  const uint64_t s2c_off_hint_coeffs =
      static_cast<uint64_t>(entry_size_multiple) * d * d;
  const uint64_t s2c_off_hint_bits = s2c_off_hint_coeffs * bits_modulus;
  const size_t s2c_off_hint_formula =
      static_cast<size_t>((s2c_off_hint_bits + 7) / 8);

  // 5b. Dynamic proof response portion (Client-Dependent: z_responses)
  const uint64_t s2c_off_proof_bits =
      static_cast<uint64_t>(kappa) * n *
      static_cast<uint64_t>(bits_q1 + bits_q2);
  const size_t s2c_off_proof_formula =
      static_cast<size_t>((s2c_off_proof_bits + 7) / 8);

  const size_t s2c_off_formula = s2c_off_hint_formula + s2c_off_proof_formula;

  // 6. Client stored material size
  const bool local_finalize = absl::GetFlag(FLAGS_local_finalize);
  const int k_gadget = local_finalize ? kPackNumDigits : 2 * kPackNumDigits;
  const uint64_t p = params.PlaintextModulus();
  const int bits_q = bits_modulus;

  const uint64_t max_val_p =
      (p == 0) ? ((sizeof(MainCoeffType) == 8)
                      ? UINT64_MAX
                      : (1ULL << (sizeof(MainCoeffType) * 8)) - 1)
               : (p - 1);
  const uint64_t max_z_val =
      static_cast<uint64_t>(total_entry_coeffs) * max_val_p;
  const int bits_z = absl::bit_width(max_z_val);

  int bits_z_prime = 0;
  if (bits_q >= 64) {
    bits_z_prime =
        absl::bit_width(static_cast<uint64_t>(total_entry_coeffs) - 1) + 64;
  } else {
    const uint64_t max_val_q = (1ULL << bits_q) - 1;
    const uint64_t max_z_prime_val =
        static_cast<uint64_t>(total_entry_coeffs) * max_val_q;
    bits_z_prime = absl::bit_width(max_z_prime_val);
  }

  const uint64_t total_bits_C =
      static_cast<uint64_t>(kappa) * total_entry_coeffs * 1;
  const uint64_t total_bits_Z = static_cast<uint64_t>(kappa) * n * bits_z;
  const uint64_t total_bits_Z_prime =
      static_cast<uint64_t>(kappa) * k_gadget * d * bits_z_prime;
  const uint64_t total_bits_prep =
      static_cast<uint64_t>(entry_size_multiple) * d * 1 * bits_q;
  const uint64_t total_bits_t_vec_h =
      local_finalize ? static_cast<uint64_t>(entry_size_multiple) *
                           kPackNumDigits * d * kPackLogDigit
                     : 0;

  const size_t client_store_formula = static_cast<size_t>(
      (total_bits_C + total_bits_Z + total_bits_Z_prime + total_bits_prep +
       total_bits_t_vec_h + 7) /
      8);

  // 7. Server stored material size
  const size_t server_store_formula =
      static_cast<size_t>(entry_size_multiple) *
      (2ULL * d * n * sizeof(ZDbDataType) +
       static_cast<size_t>(d) * kPackNumDigits * d * sizeof(MainMatCoeffType) +
       (local_finalize ? 0
                       : static_cast<size_t>(d) * (kPackNumDigits * d + d) *
                             sizeof(MainMatCoeffType)) +
       static_cast<size_t>(d) * (kPackNumDigits + 1) * sizeof(MainCoeffType) +
       static_cast<size_t>(kappa) * kPackNumDigits * d * sizeof(ZCoeffType) +
       static_cast<size_t>(n) * d * sizeof(ZCoeffType));

  // Online phase timing
  LOG(INFO) << "[Step 6/6] Running Online Phase timing iterations...";
  absl::Duration client_query_gen_time = absl::ZeroDuration();
  absl::Duration server_proc_resp_time = absl::ZeroDuration();
  absl::Duration client_resp_verify_time = absl::ZeroDuration();
  absl::Duration client_resp_decrypt_time = absl::ZeroDuration();
  absl::Duration client_proc_resp_time = absl::ZeroDuration();

  VpirRequest<MainCoeffType> req;
  VpirResponse<MainCoeffType> resp;
  std::vector<MainCoeffType> rec_msg;

  for (int iter = 0; iter < iterations; ++iter) {
    start_time = absl::Now();
    ASSIGN_OR_RETURN(req, client->CreateRequest(kTargetIndex));
    client_query_gen_time += (absl::Now() - start_time);

    start_time = absl::Now();
    ASSIGN_OR_RETURN(resp, server->ProcessResponse(req));
    server_proc_resp_time += (absl::Now() - start_time);

    start_time = absl::Now();
    RETURN_IF_ERROR(client->VerifyResponse(resp, req));
    client_resp_verify_time += (absl::Now() - start_time);

    start_time = absl::Now();
    ASSIGN_OR_RETURN(auto rec_msg_temp, client->DecryptResponse(resp));
    client_resp_decrypt_time += (absl::Now() - start_time);

    if (iter == 0) {
      rec_msg = std::move(rec_msg_temp);
    }
  }

  client_query_gen_time /= iterations;
  server_proc_resp_time /= iterations;
  client_resp_verify_time /= iterations;
  client_resp_decrypt_time /= iterations;
  client_proc_resp_time = client_resp_verify_time + client_resp_decrypt_time;

  bool verification_success = true;
  size_t mismatches = 0;
  if (!absl::GetFlag(FLAGS_skip_verification) && !rec_msg.empty()) {
    for (size_t i = 0; i < static_cast<size_t>(d) * entry_size_multiple; ++i) {
      if (rec_msg[i] != db_raw[kTargetIndex * d * entry_size_multiple + i]) {
        verification_success = false;
        mismatches++;
      }
    }
  }

  // 8. Client to server online message (query)
  const uint64_t c2s_on_bits =
      (static_cast<uint64_t>(n) + static_cast<uint64_t>(k_gadget) * d) *
      bits_modulus;
  const size_t c2s_on_formula = static_cast<size_t>((c2s_on_bits + 7) / 8);

  // 9. Server to client online message (response - only b_responses)
  const uint64_t s2c_on_bits =
      static_cast<uint64_t>(entry_size_multiple) * d * bits_modulus;
  const size_t s2c_on_formula = static_cast<size_t>((s2c_on_bits + 7) / 8);

  std::string label = absl::GetFlag(FLAGS_label);
  if (label.empty()) {
    label = absl::StrCat("DB_", total_db_bytes / (1024 * 1024), "MB_n", n,
                         "_ell", entry_size_multiple);
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
  *out_stream << "degree=" << d << "\n";
  *out_stream << "kappa=" << kappa << "\n";
  *out_stream << "db_size_mb=" << (total_db_bytes / (1024.0 * 1024.0)) << "\n";
  *out_stream << "num_entries=" << n << "\n";
  *out_stream << "entry_size_multiple=" << entry_size_multiple << "\n";
  *out_stream << "local_finalize=" << (local_finalize ? 1 : 0) << "\n";
  *out_stream << "z_modulus=" << kZModulus << "\n";
  *out_stream << "z_variance=" << kZVariance << "\n";
  *out_stream << "z_plaintext_modulus=" << kZPlaintextModulus << "\n";
  *out_stream << "z_pack_log_digit=" << kZPackLogDigit << "\n";
  *out_stream << "z_pack_num_digits=" << kZPackNumDigits << "\n";
  *out_stream << "modulus=" << kModulus << "\n";
  *out_stream << "variance=" << kVariance << "\n";
  *out_stream << "plaintext_modulus=" << kPlaintextModulus << "\n";
  *out_stream << "pack_log_digit=" << kPackLogDigit << "\n";
  *out_stream << "pack_num_digits=" << kPackNumDigits << "\n";
  *out_stream << "prep_server_time_ms="
              << absl::ToDoubleMilliseconds(prep_server_time) << "\n";
  *out_stream << "off_server_proof_time_ms="
              << absl::ToDoubleMilliseconds(server_off_proof_time) << "\n";
  *out_stream << "off_client_time_ms="
              << absl::ToDoubleMilliseconds(client_off_time) << "\n";
  *out_stream << "c2s_off_msg_bytes_formula=" << c2s_off_formula << "\n";
  *out_stream << "s2c_off_hint_msg_bytes_formula=" << s2c_off_hint_formula
              << "\n";
  *out_stream << "s2c_off_proof_msg_bytes_formula=" << s2c_off_proof_formula
              << "\n";
  *out_stream << "s2c_off_msg_bytes_formula=" << s2c_off_formula << "\n";
  *out_stream << "client_store_bytes_formula=" << client_store_formula << "\n";
  *out_stream << "server_store_bytes_formula=" << server_store_formula << "\n";
  *out_stream << "c2s_on_msg_bytes_formula=" << c2s_on_formula << "\n";
  *out_stream << "s2c_on_msg_bytes_formula=" << s2c_on_formula << "\n";
  *out_stream << "client_query_gen_time_ms="
              << absl::ToDoubleMilliseconds(client_query_gen_time) << "\n";
  *out_stream << "server_proc_resp_time_ms="
              << absl::ToDoubleMilliseconds(server_proc_resp_time) << "\n";
  *out_stream << "client_resp_verify_time_ms="
              << absl::ToDoubleMilliseconds(client_resp_verify_time) << "\n";
  *out_stream << "client_resp_decrypt_time_ms="
              << absl::ToDoubleMilliseconds(client_resp_decrypt_time) << "\n";
  *out_stream << "client_proc_resp_time_ms="
              << absl::ToDoubleMilliseconds(client_proc_resp_time) << "\n";
  *out_stream << "verification_status=" << (verification_success ? 1 : 0)
              << "\n";
  *out_stream << "verification_mismatches=" << mismatches << "\n";
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
  if (auto status = private_membership::rlwe::v2::RunMetrics(); !status.ok()) {
    LOG(FATAL) << "VPIR metrics tool failed: " << status;
  }
  return 0;
}
