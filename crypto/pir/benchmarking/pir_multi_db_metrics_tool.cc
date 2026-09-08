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

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#include "crypto/pir/server/pir_preprocessed_data.h"
#include "crypto/pir/server/pir_server.h"
#include "crypto/pir/server/preprocessing_server.h"
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

ABSL_FLAG(int, num_entries, 262144,
          "Number of entries per database shard (n).");
ABSL_FLAG(int, entry_size_multiple, 1, "Number of database shards.");
ABSL_FLAG(int, degree, 2048, "RLWE ring degree (d).");
ABSL_FLAG(int, interpolation_degree, 2, "Interpolation degree (t).");
ABSL_FLAG(int, iterations, 5, "Number of timing iterations for online phase.");
ABSL_FLAG(std::string, output_file, "", "File to append metrics to.");
ABSL_FLAG(std::string, label, "", "Benchmark run label.");
ABSL_FLAG(int, pack_log_digit, 19, "Gadget log digit for packing.");
ABSL_FLAG(int, pack_num_digits, 2, "Number of gadget digits for packing.");
ABSL_FLAG(bool, skip_verification, false,
          "If true, skip correctness verification check.");
#include "absl/random/random.h"

ABSL_FLAG(std::string, disk_storage_file, "",
          "Path to write and read the precomputed DB array to/from disk.");
ABSL_FLAG(int, num_databases, 1,
          "Number of databases to create on disk for the multi-db simulation.");

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using DbDataType = uint16_t;
using CoeffType = uint64_t;
using MatCoeffType = int32_t;

// General RLWE parameters
constexpr uint64_t kModulus = 1ULL << 56;
constexpr int kVariance = 6;

// Online Phase Parameters
constexpr int kPlaintextModulus = 65535;
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

template <typename CoeffType>
size_t MatrixByteSize(const Matrix<CoeffType>& m) {
  return m.Data().size() * sizeof(CoeffType) +
         m.CondensedData().size() * sizeof(uint64_t);
}

absl::Status RunMetrics() {
  const int n = absl::GetFlag(FLAGS_num_entries);
  const int entry_size_multiple = absl::GetFlag(FLAGS_entry_size_multiple);
  const int d = absl::GetFlag(FLAGS_degree);
  const int interpolation_degree = absl::GetFlag(FLAGS_interpolation_degree);
  const int iterations = absl::GetFlag(FLAGS_iterations);
  int log_p = absl::bit_width(static_cast<uint32_t>(kPlaintextModulus));
  int log_d = absl::bit_width(static_cast<uint32_t>(d)) - 1;
  uint64_t mod1_after_switch = 1ULL << (log_p + log_d + 2);
  uint64_t mod2_after_switch = 1ULL << (log_p + 2);

  const int pack_log_digit = absl::GetFlag(FLAGS_pack_log_digit);
  const int pack_num_digits = absl::GetFlag(FLAGS_pack_num_digits);

  LOG(INFO) << "Starting PIR Metrics Run: d=" << d << ", n=" << n
            << ", entry_size_multiple=" << entry_size_multiple
            << ", interpolation_degree=" << interpolation_degree
            << ", q1_bits=" << (log_p + log_d + 2)
            << ", q2_bits=" << (log_p + 2)
            << ", pack_log_digit=" << pack_log_digit
            << ", pack_num_digits=" << pack_num_digits;

  ASSIGN_OR_RETURN(auto rlwe_params,
                   RlweParams<CoeffType>::Create(
                       d, kModulus, mod1_after_switch, mod2_after_switch,
                       kVariance));

  GadgetParams pack_gadget{pack_log_digit, pack_num_digits};
  GadgetParams poly_eval_gadget{kEvalLogDigit, kEvalNumDigits};

  ASSIGN_OR_RETURN(auto params,
                   PirParams<CoeffType>::Create(
                       rlwe_params, kPlaintextModulus, pack_gadget,
                       poly_eval_gadget, n, interpolation_degree,
                       entry_size_multiple));

  // Generate random DB
  const size_t total_db_bytes =
      static_cast<size_t>(n) * d * entry_size_multiple * sizeof(DbDataType);
  LOG(INFO) << "Database size: " << total_db_bytes / (1024.0 * 1024.0) << " MB";
  std::vector<DbDataType> db_raw(static_cast<size_t>(n) * d *
                                 entry_size_multiple);
  ASSIGN_OR_RETURN(auto db_prng, CreatePrng());
  for (size_t i = 0; i < db_raw.size(); ++i) {
    ASSIGN_OR_RETURN(auto rand_val, db_prng->Rand64());
    db_raw[i] = rand_val % kPlaintextModulus;
  }

  // 1. Preprocessing Server Runtime
  LOG(INFO) << "[Step 1/3] Running Preprocessing Server on DB...";
  ASSIGN_OR_RETURN(
      auto prep_server,
      (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          &params)));
  absl::Time start_time = absl::Now();
  ASSIGN_OR_RETURN(auto preprocessed_data, prep_server->Preprocess(db_raw));
  absl::Duration prep_server_time = absl::Now() - start_time;

  // Save preprocessed server material and database to disk (if requested).
  std::string db_file_path;
  size_t server_disk_storage_bytes = 0;

  db_file_path = absl::GetFlag(FLAGS_disk_storage_file);
  if (db_file_path.empty()) {
    db_file_path =
        absl::StrCat("/tmp/pir_server_db_", getpid(), "_", entry_size_multiple,
                     "_", interpolation_degree, "_", n, ".bin");
  }

  int num_dbs = absl::GetFlag(FLAGS_num_databases);
  LOG(INFO) << "Saving preprocessed server material to " << num_dbs
            << " separate database files at prefix: " << db_file_path;
  ASSIGN_OR_RETURN(
      auto initial_server,
      (PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params, std::move(preprocessed_data))));

  for (int i = 0; i < num_dbs; ++i) {
    std::string path_i = absl::StrCat(db_file_path, "_", i);
    struct stat buffer;
    if (::stat(path_i.c_str(), &buffer) == 0 && buffer.st_size > 0) {
      // File exists, skip saving
      continue;
    }
    RETURN_IF_ERROR(initial_server->SaveToFile(path_i));
  }

  std::string first_db = absl::StrCat(db_file_path, "_0");
  std::ifstream file_stat(first_db, std::ios::binary | std::ios::ate);
  if (file_stat.is_open()) {
    server_disk_storage_bytes = static_cast<size_t>(file_stat.tellg());
    file_stat.close();
  }
  LOG(INFO) << "Size of ONE stored database on disk: "
            << server_disk_storage_bytes / (1024.0 * 1024.0) << " MB";

  initial_server.reset();

  // 2. Server Stored Material Size
  // Calculate ideal server stored material size in bits and bytes (excluding
  // database size).
  const uint64_t bits_q = absl::bit_width(static_cast<uint64_t>(kModulus - 1));
  const uint64_t bits_d = absl::bit_width(static_cast<uint64_t>(d - 1));
  const uint64_t server_store_ideal_bits =
      static_cast<uint64_t>(entry_size_multiple) *
      static_cast<uint64_t>(interpolation_degree) *
      (static_cast<uint64_t>(d) * pack_num_digits * d *
           (bits_d + pack_log_digit) +
       static_cast<uint64_t>(pack_num_digits) * d * bits_q);
  const size_t server_preprocessed_ideal_bytes =
      static_cast<size_t>((server_store_ideal_bits + 7) / 8);

  // 3. Online Phase Timing and Message Sizes
  LOG(INFO) << "[Step 3/3] Running Online Phase timing iterations...";
  absl::Duration client_query_gen_time = absl::ZeroDuration();
  absl::Duration server_data_load_time = absl::ZeroDuration();
  absl::Duration server_proc_resp_time = absl::ZeroDuration();
  absl::Duration client_proc_resp_time = absl::ZeroDuration();

  absl::Duration server_db_mult_time = absl::ZeroDuration();
  absl::Duration server_packing_time = absl::ZeroDuration();
  absl::Duration server_poly_eval_time = absl::ZeroDuration();

  std::unique_ptr<PirRequest<CoeffType>> req;
  std::unique_ptr<PirResponse<CoeffType>> resp;
  std::vector<CoeffType> rec_msg;

  // Pre-allocate the memory pool for DB fetches.
  uint8_t* preallocated_buffer_ptr = nullptr;
  LOG(INFO) << "Preallocating " << server_disk_storage_bytes / (1024.0 * 1024)
            << " MB for sequential read pool.";
  int ret = posix_memalign(reinterpret_cast<void**>(&preallocated_buffer_ptr),
                           4096, server_disk_storage_bytes);
  if (ret != 0) {
    return absl::InternalError("posix_memalign failed");
  }

  for (int iter = 0; iter < iterations; ++iter) {
    ASSIGN_OR_RETURN(auto client_a_prng,
                     CreatePrng(prep_server->GetPrngSeed()));
    ASSIGN_OR_RETURN(auto client_prng, CreatePrng());
    ASSIGN_OR_RETURN(
        auto client,
        (PirClient<CoeffType>::Create(params, std::move(client_a_prng),
                                     std::move(client_prng))));

    start_time = absl::Now();
    ASSIGN_OR_RETURN(auto req_val, client->CreateRequest(kTargetIndex));
    req = std::make_unique<PirRequest<CoeffType>>(std::move(req_val));
    client_query_gen_time += (absl::Now() - start_time);

    std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>
        loaded_server;
    PirServer<DbDataType, CoeffType, MatCoeffType>* server_ptr = nullptr;

    absl::BitGen bitgen;
    int db_idx = absl::Uniform(bitgen, 0, num_dbs);
    std::string chosen_db_path = absl::StrCat(db_file_path, "_", db_idx);

    start_time = absl::Now();
    int fd = open(chosen_db_path.c_str(), O_RDONLY);
    if (fd >= 0) {
      posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    }
    if (fd < 0) {
      if (preallocated_buffer_ptr) free(preallocated_buffer_ptr);
      return absl::InternalError(
          absl::StrCat("Failed to open ", chosen_db_path));
    }
    size_t total_read = 0;
    while (total_read < server_disk_storage_bytes) {
      if (preallocated_buffer_ptr == nullptr) {
        close(fd);
        return absl::InternalError("preallocated_buffer_ptr is null");
      }
      ssize_t bytes = read(fd, preallocated_buffer_ptr + total_read,
                           server_disk_storage_bytes - total_read);
      if (bytes < 0) {
        if (errno == EINTR || errno == EAGAIN) continue;
        close(fd);
        if (preallocated_buffer_ptr) free(preallocated_buffer_ptr);
        return absl::InternalError(
            absl::StrCat("Read failed with errno ", errno));
      }
      if (bytes == 0) break;
      total_read += bytes;
    }
    close(fd);
    LOG(INFO) << "total_read=" << total_read
              << " expected=" << server_disk_storage_bytes;
    ASSIGN_OR_RETURN(
        loaded_server,
        (PirServer<DbDataType, CoeffType, MatCoeffType>::LoadFromBuffer(
            params, preallocated_buffer_ptr, server_disk_storage_bytes)));
    server_data_load_time += (absl::Now() - start_time);
    server_ptr = loaded_server.get();

    // Server processes response.
    start_time = absl::Now();
    ASSIGN_OR_RETURN(auto resp_val, server_ptr->ProcessResponse(*req));
    resp = std::make_unique<PirResponse<CoeffType>>(std::move(resp_val));
    server_proc_resp_time += (absl::Now() - start_time);

    server_db_mult_time += server_ptr->GetDbMultiplyTime();
    server_packing_time += server_ptr->GetPackingTime();
    server_poly_eval_time += server_ptr->GetPolyEvalTime();

    loaded_server.reset();

    start_time = absl::Now();
    ASSIGN_OR_RETURN(rec_msg, client->ProcessResponse(*resp));
    client_proc_resp_time += (absl::Now() - start_time);
  }

  client_query_gen_time /= iterations;
  server_data_load_time /= iterations;
  server_proc_resp_time /= iterations;
  absl::Duration server_response_latency =
      server_data_load_time + server_proc_resp_time;
  client_proc_resp_time /= iterations;

  bool verification_success = true;
  size_t mismatches = 0;
  if (!absl::GetFlag(FLAGS_skip_verification)) {
    for (size_t i = 0; i < static_cast<size_t>(d) * entry_size_multiple; ++i) {
      if (rec_msg[i] != db_raw[i]) {
        verification_success = false;
        mismatches++;
      }
    }
  }
  // C2S online query size (Ideal bit-packed size)
  const uint64_t c2s_on_ideal_bits =
      (static_cast<uint64_t>(n) / (interpolation_degree * d) +
       (interpolation_degree > 1 ? 2ULL * kEvalNumDigits : 0ULL) +
       2ULL * pack_num_digits) *
      d * bits_q;
  const size_t c2s_on_ideal_bytes =
      static_cast<size_t>((c2s_on_ideal_bits + 7) / 8);

  // S2C online response size (Ideal bit-packed size)
  const uint64_t s2c_on_ideal_bits =
      static_cast<uint64_t>(entry_size_multiple) * d * (log_p + 2);
  const size_t s2c_on_ideal_bytes =
      static_cast<size_t>((s2c_on_ideal_bits + 7) / 8);

  std::ostream* out_stream = &std::cout;
  std::ofstream file_stream;
  std::string output_file = absl::GetFlag(FLAGS_output_file);
  if (!output_file.empty()) {
    file_stream.open(output_file, std::ios::app);
    if (file_stream.is_open()) {
      out_stream = &file_stream;
    }
  }

  *out_stream << "=== PIR Multi-DB Benchmark Results ===" << "\n";
  *out_stream << "scheme=pir\n";
  *out_stream << "database_size_mb=" << total_db_bytes / (1024.0 * 1024.0)
              << "\n";
  *out_stream << "entry_size_bytes=" << d * entry_size_multiple * 2 << "\n";
  *out_stream << "num_entries=" << n << "\n";
  *out_stream << "num_databases=" << num_dbs << "\n";
  *out_stream << "degree=" << d << "\n";
  *out_stream << "interpolation_degree=" << interpolation_degree << "\n";
  *out_stream << "entry_size_multiple=" << entry_size_multiple << "\n";
  *out_stream << "is_correct=" << (verification_success ? "true" : "false")
              << "\n";
  *out_stream << "mismatches=" << mismatches << "\n";
  *out_stream << "c2s_query_ideal_bytes=" << c2s_on_ideal_bytes << "\n";
  *out_stream << "s2c_response_ideal_bytes=" << s2c_on_ideal_bytes << "\n";
  *out_stream << "server_preprocessed_ideal_bytes="
              << server_preprocessed_ideal_bytes << "\n";
  *out_stream << "server_disk_storage_bytes=" << server_disk_storage_bytes
              << "\n";
  *out_stream << "prep_server_time_ms="
              << absl::ToDoubleMilliseconds(prep_server_time) << "\n";
  *out_stream << "client_query_gen_time_ms="
              << absl::ToDoubleMilliseconds(client_query_gen_time) << "\n";
  *out_stream << "server_data_load_time_ms="
              << absl::ToDoubleMilliseconds(server_data_load_time) << "\n";
  *out_stream << "server_proc_resp_time_ms="
              << absl::ToDoubleMilliseconds(server_proc_resp_time) << "\n";
  *out_stream << "server_response_latency_ms="
              << absl::ToDoubleMilliseconds(server_response_latency) << "\n";
  *out_stream << "client_proc_resp_time_ms="
              << absl::ToDoubleMilliseconds(client_proc_resp_time) << "\n";
  *out_stream << "========================================\n\n";
  out_stream->flush();

  // Clean up temporary disk file if generated automatically.
  if (absl::GetFlag(FLAGS_disk_storage_file).empty()) {
    std::remove(db_file_path.c_str());
  }
  if (preallocated_buffer_ptr) free(preallocated_buffer_ptr);
  return absl::OkStatus();
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = private_membership::rlwe::v2::RunMetrics();
  if (!status.ok()) {
    LOG(ERROR) << "RunMetrics failed: " << status;
    return 1;
  }
  return 0;
}
