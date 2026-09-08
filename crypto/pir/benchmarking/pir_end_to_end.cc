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
#include <string>
#include <utility>
#include <vector>

#include "crypto/db_row_interpolation.h"
#include "crypto/encryption.h"
#include "crypto/hint_computation.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_preprocessed_data.h"
#include "crypto/pir/server/pir_server.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "crypto/status_macros.h"
#include "absl/strings/string_view.h"
#include "absl/flags/parse.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using DbDataType = uint16_t;
using CoeffType = uint64_t;
using MatCoeffType = int32_t;

// Database parameters
constexpr int kNumEntries = 262144;
constexpr int kEntrySizeMultiple = 1;
constexpr int kDegree = 2048;
constexpr int kInterpolationDegree = 2;
constexpr int kPlaintextModulus = 65535;

// RLWE parameters
constexpr uint64_t kModulus = 1ULL << 52;
constexpr int kVariance = 40;

// Gadget parameters
constexpr int kPackLogDigit = 19;
constexpr int kPackNumDigits = 2;
constexpr int kEvalLogDigit = 19;
constexpr int kEvalNumDigits = 3;

// Target query index and verification
constexpr int kTargetIndex = 123;
constexpr bool kSkipVerification = false;

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng(
    absl::string_view prng_seed) {
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() {
  ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
  return ::rlwe::ChaChaPrng::Create(prng_seed);
}

absl::Status RunPirEndToEnd() {
  const int log_p = absl::bit_width(static_cast<uint32_t>(kPlaintextModulus));
  const int log_d = absl::bit_width(static_cast<uint32_t>(kDegree)) - 1;
  const uint64_t mod1_after_switch = 1ULL << (log_p + log_d + 2);
  const uint64_t mod2_after_switch = 1ULL << (log_p + 2);

  LOG(INFO) << "Initializing PIR parameters: d=" << kDegree
            << ", n=" << kNumEntries
            << ", entry_size_multiple=" << kEntrySizeMultiple
            << ", interpolation_degree=" << kInterpolationDegree;

  ASSIGN_OR_RETURN(
      auto rlwe_params,
      RlweParams<CoeffType>::Create(kDegree, kModulus, mod1_after_switch,
                                    mod2_after_switch, kVariance));

  GadgetParams pack_gadget{kPackLogDigit, kPackNumDigits};
  GadgetParams poly_eval_gadget{kEvalLogDigit, kEvalNumDigits};

  ASSIGN_OR_RETURN(
      auto params,
      PirParams<CoeffType>::Create(rlwe_params, kPlaintextModulus, pack_gadget,
                                   poly_eval_gadget, kNumEntries,
                                   kInterpolationDegree, kEntrySizeMultiple));

  // Generate random DB
  LOG(INFO) << "Generating random database...";
  std::vector<DbDataType> db_raw(static_cast<size_t>(kNumEntries) * kDegree *
                                 kEntrySizeMultiple);
  ASSIGN_OR_RETURN(auto db_prng, CreatePrng());
  for (size_t i = 0; i < db_raw.size(); ++i) {
    ASSIGN_OR_RETURN(auto rand_val, db_prng->Rand64());
    db_raw[i] = rand_val % kPlaintextModulus;
  }

  ///// Preprocessing Phase ////

  // Preprocess database directly inline (unpacked from PreprocessingServer)
  LOG(INFO) << "Preprocessing database...";
  ASSIGN_OR_RETURN(std::string prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
  ASSIGN_OR_RETURN(auto a_prng, ::rlwe::ChaChaPrng::Create(prng_seed));

  std::vector<Matrix<DbDataType>> db_matrices;
  db_matrices.reserve(kEntrySizeMultiple);

  for (size_t k = 0; k < kEntrySizeMultiple; ++k) {
    std::vector<DbDataType> db_shard_raw(static_cast<size_t>(kNumEntries) *
                                         kDegree);
    for (size_t idx = 0; idx < kNumEntries; ++idx) {
      for (size_t offset = 0; offset < kDegree; ++offset) {
        db_shard_raw[idx * static_cast<size_t>(kDegree) + offset] =
            db_raw[idx * static_cast<size_t>(kDegree) * kEntrySizeMultiple +
                   k * static_cast<size_t>(kDegree) + offset];
      }
    }

    // Interpolate and transpose shard
    std::vector<std::vector<DbDataType>> db_reshaped(
        kNumEntries / kInterpolationDegree,
        std::vector<DbDataType>(static_cast<size_t>(kInterpolationDegree) *
                                kDegree));
    for (size_t i = 0; i < kNumEntries / kInterpolationDegree; ++i) {
      auto start = db_shard_raw.begin() +
                   i * (static_cast<size_t>(kInterpolationDegree) * kDegree);
      db_reshaped[i].assign(
          start, start + static_cast<size_t>(kInterpolationDegree) * kDegree);
    }
    std::vector<std::vector<DbDataType>> db_interpolated(
        kNumEntries / kInterpolationDegree,
        std::vector<DbDataType>(static_cast<size_t>(kInterpolationDegree) *
                                kDegree));
    for (size_t i = 0; i < kNumEntries / kInterpolationDegree; ++i) {
      ASSIGN_OR_RETURN(
          db_interpolated[i],
          Interpolate(db_reshaped[i], kDegree, kInterpolationDegree,
                      static_cast<DbDataType>(kPlaintextModulus)));
    }
    std::vector<std::vector<DbDataType>> db_transposed(
        static_cast<size_t>(kInterpolationDegree) * kDegree,
        std::vector<DbDataType>(kNumEntries / kInterpolationDegree));
    for (size_t i = 0;
         i < static_cast<size_t>(kInterpolationDegree) * kDegree; ++i) {
      for (size_t j = 0; j < kNumEntries / kInterpolationDegree; ++j) {
        db_transposed[i][j] = db_interpolated[j][i];
      }
    }
    ASSIGN_OR_RETURN(auto db_matrix, Matrix<DbDataType>::Create(db_transposed));
    db_matrices.push_back(std::move(db_matrix));
  }

  // Sample 'a' components
  const int first_dim_a_samples = params.NumFirstDimSamples();
  ASSIGN_OR_RETURN(auto first_dim_a,
                   SampleAComponents(params.RlweParameters(),
                                     first_dim_a_samples, *a_prng));
  const int second_dim_a_samples =
      kInterpolationDegree > 1
          ? 2 * params.PolyEvalGadgetParams().num_digits
          : 0;
  std::vector<Polynomial<CoeffType>> second_dim_a;
  if (second_dim_a_samples > 0) {
    ASSIGN_OR_RETURN(second_dim_a,
                     SampleAComponents(params.RlweParameters(),
                                       second_dim_a_samples, *a_prng));
  }
  const int pack_a_samples = 2 * params.PackGadgetParams().num_digits;
  ASSIGN_OR_RETURN(auto pack_a, SampleAComponents(params.RlweParameters(),
                                                  pack_a_samples, *a_prng));

  // Compute preprocessed outputs
  ASSIGN_OR_RETURN(
      auto fft_ctx,
      FftContext::Create(absl::bit_width(static_cast<uint32_t>(kDegree)) - 1));

  std::vector<Polynomial<CoeffType>> w_vec_g(
      pack_a.begin(),
      pack_a.begin() + params.PackGadgetParams().num_digits);
  std::vector<Polynomial<CoeffType>> w_vec_h(
      pack_a.begin() + params.PackGadgetParams().num_digits, pack_a.end());

  const int padded_cols = params.PaddedCols();
  std::vector<std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
      preprocessed_outputs;
  preprocessed_outputs.reserve(kEntrySizeMultiple);

  for (size_t k = 0; k < kEntrySizeMultiple; ++k) {
    Matrix<DbDataType> db_matrix_for_hints;
    if (db_matrices[k].Cols() < padded_cols) {
      std::vector<std::vector<DbDataType>> padded_data(
          db_matrices[k].Rows(), std::vector<DbDataType>(padded_cols, 0));
      for (int i = 0; i < db_matrices[k].Rows(); ++i) {
        for (int j = 0; j < db_matrices[k].Cols(); ++j) {
          padded_data[i][j] =
              db_matrices[k].Data()[i * db_matrices[k].Cols() + j];
        }
      }
      ASSIGN_OR_RETURN(db_matrix_for_hints,
                       Matrix<DbDataType>::Create(std::move(padded_data)));
    } else {
      db_matrix_for_hints = db_matrices[k];
    }
    ASSIGN_OR_RETURN(auto hint_matrix,
                     (ComputeHintMatrix<CoeffType, DbDataType, CoeffType>(
                         db_matrix_for_hints, first_dim_a, *fft_ctx)));

    std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
        shard_outputs;
    shard_outputs.reserve(kInterpolationDegree);
    for (size_t i = 0; i < kInterpolationDegree; ++i) {
      std::vector<std::vector<CoeffType>> block;
      block.reserve(kDegree);
      for (size_t r = 0; r < kDegree; ++r) {
        auto start = hint_matrix.Data().begin() +
                     static_cast<size_t>(i * kDegree + r) * hint_matrix.Cols();
        block.emplace_back(start, start + kDegree);
      }
      ASSIGN_OR_RETURN(auto pack_output,
                       (PreprocessMatrixPack<CoeffType, MatCoeffType>(
                           params.RlweParameters(), block, w_vec_g, w_vec_h,
                           params.PackGadgetParams(), *fft_ctx)));
      shard_outputs.push_back(std::move(pack_output));
    }
    preprocessed_outputs.push_back(std::move(shard_outputs));
  }

  // SIMD padding & condensation
  for (size_t k = 0; k < kEntrySizeMultiple; ++k) {
    int cols = db_matrices[k].Cols();
    constexpr int simd_multiple = sizeof(DbDataType) == 1 ? 64 : 32;
    if (cols % simd_multiple != 0) {
      int padded_simd_cols =
          ((cols + simd_multiple - 1) / simd_multiple) * simd_multiple;
      std::vector<std::vector<DbDataType>> padded_data(
          db_matrices[k].Rows(),
          std::vector<DbDataType>(padded_simd_cols, 0));
      for (int i = 0; i < db_matrices[k].Rows(); ++i) {
        for (int j = 0; j < cols; ++j) {
          padded_data[i][j] = db_matrices[k].Data()[i * cols + j];
        }
      }
      ASSIGN_OR_RETURN(db_matrices[k],
                       Matrix<DbDataType>::Create(std::move(padded_data)));
    }
    auto condensed_matrix_or =
        Matrix<DbDataType>::CreateCondensed(db_matrices[k]);
    if (condensed_matrix_or.ok()) {
      db_matrices[k] = std::move(*condensed_matrix_or);
    }
  }

  PirPreprocessedData<DbDataType, CoeffType, MatCoeffType> preprocessed_data{
      std::move(db_matrices),
      std::move(preprocessed_outputs),
      std::move(second_dim_a),
  };

  //// Preprocessing is done ////

  // Keep preprocessed server material in memory (no disk storage).
  LOG(INFO) << "Creating PIR server (in-memory)...";
  ASSIGN_OR_RETURN(
      auto server,
      (PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
          params, std::move(preprocessed_data))));

  // Online Phase: Single query
  LOG(INFO) << "Creating PIR client...";
  ASSIGN_OR_RETURN(auto client_a_prng, CreatePrng(prng_seed));
  ASSIGN_OR_RETURN(auto client_prng, CreatePrng());
  ASSIGN_OR_RETURN(
      auto client,
      PirClient<CoeffType>::Create(params, std::move(client_a_prng),
                                   std::move(client_prng)));

  LOG(INFO) << "Generating PIR request for target index " << kTargetIndex
            << "...";
  ASSIGN_OR_RETURN(auto req, client->CreateRequest(kTargetIndex));

  LOG(INFO) << "Server processing response...";
  ASSIGN_OR_RETURN(auto resp, server->ProcessResponse(req));

  LOG(INFO) << "Client processing response...";
  ASSIGN_OR_RETURN(auto rec_msg, client->ProcessResponse(resp));

  // Correctness verification
  if (!kSkipVerification) {
    LOG(INFO) << "Verifying correctness...";
    const size_t offset =
        static_cast<size_t>(kTargetIndex) * kDegree * kEntrySizeMultiple;
    bool verification_success = true;
    size_t mismatches = 0;
    for (size_t i = 0; i < static_cast<size_t>(kDegree) * kEntrySizeMultiple;
         ++i) {
      if (rec_msg[i] != db_raw[offset + i]) {
        verification_success = false;
        mismatches++;
      }
    }
    if (!verification_success) {
      LOG(ERROR) << "Verification FAILED: " << mismatches << " mismatches!";
      return absl::InternalError("PIR verification failed.");
    }
    LOG(INFO) << "Verification SUCCESS: recovered entry matches database!";
  }

  LOG(INFO) << "PIR end-to-end execution completed successfully.";
  return absl::OkStatus();
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (auto status = private_membership::rlwe::v2::RunPirEndToEnd();
      !status.ok()) {
    LOG(FATAL) << "PIR end-to-end failed: " << status;
  }
  return 0;
}
