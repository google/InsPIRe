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

#ifndef CRYPTO_PIR_SERVER_PREPROCESSING_SERVER_H_
#define CRYPTO_PIR_SERVER_PREPROCESSING_SERVER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "crypto/db_row_interpolation.h"
#include "crypto/encryption.h"
#include "crypto/hint_computation.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_preprocessed_data.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "crypto/status_macros.h"
#include "shell_encryption/prng/chacha_prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// This class is used to perform all preprocessing operations on the database.
// This class takes in PIR parameters and a database, and outputs all necessary
// preprocessing material as one object that is then passed on to the PirServer.
template <typename DbValueType, typename CoeffType, typename MatCoeffType>
class PreprocessingServer {
 public:
  // Creates a new PreprocessingServer.
  static absl::StatusOr<std::unique_ptr<PreprocessingServer>> Create(
      const PirParams<CoeffType>* params) {
    ASSIGN_OR_RETURN(std::string prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
    return absl::WrapUnique(new PreprocessingServer(params, prng_seed));
  }

  static absl::StatusOr<std::unique_ptr<PreprocessingServer>> Create(
      const PirParams<CoeffType>* params, absl::string_view prng_seed) {
    return absl::WrapUnique(
        new PreprocessingServer(params, std::string(prng_seed)));
  }

  absl::StatusOr<PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>>
  Preprocess(const std::vector<DbValueType>& db_raw) const {
    ASSIGN_OR_RETURN(auto a_prng, ::rlwe::ChaChaPrng::Create(prng_seed_));
    const int d = params_->RlweParameters().Degree();
    const int t = params_->InterpolationDegree();
    const int n = params_->NumEntries();
    const int entry_size_multiple = params_->EntrySizeMultiple();

    if (db_raw.size() != static_cast<size_t>(n) * d * entry_size_multiple) {
      return absl::InvalidArgumentError("Invalid raw database size.");
    }

    std::vector<Matrix<DbValueType>> db_matrices;
    db_matrices.reserve(entry_size_multiple);

    for (size_t k = 0; k < entry_size_multiple; ++k) {
      std::vector<DbValueType> db_shard_raw(static_cast<size_t>(n) * d);
      for (size_t idx = 0; idx < n; ++idx) {
        for (size_t offset = 0; offset < d; ++offset) {
          db_shard_raw[idx * static_cast<size_t>(d) + offset] =
              db_raw[idx * static_cast<size_t>(d) * entry_size_multiple +
                     k * static_cast<size_t>(d) + offset];
        }
      }
      absl::Time start_interp = absl::Now();
      ASSIGN_OR_RETURN(auto db_matrix, InterpolateAndTranspose(db_shard_raw));
      interpolate_time_ += absl::Now() - start_interp;
      db_matrices.push_back(std::move(db_matrix));
    }

    const int first_dim_a_samples = params_->NumFirstDimSamples();
    ASSIGN_OR_RETURN(auto first_dim_a,
                     SampleAComponents(params_->RlweParameters(),
                                       first_dim_a_samples, *a_prng));
    const int second_dim_a_samples =
        t > 1 ? 2 * params_->PolyEvalGadgetParams().num_digits : 0;
    std::vector<Polynomial<CoeffType>> second_dim_a;
    if (second_dim_a_samples > 0) {
      ASSIGN_OR_RETURN(second_dim_a,
                       SampleAComponents(params_->RlweParameters(),
                                         second_dim_a_samples, *a_prng));
    }
    const int pack_a_samples = 2 * params_->PackGadgetParams().num_digits;
    ASSIGN_OR_RETURN(auto pack_a, SampleAComponents(params_->RlweParameters(),
                                                    pack_a_samples, *a_prng));

    std::vector<
        std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
        preprocessed_outputs;
    preprocessed_outputs.reserve(entry_size_multiple);
    for (size_t k = 0; k < entry_size_multiple; ++k) {
      absl::Time start_comp = absl::Now();
      ASSIGN_OR_RETURN(
          auto preprocessed_output,
          ComputePreprocessedOutputs(db_matrices[k], first_dim_a, pack_a));
      compute_outputs_time_ += absl::Now() - start_comp;
      preprocessed_outputs.push_back(std::move(preprocessed_output));
    }

    if constexpr ((std::is_same_v<DbValueType, uint16_t> ||
                   std::is_same_v<DbValueType, uint8_t>) &&
                  std::is_same_v<CoeffType, uint64_t>) {
      for (size_t k = 0; k < entry_size_multiple; ++k) {
        int cols = db_matrices[k].Cols();
        int simd_multiple = std::is_same_v<DbValueType, uint8_t> ? 64 : 32;
        if (cols % simd_multiple != 0) {
          int padded_simd_cols =
              ((cols + simd_multiple - 1) / simd_multiple) * simd_multiple;
          std::vector<std::vector<DbValueType>> padded_data(
              db_matrices[k].Rows(),
              std::vector<DbValueType>(padded_simd_cols, 0));
          for (int i = 0; i < db_matrices[k].Rows(); ++i) {
            for (int j = 0; j < cols; ++j) {
              padded_data[i][j] = db_matrices[k].Data()[i * cols + j];
            }
          }
          ASSIGN_OR_RETURN(db_matrices[k],
                           Matrix<DbValueType>::Create(std::move(padded_data)));
        }
        auto condensed_matrix_or =
            Matrix<DbValueType>::CreateCondensed(db_matrices[k]);
        if (condensed_matrix_or.ok()) {
          db_matrices[k] = std::move(*condensed_matrix_or);
        }
      }
    }

    return PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>{
        std::move(db_matrices),
        std::move(preprocessed_outputs),
        std::move(second_dim_a),
    };
  }

  // Returns PRNG seed.
  const std::string& GetPrngSeed() const { return prng_seed_; }

  absl::Duration GetInterpolateTime() const { return interpolate_time_; }
  absl::Duration GetComputeOutputsTime() const { return compute_outputs_time_; }
  absl::Duration GetComputeHintsTime() const { return compute_hints_time_; }
  absl::Duration GetPreprocessPackTime() const { return preprocess_pack_time_; }

 private:
  explicit PreprocessingServer(const PirParams<CoeffType>* params,
                               absl::string_view prng_seed)
      : params_(params), prng_seed_(prng_seed) {}

  // Stage 1: DB interpolation.
  absl::StatusOr<Matrix<DbValueType>> InterpolateAndTranspose(
      const std::vector<DbValueType>& db_raw) const {
    const int d = params_->RlweParameters().Degree();
    const int t = params_->InterpolationDegree();
    const int n = params_->NumEntries();
    const int plaintext_modulus = params_->PlaintextModulus();

    if (db_raw.size() != static_cast<size_t>(n) * d) {
      return absl::InvalidArgumentError("Invalid raw database size.");
    }

    std::vector<std::vector<DbValueType>> db_reshaped(
        n / t, std::vector<DbValueType>(static_cast<size_t>(t) * d));
    for (size_t i = 0; i < n / t; ++i) {
      auto start = db_raw.begin() + i * (static_cast<size_t>(t) * d);
      db_reshaped[i].assign(start, start + static_cast<size_t>(t) * d);
    }
    std::vector<std::vector<DbValueType>> db_interpolated(
        n / t, std::vector<DbValueType>(static_cast<size_t>(t) * d));
    for (size_t i = 0; i < n / t; ++i) {
      ASSIGN_OR_RETURN(
          db_interpolated[i],
          Interpolate(db_reshaped[i], d, t,
                      static_cast<DbValueType>(plaintext_modulus)));
    }
    std::vector<std::vector<DbValueType>> db_transposed(
        static_cast<size_t>(t) * d, std::vector<DbValueType>(n / t));
    for (size_t i = 0; i < static_cast<size_t>(t) * d; ++i) {
      for (size_t j = 0; j < n / t; ++j) {
        db_transposed[i][j] = db_interpolated[j][i];
      }
    }
    return Matrix<DbValueType>::Create(db_transposed);
  }

  // Stage 2: Compute preprocessed materials.
  absl::StatusOr<
      std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
  ComputePreprocessedOutputs(
      const Matrix<DbValueType>& db_matrix,
      const std::vector<Polynomial<CoeffType>>& first_dim_a,
      const std::vector<Polynomial<CoeffType>>& pack_a) const {
    const int d = params_->RlweParameters().Degree();
    const int t = params_->InterpolationDegree();

    ASSIGN_OR_RETURN(
        auto ctx,
        Context::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
    ASSIGN_OR_RETURN(
        auto fft_ctx,
        FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
    const int padded_cols = params_->PaddedCols();
    Matrix<DbValueType> db_matrix_for_hints;
    if (db_matrix.Cols() < padded_cols) {
      std::vector<std::vector<DbValueType>> padded_data(
          db_matrix.Rows(), std::vector<DbValueType>(padded_cols, 0));
      for (int i = 0; i < db_matrix.Rows(); ++i) {
        for (int j = 0; j < db_matrix.Cols(); ++j) {
          padded_data[i][j] = db_matrix.Data()[i * db_matrix.Cols() + j];
        }
      }
      ASSIGN_OR_RETURN(db_matrix_for_hints,
                       Matrix<DbValueType>::Create(std::move(padded_data)));
    } else {
      db_matrix_for_hints = db_matrix;
    }
    absl::Time start_hint = absl::Now();
    ASSIGN_OR_RETURN(auto hint_matrix,
                     (ComputeHintMatrix<CoeffType, DbValueType, CoeffType>(
                         db_matrix_for_hints, first_dim_a, *fft_ctx)));
    compute_hints_time_ += absl::Now() - start_hint;

    std::vector<Polynomial<CoeffType>> w_vec_g(
        pack_a.begin(),
        pack_a.begin() + params_->PackGadgetParams().num_digits);
    std::vector<Polynomial<CoeffType>> w_vec_h(
        pack_a.begin() + params_->PackGadgetParams().num_digits, pack_a.end());
    std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
        preprocessed_outputs;
    preprocessed_outputs.reserve(t);
    for (size_t i = 0; i < t; ++i) {
      std::vector<std::vector<CoeffType>> block;
      block.reserve(d);
      for (size_t r = 0; r < d; ++r) {
        auto start = hint_matrix.Data().begin() +
                     static_cast<size_t>(i * d + r) * hint_matrix.Cols();
        block.emplace_back(start, start + d);
      }
      absl::Time start_pack = absl::Now();
      ASSIGN_OR_RETURN(auto pack_output,
                       (PreprocessMatrixPack<CoeffType, MatCoeffType>(
                           params_->RlweParameters(), block, w_vec_g, w_vec_h,
                           params_->PackGadgetParams(), *fft_ctx)));
      preprocess_pack_time_ += absl::Now() - start_pack;
      preprocessed_outputs.push_back(std::move(pack_output));
    }
    return preprocessed_outputs;
  }

  const PirParams<CoeffType>* params_;
  std::string prng_seed_;
  mutable absl::Duration interpolate_time_ = absl::ZeroDuration();
  mutable absl::Duration compute_outputs_time_ = absl::ZeroDuration();
  mutable absl::Duration compute_hints_time_ = absl::ZeroDuration();
  mutable absl::Duration preprocess_pack_time_ = absl::ZeroDuration();
};

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_PIR_SERVER_PREPROCESSING_SERVER_H_
