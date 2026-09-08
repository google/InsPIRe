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

#ifndef CRYPTO_VPIR_SERVER_VPIR_PREPROCESSING_SERVER_H_
#define CRYPTO_VPIR_SERVER_VPIR_PREPROCESSING_SERVER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/hint_computation.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/server/preprocessing_server.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "crypto/vpir/server/vpir_preprocessed_data.h"
#include "crypto/vpir/vpir_params.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "crypto/status_macros.h"
#include "shell_encryption/prng/chacha_prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename CoeffType>
void PrintMatrix(absl::string_view name, const Matrix<CoeffType>& mat,
                 uint64_t mask = 0xFFFFFFFFFFFFFFFFULL) {
  LOG(ERROR) << "Matrix " << name << " (" << mat.Rows() << "x" << mat.Cols()
             << "):";
  for (int r = 0; r < mat.Rows(); ++r) {
    std::string row_str = "  [";
    for (int c = 0; c < mat.Cols(); ++c) {
      absl::StrAppend(&row_str, mat.Data()[r * mat.Cols() + c] & mask, ", ");
    }
    absl::StrAppend(&row_str, "]");
    LOG(ERROR) << row_str;
  }
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType,
          typename ZDbValueType = DbValueType, typename ZCoeffType = CoeffType,
          typename ZMatCoeffType = MatCoeffType>
class VpirPreprocessingServer {
 public:
  static absl::StatusOr<std::unique_ptr<VpirPreprocessingServer>> Create(
      const VpirParams<CoeffType, ZCoeffType>* params) {
    ASSIGN_OR_RETURN(std::string prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
    return absl::WrapUnique(new VpirPreprocessingServer(params, prng_seed));
  }

  absl::StatusOr<VpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                                      ZDbValueType, ZCoeffType, ZMatCoeffType>>
  Preprocess(const std::vector<DbValueType>& db_raw) const {
    ASSIGN_OR_RETURN(auto a_prng, ::rlwe::ChaChaPrng::Create(prng_seed_));
    const int d = params_->RlweParameters().Degree();
    const int n = params_->NumEntries();
    const int ell = params_->Ell();
    const int total_entry_coeffs = ell * d;

    if (db_raw.size() != static_cast<size_t>(n) * total_entry_coeffs) {
      return absl::InvalidArgumentError("Invalid raw database size.");
    }

    ASSIGN_OR_RETURN(auto db_matrix, TransposeDbMatrix(db_raw));

    const int first_dim_a_samples = n / d;
    ASSIGN_OR_RETURN(auto first_dim_a,
                     SampleAComponents(params_->RlweParameters(),
                                       first_dim_a_samples, *a_prng));
    const int pack_a_samples = 2 * params_->PackGadgetParams().num_digits;
    ASSIGN_OR_RETURN(auto pack_a, SampleAComponents(params_->RlweParameters(),
                                                    pack_a_samples, *a_prng));

    ASSIGN_OR_RETURN(
        auto fft_ctx,
        FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
    ASSIGN_OR_RETURN(auto hint_matrix,
                     (ComputeHintMatrix<CoeffType, DbValueType, CoeffType>(
                         db_matrix, first_dim_a, *fft_ctx)));

    std::vector<Polynomial<CoeffType>> w_vec_g(
        pack_a.begin(),
        pack_a.begin() + params_->PackGadgetParams().num_digits);
    std::vector<Polynomial<CoeffType>> w_vec_h(
        pack_a.begin() + params_->PackGadgetParams().num_digits, pack_a.end());

    std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
        preprocessed_outputs;
    preprocessed_outputs.reserve(ell);

    for (int k = 0; k < ell; ++k) {
      std::vector<std::vector<CoeffType>> block;
      block.reserve(d);
      for (size_t r = 0; r < static_cast<size_t>(d); ++r) {
        auto start =
            hint_matrix.Data().begin() + (k * d + r) * hint_matrix.Cols();
        block.emplace_back(start, start + d);
      }
      ASSIGN_OR_RETURN(auto pack_output,
                       (PreprocessMatrixPack<CoeffType, MatCoeffType>(
                           params_->RlweParameters(), block, w_vec_g, w_vec_h,
                           params_->PackGadgetParams(), *fft_ctx)));
      preprocessed_outputs.push_back(std::move(pack_output));
    }

    ASSIGN_OR_RETURN(
        auto z_prep_server,
        (PreprocessingServer<ZDbValueType, ZCoeffType, ZMatCoeffType>::Create(
            &params_->ZPirParams(), prng_seed_)));

    std::vector<ZDbValueType> z_db_raw;
    z_db_raw.reserve(db_matrix.Data().size());
    for (const auto& val : db_matrix.Data()) {
      z_db_raw.push_back(static_cast<ZDbValueType>(val));
    }
    ASSIGN_OR_RETURN(auto z_preprocessed_data,
                     z_prep_server->Preprocess(z_db_raw));

    if constexpr ((std::is_same_v<DbValueType, uint16_t> ||
                   std::is_same_v<DbValueType, uint8_t>) &&
                  (std::is_same_v<CoeffType, uint64_t> ||
                   std::is_same_v<CoeffType, uint32_t>)) {
      int cols = db_matrix.Cols();
      int simd_multiple = std::is_same_v<DbValueType, uint8_t> ? 64 : 32;
      if (cols % simd_multiple == 0) {
        auto condensed_matrix_or =
            Matrix<DbValueType>::CreateCondensed(db_matrix);
        if (condensed_matrix_or.ok()) {
          db_matrix = std::move(*condensed_matrix_or);
        }
      }
    }

    return VpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                                ZDbValueType, ZCoeffType, ZMatCoeffType>{
        .db_matrix = std::move(db_matrix),
        .preprocessed_outputs = std::move(preprocessed_outputs),
        .hint_matrix = std::move(hint_matrix),
        .prng_seed = prng_seed_,
        .z_preprocessed_data = std::move(z_preprocessed_data),
    };
  }

  const std::string& GetPrngSeed() const { return prng_seed_; }

 private:
  struct ShardPreprocessResult {
    PreprocessMatrixPackOutput<CoeffType, MatCoeffType> output;
    Matrix<CoeffType> hint_matrix;
  };

  explicit VpirPreprocessingServer(
      const VpirParams<CoeffType, ZCoeffType>* params,
      absl::string_view prng_seed)
      : params_(params), prng_seed_(prng_seed) {}

  absl::StatusOr<Matrix<DbValueType>> TransposeDbMatrix(
      const std::vector<DbValueType>& db_raw) const {
    const int d = params_->RlweParameters().Degree();
    const int n = params_->NumEntries();
    const int ell = params_->Ell();
    const size_t rows = static_cast<size_t>(ell * d);

    if (db_raw.size() != static_cast<size_t>(n) * rows) {
      return absl::InvalidArgumentError("Invalid raw database size.");
    }

    std::vector<std::vector<DbValueType>> db_transposed(
        rows, std::vector<DbValueType>(n));
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < rows; ++j) {
        db_transposed[j][i] = db_raw[i * rows + j];
      }
    }

    return Matrix<DbValueType>::Create(db_transposed);
  }

  absl::StatusOr<ShardPreprocessResult> ComputePreprocessedOutputs(
      const Matrix<DbValueType>& db_matrix,
      const std::vector<Polynomial<CoeffType>>& first_dim_a,
      const std::vector<Polynomial<CoeffType>>& pack_a) const {
    const int d = params_->RlweParameters().Degree();
    const int n = params_->NumEntries();
    const int num_digits = params_->PackGadgetParams().num_digits;

    if (first_dim_a.size() != n / d) {
      return absl::InvalidArgumentError("Invalid first_dim_a size.");
    }
    if (pack_a.size() != 2 * num_digits) {
      return absl::InvalidArgumentError("Invalid pack_a size.");
    }

    ASSIGN_OR_RETURN(auto a_mat_server, ExpandAMatrix<CoeffType>(first_dim_a));

    ASSIGN_OR_RETURN(
        auto fft_ctx,
        FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
    ASSIGN_OR_RETURN(auto hint_matrix,
                     (ComputeHintMatrix<CoeffType, DbValueType, CoeffType>(
                         db_matrix, first_dim_a, *fft_ctx)));

    std::vector<Polynomial<CoeffType>> w_vec_g(
        pack_a.begin(),
        pack_a.begin() + params_->PackGadgetParams().num_digits);
    std::vector<Polynomial<CoeffType>> w_vec_h(
        pack_a.begin() + params_->PackGadgetParams().num_digits, pack_a.end());
    std::vector<std::vector<CoeffType>> block;
    block.reserve(d);
    for (size_t r = 0; r < d; ++r) {
      auto start = hint_matrix.Data().begin() + r * hint_matrix.Cols();
      block.emplace_back(start, start + d);
    }
    ASSIGN_OR_RETURN(auto pack_output,
                     (PreprocessMatrixPack<CoeffType, MatCoeffType>(
                         params_->RlweParameters(), block, w_vec_g, w_vec_h,
                         params_->PackGadgetParams(), *fft_ctx)));
    return ShardPreprocessResult{std::move(pack_output),
                                 std::move(hint_matrix)};
  }

  const VpirParams<CoeffType, ZCoeffType>* params_;
  const std::string prng_seed_;
};

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_VPIR_SERVER_VPIR_PREPROCESSING_SERVER_H_
