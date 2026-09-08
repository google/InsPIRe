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

#ifndef CRYPTO_VPIR_SERVER_VPIR_SERVER_H_
#define CRYPTO_VPIR_SERVER_VPIR_SERVER_H_

#include <memory>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/external_product.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_server.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "crypto/vpir/server/vpir_preprocessed_data.h"
#include "crypto/vpir/vpir_params.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename CoeffType, typename MatCoeffType,
          typename ZCoeffType = CoeffType>
struct VpirProofPreprocessMaterial {
  std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
      preprocessed_outputs;
  Matrix<CoeffType> hint_matrix;
  std::vector<PirResponse<ZCoeffType>> z_responses;
};

template <typename DbDataType, typename CoeffType, typename MatCoeffType,
          typename ZDbDataType = DbDataType, typename ZCoeffType = CoeffType,
          typename ZMatCoeffType = MatCoeffType>
class VpirServer {
 public:
  static absl::StatusOr<std::unique_ptr<VpirServer<
      DbDataType, CoeffType, MatCoeffType, ZDbDataType, ZCoeffType,
      ZMatCoeffType>>>
  Create(const VpirParams<CoeffType, ZCoeffType>& params,
         VpirPreprocessedData<DbDataType, CoeffType, MatCoeffType, ZDbDataType,
                              ZCoeffType, ZMatCoeffType>
             preprocessed_data);

  absl::StatusOr<VpirResponse<CoeffType>> ProcessResponse(
      const VpirRequest<CoeffType>& request) const;

  absl::StatusOr<VpirResponse<CoeffType>> ProcessResponseWithMatMul(
      const VpirRequest<CoeffType>& request) const;

  absl::StatusOr<
      VpirProofPreprocessMaterial<CoeffType, MatCoeffType, ZCoeffType>>
  GetProofPreprocessMaterial(
      const std::vector<PirRequest<ZCoeffType>>& z_requests) const;

  const std::string& GetPrngSeed() const { return prng_seed_; }

  const Matrix<DbDataType>& GetDbMatrix() const { return db_matrix_; }

  const std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>&
  GetPreprocessedOutputs() const {
    return preprocessed_outputs_;
  }

 private:
  absl::StatusOr<std::vector<CoeffType>> FlattenPolynomials(
      absl::Span<const Polynomial<CoeffType>> polys) const;

  absl::StatusOr<Matrix<MatCoeffType>> CreateHPrimeMatrix() const;

  VpirServer(
      const VpirParams<CoeffType, ZCoeffType>& params,
      Matrix<DbDataType> db_matrix, Matrix<MatCoeffType> combined_pack_matrix,
      std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
          preprocessed_outputs,
      Matrix<CoeffType> hint_matrix, std::string prng_seed,
      std::unique_ptr<PirServer<ZDbDataType, ZCoeffType, ZMatCoeffType>>
          z_pir_server,
      Context ctx)
      : params_(params),
        db_matrix_(std::move(db_matrix)),
        combined_pack_matrix_(std::move(combined_pack_matrix)),
        preprocessed_outputs_(std::move(preprocessed_outputs)),
        hint_matrix_(std::move(hint_matrix)),
        prng_seed_(std::move(prng_seed)),
        z_pir_server_(std::move(z_pir_server)),
        ctx_(std::move(ctx)) {}

  const VpirParams<CoeffType, ZCoeffType> params_;
  const Matrix<DbDataType> db_matrix_;
  const Matrix<MatCoeffType> combined_pack_matrix_;
  const std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
      preprocessed_outputs_;
  const Matrix<CoeffType> hint_matrix_;
  const std::string prng_seed_;
  const std::unique_ptr<PirServer<ZDbDataType, ZCoeffType, ZMatCoeffType>>
      z_pir_server_;
  const Context ctx_;
};

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_VPIR_SERVER_VPIR_SERVER_H_
