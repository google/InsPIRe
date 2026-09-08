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

#ifndef CRYPTO_VPIR_CLIENT_VPIR_CLIENT_H_
#define CRYPTO_VPIR_CLIENT_VPIR_CLIENT_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/polynomial.h"
#include "crypto/vpir/vpir_params.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "shell_encryption/prng/prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename DbDataType, typename CoeffType, typename MatCoeffType,
          typename ZDbDataType, typename ZCoeffType, typename ZMatCoeffType>
class VpirServer;

template <typename CoeffType, typename ZCoeffType = CoeffType>
class VpirClient {
 public:
  static absl::StatusOr<std::unique_ptr<VpirClient<CoeffType, ZCoeffType>>>
  Create(const VpirParams<CoeffType, ZCoeffType>& params,
         std::unique_ptr<::rlwe::SecurePrng> a_component_prng,
         std::unique_ptr<::rlwe::SecurePrng> prng);

  absl::StatusOr<VpirRequest<CoeffType>> CreateRequest(int index);

  const Polynomial<CoeffType>& GetSecretKey() const { return secret_key_; }
  const VpirParams<CoeffType, ZCoeffType>& GetParams() const {
    return params_;
  }

  ::rlwe::SecurePrng* GetAComponentPrng() const {
    return a_component_prng_.get();
  }

  ::rlwe::SecurePrng* GetPrng() const { return prng_.get(); }

  absl::StatusOr<std::vector<CoeffType>> ProcessResponse(
      const VpirResponse<CoeffType>& response,
      const VpirRequest<CoeffType>& request, int index) const;

  absl::Status VerifyResponse(const VpirResponse<CoeffType>& response,
                              const VpirRequest<CoeffType>& request) const;

  absl::StatusOr<std::vector<CoeffType>> DecryptResponse(
      const VpirResponse<CoeffType>& response) const;

  absl::StatusOr<std::vector<Polynomial<CoeffType>>> CreatePackingKey() const;

  template <typename DbDataType, typename MatCoeffType,
            typename ZDbDataType = DbDataType,
            typename ZMatCoeffType = MatCoeffType>
  absl::Status GetProofPreprocessMaterial(
      int kappa,
      const VpirServer<DbDataType, CoeffType, MatCoeffType, ZDbDataType,
                       ZCoeffType, ZMatCoeffType>& server);

  const std::vector<PreprocessMatrixPackOutput<CoeffType>>&
  GetPreprocessedOutputs() const {
    return preprocessed_outputs_;
  }

  void SetPreprocessedOutputs(
      std::vector<PreprocessMatrixPackOutput<CoeffType>> preprocessed_outputs) {
    preprocessed_outputs_ = std::move(preprocessed_outputs);
  }

  const Matrix<CoeffType>& GetProofC() const { return c_matrix_; }

  const Matrix<CoeffType>& GetProofZ() const { return z_matrix_; }

  const Matrix<CoeffType>& GetHPrimeMatrix() const { return h_prime_matrix_; }

  const Matrix<CoeffType>& GetZPrimeMatrix() const { return z_prime_matrix_; }

 private:
  absl::StatusOr<std::vector<Polynomial<CoeffType>>> CreateFirstDimensionQuery(
      int index) const;

  template <typename MatCoeffType>
  absl::StatusOr<Matrix<CoeffType>> CreateHPrimeMatrix(
      const std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>&
          preprocessed_outputs) const;

  VpirClient(const VpirParams<CoeffType, ZCoeffType>& params,
             std::unique_ptr<::rlwe::SecurePrng> a_component_prng,
             std::unique_ptr<::rlwe::SecurePrng> prng,
             Polynomial<CoeffType> secret_key,
             Context ctx,
             NttPolynomial secret_key_ntt)
      : params_(params),
        a_component_prng_(std::move(a_component_prng)),
        prng_(std::move(prng)),
        secret_key_(std::move(secret_key)),
        ctx_(std::move(ctx)),
        secret_key_ntt_(std::move(secret_key_ntt)) {}

  const VpirParams<CoeffType, ZCoeffType> params_;
  const std::unique_ptr<::rlwe::SecurePrng> a_component_prng_;
  const std::unique_ptr<::rlwe::SecurePrng> prng_;
  const Polynomial<CoeffType> secret_key_;
  const Context ctx_;
  const NttPolynomial secret_key_ntt_;

  std::vector<PreprocessMatrixPackOutput<CoeffType>> preprocessed_outputs_;
  std::vector<Polynomial<CoeffType>> last_y_vec_h_;
  Matrix<CoeffType> c_matrix_;
  Matrix<CoeffType> z_matrix_;
  Matrix<CoeffType> h_prime_matrix_;
  Matrix<CoeffType> z_prime_matrix_;
};

extern template class VpirClient<uint32_t, uint32_t>;
extern template class VpirClient<uint64_t, uint64_t>;
extern template class VpirClient<uint32_t, uint64_t>;
extern template class VpirClient<uint64_t, uint32_t>;

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_VPIR_CLIENT_VPIR_CLIENT_H_
