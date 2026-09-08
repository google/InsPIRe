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

#ifndef CRYPTO_PIR_SERVER_PIR_SERVER_H_
#define CRYPTO_PIR_SERVER_PIR_SERVER_H_

#include <memory>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/external_product.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_preprocessed_data.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// PirServer represents the server side of a Private Information Retrieval (PIR)
// protocol.
template <typename DbDataType, typename CoeffType, typename MatCoeffType>
class PirServer {
 public:
  // Creates a PirServer instance.
  static absl::StatusOr<
      std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
  Create(const PirParams<CoeffType>& params,
         PirPreprocessedData<DbDataType, CoeffType, MatCoeffType>
             preprocessed_data);

  // Processes a PirRequest to produce a PirResponse.

  // Serializes the server's preprocessed data and database to a binary file on
  // disk.
  absl::Status SaveToFile(absl::string_view file_path) const;

  // Loads the server's preprocessed data and database from a byte buffer.
  static absl::StatusOr<
      std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
  LoadFromBuffer(const PirParams<CoeffType>& params, const uint8_t* buffer,
                 size_t size);

  static absl::StatusOr<
      std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
  LoadFromFile(const PirParams<CoeffType>& params, absl::string_view file_path);

  absl::StatusOr<PirResponse<CoeffType>> ProcessResponse(
      const PirRequest<CoeffType>& request) const;

  absl::Duration GetDbMultiplyTime() const { return db_mult_time_; }
  absl::Duration GetPackingTime() const { return packing_time_; }
  absl::Duration GetPolyEvalTime() const { return poly_eval_time_; }

 private:
  PirServer(
      const PirParams<CoeffType>& params, Matrix<DbDataType> combined_db_matrix,
      Matrix<MatCoeffType> combined_pack_matrix,
      std::vector<
          std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
          preprocessed_outputs,
      std::vector<Polynomial<CoeffType>> second_dim_query_a, Context ctx)
      : params_(params),
        combined_db_matrix_(std::move(combined_db_matrix)),
        combined_pack_matrix_(std::move(combined_pack_matrix)),
        preprocessed_outputs_(std::move(preprocessed_outputs)),
        second_dim_query_a_(std::move(second_dim_query_a)),
        ctx_(std::move(ctx)),
        db_mult_time_(absl::ZeroDuration()),
        packing_time_(absl::ZeroDuration()),
        poly_eval_time_(absl::ZeroDuration()) {}

  // Evaluates a polynomial where coefficients are given by `coeffs` using
  // Horner's method at the point x = `eval_point`.
  // Here, `coeffs` encrypts the coefficients c_0, c_1, ..., c_{t-1}.
  absl::StatusOr<RlweCiphertext<CoeffType>> EvalPoly(
      const std::vector<RlweCiphertext<CoeffType>>& coeffs,
      const RgswCiphertext<CoeffType>& eval_point, FftContext& ctx) const;

  const PirParams<CoeffType> params_;
  const Matrix<DbDataType> combined_db_matrix_;
  const Matrix<MatCoeffType> combined_pack_matrix_;
  const std::vector<
      std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
      preprocessed_outputs_;
  const std::vector<Polynomial<CoeffType>> second_dim_query_a_;
  Context ctx_;

  mutable absl::Duration db_mult_time_;
  mutable absl::Duration packing_time_;
  mutable absl::Duration poly_eval_time_;
};

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_PIR_SERVER_PIR_SERVER_H_
