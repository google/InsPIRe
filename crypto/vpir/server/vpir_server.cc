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

#include "crypto/vpir/server/vpir_server.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_server.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "crypto/vpir/server/vpir_preprocessed_data.h"
#include "crypto/vpir/vpir_params.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "crypto/status_macros.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename DbDataType, typename CoeffType, typename MatCoeffType,
          typename ZDbDataType, typename ZCoeffType, typename ZMatCoeffType>
absl::StatusOr<std::unique_ptr<VpirServer<
    DbDataType, CoeffType, MatCoeffType, ZDbDataType, ZCoeffType,
    ZMatCoeffType>>>
VpirServer<DbDataType, CoeffType, MatCoeffType, ZDbDataType, ZCoeffType,
           ZMatCoeffType>::
    Create(const VpirParams<CoeffType, ZCoeffType>& params,
           VpirPreprocessedData<DbDataType, CoeffType, MatCoeffType,
                                ZDbDataType, ZCoeffType, ZMatCoeffType>
               preprocessed_data) {
  const int d = params.RlweParameters().Degree();
  const int n = params.NumEntries();
  const int ell = params.Ell();

  if (preprocessed_data.db_matrix.Rows() != ell * d) {
    return absl::InvalidArgumentError("DB matrix must have ell * d rows.");
  }
  if (preprocessed_data.db_matrix.Cols() != n) {
    return absl::InvalidArgumentError("DB matrix must have N columns.");
  }

  if (preprocessed_data.preprocessed_outputs.size() !=
      static_cast<size_t>(ell)) {
    return absl::InvalidArgumentError("preprocessed_outputs size mismatch.");
  }

  ASSIGN_OR_RETURN(auto z_pir_server,
                   (PirServer<ZDbDataType, ZCoeffType, ZMatCoeffType>::Create(
                       params.ZPirParams(),
                       std::move(preprocessed_data.z_preprocessed_data))));

  ASSIGN_OR_RETURN(
      Context ctx,
      Context::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));

  Matrix<DbDataType> db_matrix = preprocessed_data.db_matrix;

  std::vector<Matrix<MatCoeffType>> pack_matrices;
  pack_matrices.reserve(ell);
  for (int k = 0; k < ell; ++k) {
    pack_matrices.push_back(preprocessed_data.preprocessed_outputs[k].matrix);
  }
  ASSIGN_OR_RETURN(auto combined_pack_matrix,
                   Matrix<MatCoeffType>::VerticalConcatenate(pack_matrices));

  return absl::WrapUnique(
      new VpirServer<DbDataType, CoeffType, MatCoeffType, ZDbDataType,
                     ZCoeffType, ZMatCoeffType>(
          params, std::move(db_matrix), std::move(combined_pack_matrix),
          std::move(preprocessed_data.preprocessed_outputs),
          std::move(preprocessed_data.hint_matrix),
          preprocessed_data.prng_seed,
          std::move(z_pir_server), std::move(ctx)));
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType,
          typename ZDbDataType, typename ZCoeffType, typename ZMatCoeffType>
absl::StatusOr<VpirResponse<CoeffType>>
VpirServer<DbDataType, CoeffType, MatCoeffType, ZDbDataType, ZCoeffType,
           ZMatCoeffType>::ProcessResponse(
    const VpirRequest<CoeffType>& request) const {
  const int d = params_.RlweParameters().Degree();
  const int ell = params_.Ell();

  const int num_digits = params_.PackGadgetParams().num_digits;
  const int expected_packing_keys =
      params_.LocalFinalize() ? num_digits : 2 * num_digits;
  if (request.packing_key.size() !=
      static_cast<size_t>(expected_packing_keys)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Packing key must contain ", expected_packing_keys,
                     " polynomials."));
  }
  auto pack_begin = request.packing_key.begin();
  std::vector<Polynomial<CoeffType>> y_vec_g(pack_begin,
                                             pack_begin + num_digits);

  ASSIGN_OR_RETURN(std::vector<CoeffType> flattened_first_dim,
                   FlattenPolynomials(request.first_dimension_query));
  if (flattened_first_dim.size() != db_matrix_.Cols()) {
    return absl::InvalidArgumentError(
        "Flattened first dimension query must match DB matrix columns (N).");
  }

  // Part 1: Matrix-vector multiplication between database matrix and query.
  ASSIGN_OR_RETURN(
      std::vector<CoeffType> b_prime_all,
      db_matrix_.template Multiply<CoeffType>(flattened_first_dim));

  if (b_prime_all.size() != static_cast<size_t>(ell * d)) {
    return absl::InternalError(
        "DB multiplication resulted in incorrect vector size.");
  }

  // Part 2: Matrix-vector multiplication between packing matrix and G part.
  ASSIGN_OR_RETURN(std::vector<CoeffType> y_vec_g_vec,
                   FlattenPolynomials(y_vec_g));

  ASSIGN_OR_RETURN(
      std::vector<CoeffType> b_agg_partial_all,
      combined_pack_matrix_.template Multiply<CoeffType>(y_vec_g_vec));

  if (b_agg_partial_all.size() != static_cast<size_t>(ell * d)) {
    return absl::InternalError(
        "Packing multiplication resulted in incorrect vector size.");
  }

  // Part 3: Complete packing.
  std::vector<Polynomial<CoeffType>> b_responses;
  b_responses.reserve(ell);

  if (params_.LocalFinalize()) {
    for (int k = 0; k < ell; ++k) {
      const size_t offset = k * d;
      std::vector<CoeffType> b_chunk(d);
      for (int m = 0; m < d; ++m) {
        b_chunk[m] = b_prime_all[offset + m] + b_agg_partial_all[offset + m];
      }
      ASSIGN_OR_RETURN(Polynomial<CoeffType> b_agg,
                       Polynomial<CoeffType>::Create(std::move(b_chunk)));
      ASSIGN_OR_RETURN(b_agg,
                       b_agg.LowBits(params_.RlweParameters().LogModulus()));
      b_responses.push_back(std::move(b_agg));
    }
  } else {
    ASSIGN_OR_RETURN(
        auto fft_ctx,
        FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
    std::vector<Polynomial<CoeffType>> y_vec_h(pack_begin + num_digits,
                                               pack_begin + 2 * num_digits);
    for (int k = 0; k < ell; ++k) {
      const size_t offset = k * d;
      std::vector<CoeffType> chunk(b_prime_all.begin() + offset,
                                   b_prime_all.begin() + offset + d);
      std::vector<CoeffType> b_agg_partial(
          b_agg_partial_all.begin() + offset,
          b_agg_partial_all.begin() + offset + d);

      ASSIGN_OR_RETURN(
          RlweCiphertext<CoeffType> pack_result,
          FinalizeMatrixPack(params_.RlweParameters(), chunk, b_agg_partial,
                             y_vec_h, preprocessed_outputs_[k].t_vec_h,
                             preprocessed_outputs_[k].a_tilde_agg,
                             params_.PackGadgetParams(), *fft_ctx));

      b_responses.push_back(std::move(pack_result.b));
    }
  }

  return VpirResponse<CoeffType>{std::move(b_responses)};
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType,
          typename ZDbDataType, typename ZCoeffType, typename ZMatCoeffType>
absl::StatusOr<VpirResponse<CoeffType>>
VpirServer<DbDataType, CoeffType, MatCoeffType, ZDbDataType, ZCoeffType,
           ZMatCoeffType>::ProcessResponseWithMatMul(
    const VpirRequest<CoeffType>& request) const {
  const int d = params_.RlweParameters().Degree();
  const int ell = params_.Ell();

  const int num_digits = params_.PackGadgetParams().num_digits;
  const int expected_packing_keys =
      params_.LocalFinalize() ? num_digits : 2 * num_digits;
  if (request.packing_key.size() <
      static_cast<size_t>(expected_packing_keys)) {
    return absl::InvalidArgumentError("Invalid packing key in request.");
  }
  auto pack_begin = request.packing_key.begin();
  std::vector<Polynomial<CoeffType>> y_vec_g(pack_begin,
                                             pack_begin + num_digits);

  std::vector<CoeffType> flattened_first_dim;
  flattened_first_dim.reserve(request.first_dimension_query.size() * d);
  for (const auto& poly : request.first_dimension_query) {
    const auto& coeffs = poly.Coeffs();
    if (coeffs.size() != static_cast<size_t>(d)) {
      return absl::InvalidArgumentError(
          "Each polynomial in first_dimension_query must have degree d.");
    }
    flattened_first_dim.insert(flattened_first_dim.end(), coeffs.begin(),
                               coeffs.end());
  }

  if (flattened_first_dim.size() != static_cast<size_t>(db_matrix_.Cols())) {
    return absl::InvalidArgumentError(
        "Flattened query size must match combined DB matrix columns.");
  }

  ASSIGN_OR_RETURN(
      std::vector<CoeffType> b_prime_all,
      db_matrix_.template Multiply<CoeffType>(flattened_first_dim));

  ASSIGN_OR_RETURN(std::vector<CoeffType> y_vec_g_vec,
                   FlattenPolynomials(y_vec_g));

  ASSIGN_OR_RETURN(
      std::vector<CoeffType> b_agg_partial_all,
      combined_pack_matrix_.template Multiply<CoeffType>(y_vec_g_vec));

  std::vector<CoeffType> h_mult_result;
  if (!params_.LocalFinalize()) {
    std::vector<Polynomial<CoeffType>> y_vec_h(pack_begin + num_digits,
                                               pack_begin + 2 * num_digits);
    ASSIGN_OR_RETURN(std::vector<CoeffType> y_vec_h_vec,
                     FlattenPolynomials(y_vec_h));
    ASSIGN_OR_RETURN(auto h_prime_matrix, CreateHPrimeMatrix());
    ASSIGN_OR_RETURN(h_mult_result,
                     h_prime_matrix.template Multiply<CoeffType>(y_vec_h_vec));
  }

  std::vector<Polynomial<CoeffType>> b_responses;
  b_responses.reserve(ell);

  for (int k = 0; k < ell; ++k) {
    const size_t offset = k * d;
    std::vector<CoeffType> b_chunk(d);
    for (int m = 0; m < d; ++m) {
      b_chunk[m] = b_prime_all[offset + m] + b_agg_partial_all[offset + m];
      if (!params_.LocalFinalize()) {
        b_chunk[m] += h_mult_result[offset + m];
      }
    }
    ASSIGN_OR_RETURN(Polynomial<CoeffType> b_agg,
                     Polynomial<CoeffType>::Create(std::move(b_chunk)));
    ASSIGN_OR_RETURN(b_agg,
                     b_agg.LowBits(params_.RlweParameters().LogModulus()));
    b_responses.push_back(std::move(b_agg));
  }

  return VpirResponse<CoeffType>{std::move(b_responses)};
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType,
          typename ZDbDataType, typename ZCoeffType, typename ZMatCoeffType>
absl::StatusOr<
    VpirProofPreprocessMaterial<CoeffType, MatCoeffType, ZCoeffType>>
VpirServer<DbDataType, CoeffType, MatCoeffType, ZDbDataType, ZCoeffType,
           ZMatCoeffType>::
    GetProofPreprocessMaterial(
        const std::vector<PirRequest<ZCoeffType>>& z_requests)
        const {
  std::vector<PirResponse<ZCoeffType>> z_responses;
  z_responses.reserve(z_requests.size());
  for (const auto& req : z_requests) {
    ASSIGN_OR_RETURN(auto resp, z_pir_server_->ProcessResponse(req));
    z_responses.push_back(std::move(resp));
  }
  return VpirProofPreprocessMaterial<CoeffType, MatCoeffType, ZCoeffType>{
      .preprocessed_outputs = preprocessed_outputs_,
      .hint_matrix = hint_matrix_,
      .z_responses = std::move(z_responses),
  };
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType,
          typename ZDbDataType, typename ZCoeffType, typename ZMatCoeffType>
absl::StatusOr<std::vector<CoeffType>>
VpirServer<DbDataType, CoeffType, MatCoeffType, ZDbDataType, ZCoeffType,
           ZMatCoeffType>::FlattenPolynomials(
    absl::Span<const Polynomial<CoeffType>> polys) const {
  const int d = params_.RlweParameters().Degree();
  std::vector<CoeffType> flattened;
  flattened.reserve(polys.size() * d);
  for (const auto& poly : polys) {
    const auto& coeffs = poly.Coeffs();
    if (coeffs.size() != static_cast<size_t>(d)) {
      return absl::InvalidArgumentError(
          "Each polynomial must have degree d when flattening.");
    }
    flattened.insert(flattened.end(), coeffs.begin(), coeffs.end());
  }
  return flattened;
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType,
          typename ZDbDataType, typename ZCoeffType, typename ZMatCoeffType>
absl::StatusOr<Matrix<MatCoeffType>>
VpirServer<DbDataType, CoeffType, MatCoeffType, ZDbDataType, ZCoeffType,
           ZMatCoeffType>::CreateHPrimeMatrix() const {
  const int ell = params_.Ell();
  const int d = params_.RlweParameters().Degree();
  const int num_digits = params_.PackGadgetParams().num_digits;
  const int log_digit = params_.PackGadgetParams().log_digit;
  const int total_rows = ell * d;
  const int total_cols_h = num_digits * d;
  const int shift = 64 - log_digit;

  std::vector<std::vector<MatCoeffType>> h_matrix_data(
      total_rows, std::vector<MatCoeffType>(total_cols_h));

  for (int k = 0; k < ell; ++k) {
    const auto& t_vec_h = preprocessed_outputs_[k].t_vec_h;
    if (t_vec_h.size() < static_cast<size_t>(num_digits)) {
      return absl::InternalError(
          "t_vec_h size too small in CreateHPrimeMatrix.");
    }
    for (int r = 0; r < d; ++r) {
      for (int col_h = 0; col_h < total_cols_h; ++col_h) {
        int j = col_h / d;
        int c = col_h % d;
        int m;
        bool negate = false;
        if (r >= c) {
          m = r - c;
        } else {
          m = r + d - c;
          negate = true;
        }
        if (j >= t_vec_h.size() || m >= t_vec_h[j].Coeffs().size()) {
          return absl::InternalError("out of bounds in t_vec_h.");
        }
        int64_t signed_val =
            static_cast<int64_t>(static_cast<uint64_t>(t_vec_h[j].Coeffs()[m])
                                 << shift) >>
            shift;
        if (negate) {
          signed_val = -signed_val;
        }
        h_matrix_data[k * d + r][col_h] = static_cast<MatCoeffType>(signed_val);
      }
    }
  }

  if constexpr (std::is_same_v<MatCoeffType, uint8_t> ||
                std::is_same_v<MatCoeffType, uint16_t> ||
                std::is_same_v<MatCoeffType, int32_t>) {
    return Matrix<MatCoeffType>::CreateCondensed(h_matrix_data);
  } else {
    return Matrix<MatCoeffType>::Create(h_matrix_data);
  }
}

template class VpirServer<uint16_t, uint32_t, uint32_t, uint16_t, uint32_t,
                          uint32_t>;
template class VpirServer<uint16_t, uint64_t, uint64_t, uint16_t, uint64_t,
                          uint64_t>;
template class VpirServer<uint16_t, uint32_t, int16_t, uint16_t, uint32_t,
                          int16_t>;
template class VpirServer<uint16_t, uint64_t, int16_t, uint16_t, uint64_t,
                          int16_t>;
template class VpirServer<uint16_t, uint32_t, int32_t, uint16_t, uint32_t,
                          int32_t>;
template class VpirServer<uint16_t, uint64_t, int32_t, uint16_t, uint64_t,
                          int32_t>;

template class VpirServer<uint8_t, uint32_t, uint32_t, uint8_t, uint32_t,
                          uint32_t>;
template class VpirServer<uint8_t, uint64_t, uint64_t, uint8_t, uint64_t,
                          uint64_t>;
template class VpirServer<uint8_t, uint32_t, int16_t, uint8_t, uint32_t,
                          int16_t>;
template class VpirServer<uint8_t, uint64_t, int16_t, uint8_t, uint64_t,
                          int16_t>;
template class VpirServer<uint8_t, uint32_t, int32_t, uint8_t, uint32_t,
                          int32_t>;
template class VpirServer<uint8_t, uint64_t, int32_t, uint8_t, uint64_t,
                          int32_t>;

template class VpirServer<uint8_t, uint32_t, int32_t, uint8_t, uint64_t,
                          int32_t>;
template class VpirServer<uint8_t, uint32_t, int16_t, uint8_t, uint64_t,
                          int32_t>;
template class VpirServer<uint16_t, uint32_t, int32_t, uint16_t, uint64_t,
                          int32_t>;
template class VpirServer<uint16_t, uint32_t, int16_t, uint16_t, uint64_t,
                          int32_t>;

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
