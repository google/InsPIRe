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

#include "crypto/vpir/client/vpir_client.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/pir_params.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "crypto/vpir/server/vpir_server.h"
#include "crypto/vpir/vpir_params.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "crypto/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {
template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> InnerProduct(
    const std::vector<Polynomial<CoeffType>>& a,
    const std::vector<Polynomial<CoeffType>>& b, FftContext& ctx,
    const int a_bits = 8 * sizeof(CoeffType),
    const int b_bits = 8 * sizeof(CoeffType), bool a_is_signed = false) {
  if (a.size() != b.size() || a.empty()) {
    return absl::InvalidArgumentError("InnerProduct: size mismatch or empty");
  }
  int max_allowed_chunk = static_cast<int>(
      (53.0 - std::log2(a.size()) - std::log2(a[0].Len())) / 2.0);
  int chunk_bits = std::min(20, max_allowed_chunk);
  return Polynomial<CoeffType>::InnerProductFft(a, b, ctx, a_bits, b_bits,
                                                chunk_bits, a_is_signed);
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> MatrixMultiply(const Matrix<CoeffType>& A,
                                                 const Matrix<CoeffType>& B) {
  if (A.Cols() != B.Rows()) return absl::InvalidArgumentError("Dim mismatch");
  std::vector<std::vector<CoeffType>> out(A.Rows(),
                                          std::vector<CoeffType>(B.Cols(), 0));
  for (int i = 0; i < A.Rows(); ++i) {
    for (int k = 0; k < A.Cols(); ++k) {
      CoeffType val = A.Data()[i * A.Cols() + k];
      for (int j = 0; j < B.Cols(); ++j) {
        out[i][j] += val * B.Data()[k * B.Cols() + j];
      }
    }
  }
  return Matrix<CoeffType>::Create(std::move(out));
}

template <typename InType, typename OutType>
absl::StatusOr<Matrix<OutType>> MatrixCast(const Matrix<InType>& in) {
  std::vector<std::vector<OutType>> out(in.Rows(),
                                        std::vector<OutType>(in.Cols()));
  if (in.IsCondensed()) {
    int pack_size = sizeof(uint64_t) / sizeof(InType);
    int condensed_cols = in.Cols() / pack_size;
    for (int i = 0; i < in.Rows(); ++i) {
      for (int j = 0; j < condensed_cols; ++j) {
        uint64_t val = in.CondensedData()[i * condensed_cols + j];
        for (int k = 0; k < pack_size; ++k) {
          out[i][j * pack_size + k] = static_cast<OutType>(
              static_cast<InType>(val >> (k * 8 * sizeof(InType))));
        }
      }
    }
  } else {
    for (int i = 0; i < in.Rows(); ++i) {
      for (int j = 0; j < in.Cols(); ++j) {
        out[i][j] = static_cast<OutType>(in.Data()[i * in.Cols() + j]);
      }
    }
  }
  return Matrix<OutType>::Create(std::move(out));
}

template <typename OutputType, typename T>
absl::StatusOr<Matrix<OutputType>> LocalExpandAMatrix(
    const std::vector<Polynomial<T>>& a_components) {
  if (a_components.empty())
    return absl::InvalidArgumentError("Empty a_components");
  int n = a_components.size();
  int d = a_components[0].Len();
  std::vector<std::vector<OutputType>> out(n * d, std::vector<OutputType>(d));
  for (int k = 0; k < n; ++k) {
    const auto& coeffs = a_components[k].Coeffs();
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < d; ++j) {
        if (i >= j) {
          out[k * d + i][j] = static_cast<OutputType>(coeffs[i - j]);
        } else {
          out[k * d + i][j] = static_cast<OutputType>(-coeffs[d + i - j]);
        }
      }
    }
  }
  return Matrix<OutputType>::Create(std::move(out));
}

template <typename CoeffType>
absl::StatusOr<Matrix<CoeffType>> MatrixHorizontalConcatenate(
    const std::vector<Matrix<CoeffType>>& matrices) {
  if (matrices.empty()) return absl::InvalidArgumentError("Empty input");
  int rows = matrices[0].Rows();
  int cols = 0;
  for (const auto& m : matrices) {
    if (m.IsCondensed()) {
      return absl::InvalidArgumentError(
          "Condensed matrix not supported in concat");
    }
    if (m.Rows() != rows) return absl::InvalidArgumentError("Row mismatch");
    cols += m.Cols();
  }
  std::vector<std::vector<CoeffType>> out(rows, std::vector<CoeffType>(cols));
  for (int i = 0; i < rows; ++i) {
    int c = 0;
    for (const auto& m : matrices) {
      if (m.IsCondensed()) {
        return absl::InvalidArgumentError(
            "Condensed matrix not supported in concat");
      }
      for (int j = 0; j < m.Cols(); ++j) {
        out[i][c++] = m.Data()[i * m.Cols() + j];
      }
    }
  }
  return Matrix<CoeffType>::Create(std::move(out));
}
}  // namespace

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

template <typename CoeffType, typename ZCoeffType>
absl::StatusOr<std::unique_ptr<VpirClient<CoeffType, ZCoeffType>>>
VpirClient<CoeffType, ZCoeffType>::Create(
    const VpirParams<CoeffType, ZCoeffType>& params,
    std::unique_ptr<::rlwe::SecurePrng> a_component_prng,
    std::unique_ptr<::rlwe::SecurePrng> prng) {
  if (a_component_prng == nullptr) {
    return absl::InvalidArgumentError("a_component_prng must not be null.");
  }
  if (prng == nullptr) {
    return absl::InvalidArgumentError("prng must not be null.");
  }
  ASSIGN_OR_RETURN(auto secret_key,
                   SampleSecretKey(params.RlweParameters(), *prng));
  const int d = params.RlweParameters().Degree();
  ASSIGN_OR_RETURN(
      auto ctx,
      Context::CreateForTernary(
          absl::bit_width(static_cast<uint32_t>(d)) - 1));
  ASSIGN_OR_RETURN(auto secret_key_ntt,
                   secret_key.ToNtt(ctx, /*is_ternary=*/true));
  return absl::WrapUnique(new VpirClient<CoeffType, ZCoeffType>(
      params, std::move(a_component_prng), std::move(prng),
      std::move(secret_key), std::move(ctx), std::move(secret_key_ntt)));
}

template <typename CoeffType, typename ZCoeffType>
absl::StatusOr<VpirRequest<CoeffType>>
VpirClient<CoeffType, ZCoeffType>::CreateRequest(const int index) {
  if (index < 0 || index >= params_.NumEntries()) {
    return absl::InvalidArgumentError("Index out of bounds.");
  }

  VpirRequest<CoeffType> request;
  ASSIGN_OR_RETURN(request.first_dimension_query,
                   CreateFirstDimensionQuery(index));
  ASSIGN_OR_RETURN(request.packing_key, CreatePackingKey());

  if (params_.LocalFinalize()) {
    const int num_digits = params_.PackGadgetParams().num_digits;
    if (request.packing_key.size() != static_cast<size_t>(2 * num_digits)) {
      return absl::InternalError("Packing key size mismatch.");
    }
    last_y_vec_h_.assign(
        std::make_move_iterator(request.packing_key.begin() + num_digits),
        std::make_move_iterator(request.packing_key.end()));
    request.packing_key.resize(num_digits);
  }

  return request;
}

template <typename CoeffType, typename ZCoeffType>
absl::StatusOr<std::vector<Polynomial<CoeffType>>>
VpirClient<CoeffType, ZCoeffType>::CreateFirstDimensionQuery(
    const int index) const {
  const auto& rlwe_params = params_.RlweParameters();
  const int d = rlwe_params.Degree();

  const int row_index = index;
  const int i = row_index / d;
  const int j = row_index % d;

  const int num_samples = params_.NumEntries() / d;
  ASSIGN_OR_RETURN(auto a_first, SampleAComponents(rlwe_params, num_samples,
                                                   *a_component_prng_));
  ASSIGN_OR_RETURN(
      auto rlwe_samples,
      GenerateRlweSamples(rlwe_params, secret_key_ntt_, a_first, *prng_,
                          ctx_));

  std::vector<Polynomial<CoeffType>> query;
  query.reserve(rlwe_samples.size());

  for (int idx = 0; idx < rlwe_samples.size(); ++idx) {
    std::vector<CoeffType> message(d, 0);
    if (idx == i) {
      message[j] = 1;
    }
    ASSIGN_OR_RETURN(
        auto ct, EncryptFromRlweSample(rlwe_params, params_.PlaintextModulus(),
                                       rlwe_samples[idx], message));
    query.push_back(std::move(ct.b));
  }

  return query;
}

template <typename CoeffType, typename ZCoeffType>
absl::StatusOr<std::vector<Polynomial<CoeffType>>>
VpirClient<CoeffType, ZCoeffType>::CreatePackingKey() const {
  const auto& rlwe_params = params_.RlweParameters();
  const int d = rlwe_params.Degree();

  const int num_samples = 2 * params_.PackGadgetParams().num_digits;
  ASSIGN_OR_RETURN(
      std::vector<Polynomial<CoeffType>> a_pack,
      SampleAComponents(rlwe_params, num_samples, *a_component_prng_));
  ASSIGN_OR_RETURN(
      std::vector<RlweSample<CoeffType>> rlwe_samples,
      GenerateRlweSamples(rlwe_params, secret_key_ntt_, a_pack, *prng_, ctx_));

  ASSIGN_OR_RETURN(Polynomial<CoeffType> sk_g, secret_key_.Automorph(5));
  ASSIGN_OR_RETURN(Polynomial<CoeffType> sk_h,
                   secret_key_.Automorph(2 * d - 1));

  const int half = rlwe_samples.size() / 2;
  std::vector<RlweSample<CoeffType>> pack_samples_g(
      rlwe_samples.begin(), rlwe_samples.begin() + half);
  std::vector<RlweSample<CoeffType>> pack_samples_h(rlwe_samples.begin() + half,
                                                    rlwe_samples.end());

  ASSIGN_OR_RETURN(std::vector<RlweCiphertext<CoeffType>> ksm_g,
                   EncryptGadgetCiphertextFromRlweSamples(
                       rlwe_params, params_.PackGadgetParams(),
                       pack_samples_g, sk_g.Coeffs()));
  ASSIGN_OR_RETURN(std::vector<RlweCiphertext<CoeffType>> ksm_h,
                   EncryptGadgetCiphertextFromRlweSamples(
                       rlwe_params, params_.PackGadgetParams(),
                       pack_samples_h, sk_h.Coeffs()));

  std::vector<Polynomial<CoeffType>> packing_key;
  packing_key.reserve(rlwe_samples.size());
  for (auto& ct : ksm_g) {
    packing_key.push_back(std::move(ct.b));
  }
  for (auto& ct : ksm_h) {
    packing_key.push_back(std::move(ct.b));
  }
  return packing_key;
}

template <typename CoeffType, typename ZCoeffType>
absl::Status VpirClient<CoeffType, ZCoeffType>::VerifyResponse(
    const VpirResponse<CoeffType>& response,
    const VpirRequest<CoeffType>& request) const {
  const int ell = params_.Ell();
  const int d = params_.RlweParameters().Degree();

  if (response.b_responses.size() != static_cast<size_t>(ell)) {
    return absl::InvalidArgumentError(
        "Response must contain exactly ell b_responses.");
  }

  if (c_matrix_.Rows() > 0) {
    const int num_digits = params_.PackGadgetParams().num_digits;
    const int expected_packing_keys =
        params_.LocalFinalize() ? num_digits : 2 * num_digits;
    if (request.packing_key.size() <
        static_cast<size_t>(expected_packing_keys)) {
      return absl::InvalidArgumentError(
          "Invalid packing key length in request.");
    }
    std::vector<CoeffType> u;
    u.reserve(request.first_dimension_query.size() * d +
              expected_packing_keys * d);
    for (const auto& poly : request.first_dimension_query) {
      u.insert(u.end(), poly.Coeffs().begin(), poly.Coeffs().end());
    }
    for (int i = 0; i < expected_packing_keys; ++i) {
      const auto& poly = request.packing_key[i];
      u.insert(u.end(), poly.Coeffs().begin(), poly.Coeffs().end());
    }

    const CoeffType q = params_.RlweParameters().Modulus();
    ASSIGN_OR_RETURN(auto z_concat, MatrixHorizontalConcatenate<CoeffType>(
                                        {z_matrix_, z_prime_matrix_}));
    ASSIGN_OR_RETURN(auto lhs, z_concat.template Multiply<CoeffType>(u));

    std::vector<CoeffType> x;
    x.reserve(ell * d);
    for (int k = 0; k < ell; ++k) {
      const auto& b_coeffs = response.b_responses[k].Coeffs();
      x.insert(x.end(), b_coeffs.begin(), b_coeffs.end());
    }

    ASSIGN_OR_RETURN(auto rhs, c_matrix_.template Multiply<CoeffType>(x));
    if (lhs.size() != rhs.size()) {
      return absl::InternalError("[Z Z'] u and C x dimensions do not match.");
    }
    for (size_t idx = 0; idx < lhs.size(); ++idx) {
      if ((lhs[idx] & (q - 1)) != (rhs[idx] & (q - 1))) {
        return absl::InternalError("[Z Z'] u != C x verification failed.");
      }
    }
  }

  return absl::OkStatus();
}

template <typename CoeffType, typename ZCoeffType>
absl::StatusOr<std::vector<CoeffType>>
VpirClient<CoeffType, ZCoeffType>::DecryptResponse(
    const VpirResponse<CoeffType>& response) const {
  const int ell = params_.Ell();
  const int d = params_.RlweParameters().Degree();

  if (response.b_responses.size() != static_cast<size_t>(ell)) {
    return absl::InvalidArgumentError(
        "Response must contain exactly ell b_responses.");
  }

  std::vector<CoeffType> decrypted_message;
  decrypted_message.reserve(ell * d);
  const CoeffType mod = params_.PlaintextModulus();

  if (preprocessed_outputs_.size() != static_cast<size_t>(ell)) {
    return absl::InvalidArgumentError(
        "Client missing preprocessed outputs for response decryption.");
  }

  std::unique_ptr<FftContext> fft_ctx;
  if (params_.LocalFinalize()) {
    const int num_digits = params_.PackGadgetParams().num_digits;
    if (last_y_vec_h_.size() < static_cast<size_t>(num_digits)) {
      return absl::InvalidArgumentError(
          "Client missing last_y_vec_h_ for local finalize.");
    }
    ASSIGN_OR_RETURN(
        fft_ctx,
        FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
  }

  for (int k = 0; k < ell; ++k) {
    Polynomial<CoeffType> b_k = response.b_responses[k];
    if (params_.LocalFinalize()) {
      if (preprocessed_outputs_[k].t_vec_h.empty()) {
        return absl::InvalidArgumentError(
            "Client missing t_vec_h for local finalize decryption.");
      }
      ASSIGN_OR_RETURN(
          Polynomial<CoeffType> inner_h,
          InnerProduct(preprocessed_outputs_[k].t_vec_h, last_y_vec_h_,
                       *fft_ctx, params_.PackGadgetParams().log_digit,
                       params_.RlweParameters().LogModulus(),
                       /*a_is_signed=*/true));
      ASSIGN_OR_RETURN(b_k, b_k.Add(inner_h));
      ASSIGN_OR_RETURN(
          b_k, b_k.LowBits(params_.RlweParameters().LogModulus()));
    }
    RlweCiphertext<CoeffType> ct{
        .a = preprocessed_outputs_[k].a_tilde_agg,
        .b = std::move(b_k),
    };
    ASSIGN_OR_RETURN(
        std::vector<CoeffType> shard_decrypted,
        Decrypt(params_.RlweParameters(), mod, ct, secret_key_ntt_, ctx_));
    decrypted_message.insert(decrypted_message.end(), shard_decrypted.begin(),
                             shard_decrypted.end());
  }
  return decrypted_message;
}

template <typename CoeffType, typename ZCoeffType>
absl::StatusOr<std::vector<CoeffType>>
VpirClient<CoeffType, ZCoeffType>::ProcessResponse(
    const VpirResponse<CoeffType>& response,
    const VpirRequest<CoeffType>& request, const int index) const {
  if (index < 0 || index >= params_.NumEntries()) {
    return absl::InvalidArgumentError("Index out of bounds.");
  }
  RETURN_IF_ERROR(VerifyResponse(response, request));
  return DecryptResponse(response);
}

template <typename CoeffType, typename ZCoeffType>
template <typename DbDataType, typename MatCoeffType, typename ZDbDataType,
          typename ZMatCoeffType>
absl::Status VpirClient<CoeffType, ZCoeffType>::GetProofPreprocessMaterial(
    int kappa,
    const VpirServer<DbDataType, CoeffType, MatCoeffType, ZDbDataType,
                     ZCoeffType, ZMatCoeffType>& server) {
  if (kappa <= 0) {
    return absl::InvalidArgumentError("kappa must be positive.");
  }
  const int ell = params_.Ell();
  const int d = params_.RlweParameters().Degree();
  const int total_entry_coeffs = ell * d;
  const CoeffType q = params_.RlweParameters().Modulus();

  std::vector<std::vector<CoeffType>> c_data(
      kappa, std::vector<CoeffType>(total_entry_coeffs));
  for (int i = 0; i < kappa; ++i) {
    for (int j = 0; j < total_entry_coeffs; ++j) {
      ASSIGN_OR_RETURN(auto r64, prng_->Rand64());
      c_data[i][j] = static_cast<CoeffType>(r64 & 1);
    }
  }
  ASSIGN_OR_RETURN(auto c_mat, Matrix<CoeffType>::Create(std::move(c_data)));

  std::vector<PirRequest<ZCoeffType>> z_requests;
  z_requests.reserve(kappa);
  std::vector<std::unique_ptr<PirClient<ZCoeffType>>> z_clients;
  z_clients.reserve(kappa);

  for (int i = 0; i < kappa; ++i) {
    std::vector<ZCoeffType> row_vec(total_entry_coeffs);
    for (int j = 0; j < total_entry_coeffs; ++j) {
      row_vec[j] = static_cast<ZCoeffType>(
          c_mat.Data()[i * total_entry_coeffs + j]);
    }
    ASSIGN_OR_RETURN(auto z_client_a_prng,
                     ::rlwe::ChaChaPrng::Create(server.GetPrngSeed()));
    std::string prng_seed_z(::rlwe::ChaChaPrng::SeedLength(), 0);
    for (size_t idx = 0; idx < prng_seed_z.size(); ++idx) {
      ASSIGN_OR_RETURN(auto r8, prng_->Rand8());
      prng_seed_z[idx] = static_cast<char>(r8);
    }
    ASSIGN_OR_RETURN(auto z_client_prng,
                     ::rlwe::ChaChaPrng::Create(prng_seed_z));
    ASSIGN_OR_RETURN(auto z_pir_client,
                     PirClient<ZCoeffType>::Create(params_.ZPirParams(),
                                                   std::move(z_client_a_prng),
                                                   std::move(z_client_prng)));
    ASSIGN_OR_RETURN(auto req, z_pir_client->CreateRequestForVector(row_vec));
    z_requests.push_back(std::move(req));
    z_clients.push_back(std::move(z_pir_client));
  }

  ASSIGN_OR_RETURN(auto material,
                   server.GetProofPreprocessMaterial(z_requests));

  preprocessed_outputs_.clear();
  preprocessed_outputs_.reserve(material.preprocessed_outputs.size());
  for (const auto& output : material.preprocessed_outputs) {
    preprocessed_outputs_.push_back(PreprocessMatrixPackOutput<CoeffType>{
        output.a_tilde_agg, Matrix<CoeffType>(), output.t_vec_h});
  }

  const int n = params_.NumEntries();
  if (material.z_responses.size() != static_cast<size_t>(kappa)) {
    return absl::InternalError("z_responses size mismatch.");
  }
  std::vector<std::vector<CoeffType>> z_data(kappa,
                                             std::vector<CoeffType>(n, 0));

  for (int i = 0; i < kappa; ++i) {
    ASSIGN_OR_RETURN(
        std::vector<ZCoeffType> row_i,
        z_clients[i]->ProcessResponse(material.z_responses[i]));
    if (row_i.size() != static_cast<size_t>(n)) {
      return absl::InternalError("Decrypted Z row size mismatch.");
    }
    for (int r = 0; r < n; ++r) {
      z_data[i][r] = static_cast<CoeffType>(row_i[r]) & (q - 1);
    }
  }

  ASSIGN_OR_RETURN(auto z_mat, Matrix<CoeffType>::Create(std::move(z_data)));

  c_matrix_ = std::move(c_mat);
  z_matrix_ = std::move(z_mat);

  ASSIGN_OR_RETURN(auto a_prng,
                   ::rlwe::ChaChaPrng::Create(server.GetPrngSeed()));
  const int num_samples = params_.NumEntries() / d;
  ASSIGN_OR_RETURN(auto first_dim_a, SampleAComponents(params_.RlweParameters(),
                                                       num_samples, *a_prng));
  ASSIGN_OR_RETURN(auto a_mat, LocalExpandAMatrix<CoeffType>(first_dim_a));

  ASSIGN_OR_RETURN(auto lhs, MatrixMultiply(z_matrix_, a_mat));
  ASSIGN_OR_RETURN(auto rhs, MatrixMultiply(c_matrix_, material.hint_matrix));
  if (lhs.Data().size() != rhs.Data().size()) {
    return absl::InternalError("Z A and C H dimensions do not match.");
  }
  for (size_t idx = 0; idx < lhs.Data().size(); ++idx) {
    if ((lhs.Data()[idx] & (q - 1)) != (rhs.Data()[idx] & (q - 1))) {
      if (d <= 32) {
        PrintMatrix("Z", z_matrix_);
        PrintMatrix("A", a_mat);
        PrintMatrix("C", c_matrix_);
        PrintMatrix("H", material.hint_matrix, q - 1);
        PrintMatrix("LHS", lhs, q - 1);
        PrintMatrix("RHS", rhs, q - 1);
      }
      return absl::InternalError(
          absl::StrCat("Z A != C H verification failed at idx=", idx,
                       ": lhs=", lhs.Data()[idx] & (q - 1),
                       ", rhs=", rhs.Data()[idx] & (q - 1), ", q=", q));
    }
  }

  if (params_.LocalFinalize()) {
    std::vector<Matrix<CoeffType>> p_mats;
    p_mats.reserve(ell);
    for (int k = 0; k < ell; ++k) {
      ASSIGN_OR_RETURN(
          auto p_mat_k,
          (MatrixCast<MatCoeffType, CoeffType>(
              material.preprocessed_outputs[k].matrix)));
      p_mats.push_back(std::move(p_mat_k));
    }
    ASSIGN_OR_RETURN(auto p_mat_combined,
                     Matrix<CoeffType>::VerticalConcatenate(p_mats));
    ASSIGN_OR_RETURN(z_prime_matrix_,
                     MatrixMultiply(c_matrix_, p_mat_combined));
  } else {
    ASSIGN_OR_RETURN(h_prime_matrix_,
                     CreateHPrimeMatrix(material.preprocessed_outputs));

    ASSIGN_OR_RETURN(z_prime_matrix_,
                     MatrixMultiply(c_matrix_, h_prime_matrix_));

    // Clear intermediate memory that is no longer needed in the online phase
    h_prime_matrix_ = Matrix<CoeffType>();
  }

  return absl::OkStatus();
}

template <typename CoeffType, typename ZCoeffType>
template <typename MatCoeffType>
absl::StatusOr<Matrix<CoeffType>>
VpirClient<CoeffType, ZCoeffType>::CreateHPrimeMatrix(
    const std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>&
        preprocessed_outputs) const {
  const int ell = params_.Ell();
  const int d = params_.RlweParameters().Degree();
  const int num_digits = params_.PackGadgetParams().num_digits;
  const int log_digit = params_.PackGadgetParams().log_digit;
  const int total_rows = ell * d;
  const int total_cols_h = num_digits * d;
  const int shift = 64 - log_digit;

  std::vector<std::vector<CoeffType>> h_data(
      total_rows, std::vector<CoeffType>(total_cols_h));

  std::vector<Matrix<CoeffType>> p_mats;
  p_mats.reserve(ell);

  for (int k = 0; k < ell; ++k) {
    const auto& t_vec_h = preprocessed_outputs[k].t_vec_h;
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
        h_data[k * d + r][col_h] = static_cast<CoeffType>(signed_val);
      }
    }
    ASSIGN_OR_RETURN(
        auto p_mat_k,
        (MatrixCast<MatCoeffType, CoeffType>(preprocessed_outputs[k].matrix)));
    p_mats.push_back(std::move(p_mat_k));
  }

  ASSIGN_OR_RETURN(auto h_mat, Matrix<CoeffType>::Create(std::move(h_data)));
  ASSIGN_OR_RETURN(auto p_mat_combined,
                   Matrix<CoeffType>::VerticalConcatenate(p_mats));
  ASSIGN_OR_RETURN(auto combined, MatrixHorizontalConcatenate<CoeffType>(
                                      {p_mat_combined, h_mat}));

  return combined;
}

template class VpirClient<uint32_t, uint32_t>;
template class VpirClient<uint64_t, uint64_t>;
template class VpirClient<uint32_t, uint64_t>;
template class VpirClient<uint64_t, uint32_t>;

template absl::Status
VpirClient<uint32_t, uint32_t>::GetProofPreprocessMaterial<
    uint16_t, uint32_t, uint16_t, uint32_t>(
    int, const VpirServer<uint16_t, uint32_t, uint32_t, uint16_t, uint32_t,
                          uint32_t>&);
template absl::Status
VpirClient<uint64_t, uint64_t>::GetProofPreprocessMaterial<
    uint16_t, uint64_t, uint16_t, uint64_t>(
    int, const VpirServer<uint16_t, uint64_t, uint64_t, uint16_t, uint64_t,
                          uint64_t>&);
template absl::Status
VpirClient<uint32_t, uint32_t>::GetProofPreprocessMaterial<
    uint16_t, int16_t, uint16_t, int16_t>(
    int, const VpirServer<uint16_t, uint32_t, int16_t, uint16_t, uint32_t,
                          int16_t>&);
template absl::Status
VpirClient<uint64_t, uint64_t>::GetProofPreprocessMaterial<
    uint16_t, int16_t, uint16_t, int16_t>(
    int, const VpirServer<uint16_t, uint64_t, int16_t, uint16_t, uint64_t,
                          int16_t>&);
template absl::Status
VpirClient<uint32_t, uint32_t>::GetProofPreprocessMaterial<
    uint16_t, int32_t, uint16_t, int32_t>(
    int, const VpirServer<uint16_t, uint32_t, int32_t, uint16_t, uint32_t,
                          int32_t>&);
template absl::Status
VpirClient<uint64_t, uint64_t>::GetProofPreprocessMaterial<
    uint16_t, int32_t, uint16_t, int32_t>(
    int, const VpirServer<uint16_t, uint64_t, int32_t, uint16_t, uint64_t,
                          int32_t>&);
template absl::Status
VpirClient<uint32_t, uint32_t>::GetProofPreprocessMaterial<
    uint8_t, int32_t, uint8_t, int32_t>(
    int,
    const VpirServer<uint8_t, uint32_t, int32_t, uint8_t, uint32_t, int32_t>&);
template absl::Status
VpirClient<uint64_t, uint64_t>::GetProofPreprocessMaterial<
    uint8_t, int32_t, uint8_t, int32_t>(
    int,
    const VpirServer<uint8_t, uint64_t, int32_t, uint8_t, uint64_t, int32_t>&);
template absl::Status
VpirClient<uint32_t, uint32_t>::GetProofPreprocessMaterial<
    uint8_t, int16_t, uint8_t, int16_t>(
    int,
    const VpirServer<uint8_t, uint32_t, int16_t, uint8_t, uint32_t, int16_t>&);
template absl::Status
VpirClient<uint64_t, uint64_t>::GetProofPreprocessMaterial<
    uint8_t, int16_t, uint8_t, int16_t>(
    int,
    const VpirServer<uint8_t, uint64_t, int16_t, uint8_t, uint64_t, int16_t>&);

template absl::Status
VpirClient<uint32_t, uint64_t>::GetProofPreprocessMaterial<
    uint8_t, int32_t, uint8_t, int32_t>(
    int,
    const VpirServer<uint8_t, uint32_t, int32_t, uint8_t, uint64_t, int32_t>&);
template absl::Status
VpirClient<uint32_t, uint64_t>::GetProofPreprocessMaterial<
    uint8_t, int16_t, uint8_t, int32_t>(
    int,
    const VpirServer<uint8_t, uint32_t, int16_t, uint8_t, uint64_t, int32_t>&);
template absl::Status
VpirClient<uint32_t, uint64_t>::GetProofPreprocessMaterial<
    uint16_t, int32_t, uint16_t, int32_t>(
    int, const VpirServer<uint16_t, uint32_t, int32_t, uint16_t, uint64_t,
                          int32_t>&);
template absl::Status
VpirClient<uint32_t, uint64_t>::GetProofPreprocessMaterial<
    uint16_t, int16_t, uint16_t, int32_t>(
    int, const VpirServer<uint16_t, uint32_t, int16_t, uint16_t, uint64_t,
                          int32_t>&);

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
