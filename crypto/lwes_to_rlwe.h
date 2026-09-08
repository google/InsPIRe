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

#ifndef CRYPTO_LWES_TO_RLWE_H_
#define CRYPTO_LWES_TO_RLWE_H_

#include <vector>

#include "crypto/encryption.h"
#include "crypto/matrix.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// Represents the output of the PreprocessPack algorithm.
template <typename CoeffType>
struct PreprocessPackOutput {
  Polynomial<CoeffType> a_tilde_agg;
  std::vector<std::vector<Polynomial<CoeffType>>> t_vec_g;
  std::vector<std::vector<Polynomial<CoeffType>>> t_vec_gh;
  std::vector<Polynomial<CoeffType>> t_vec_h;
};

// Represents the output of the PreprocessMatrixPack algorithm.
// This stores the aggregated polynomial and the matrix used to multiply
// the preprocessed terms with the key switching keys.
template <typename CoeffType, typename MatCoeffType = CoeffType>
struct PreprocessMatrixPackOutput {
  Polynomial<CoeffType> a_tilde_agg;
  Matrix<MatCoeffType> matrix;
  std::vector<Polynomial<CoeffType>> t_vec_h;
};

// PreprocessPack preprocesses the random vector components of d LWE ciphertexts
// for aggregation into a single RLWE ciphertext. In the LWE ciphertexts, each
// a_i corresponds to an encryption evaluating against a target index within
// the combined polynomial.
//
// This is the algorithm described in https://eprint.iacr.org/2025/1352.pdf
// adapted to support power of two moduli.
//
// a_is: d vectors of length d representing the random components of d LWE
// ciphertexts.
// w_vec_g: Random component of key switching key 1, encrypting
// Automorph(s, 5), represented as a vector of polynomials.
// w__vec_h: Random component of key switching key 2, encrypting
// Automorph(s, 2d-1), represented as a vector of polynomials.
// log_q: The underlying ciphertext modulus bit length.
template <typename CoeffType>
absl::StatusOr<PreprocessPackOutput<CoeffType>> PreprocessPack(
    const RlweParams<CoeffType>& params,
    const std::vector<std::vector<CoeffType>>& a_is,
    const std::vector<Polynomial<CoeffType>>& w_vec_g,
    const std::vector<Polynomial<CoeffType>>& w_vec_h,
    const GadgetParams& gadget_params, FftContext& ctx);

// Pack aggregates the second components (message encodings) of the individual
// LWE ciphertexts into the resulting RLWE ciphertext using the preprocessed
// materials from PreprocessPack.
//
// This is the algorithm described in https://eprint.iacr.org/2025/1352.pdf
// adapted to support power of two moduli.
//
// b: The scalar elements corresponding to the second LWE components (length d).
// y_vec_g: Message encoding component of key switching key 1, encrypting
// Automorph(s, 5), represented as a vector of polynomials.
// y_vec_h: Message encoding component of key switching key 2, encrypting
// Automorph(s, 2d-1), represented as a vector of polynomials.
// preprocess_output: The generated Gadget decomposition traces output from
// PreprocessPack.
// log_q: The underlying ciphertext modulus bit length.
template <typename CoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>> Pack(
    const RlweParams<CoeffType>& params, const std::vector<CoeffType>& b,
    const std::vector<Polynomial<CoeffType>>& y_vec_g,
    const std::vector<Polynomial<CoeffType>>& y_vec_h,
    const PreprocessPackOutput<CoeffType>& preprocess_output,
    const GadgetParams& gadget_params, FftContext& ctx);

// PreprocessMatrixPack acts similarly to PreprocessPack, but generates a
// matrix representation `PreprocessMatrixPackOutput.matrix` of the gadget
// decomposition traces `t_vec_g` and `t_vec_gh` such that:
//   b_agg_partial = matrix.Multiply(y_vec_g_vec)
// where y_vec_g_vec is the flattened vector of y_vec_g's coefficients.
// This replaces the nested summation of inner products in the original
// PreprocessPack.
template <typename CoeffType, typename MatCoeffType = CoeffType>
absl::StatusOr<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
PreprocessMatrixPack(const RlweParams<CoeffType>& params,
                     const std::vector<std::vector<CoeffType>>& a_is,
                     const std::vector<Polynomial<CoeffType>>& w_vec_g,
                     const std::vector<Polynomial<CoeffType>>& w_vec_h,
                     const GadgetParams& gadget_params, FftContext& ctx);

// FinalizeMatrixPack takes the precomputed `b_agg_partial` (result of matrix
// multiplication) and aggregates it with the remainder parts.
// This allows performing one large matrix multiplication for multiple
// chunks/shards externally.
template <typename CoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>> FinalizeMatrixPack(
    const RlweParams<CoeffType>& params, const std::vector<CoeffType>& b,
    const std::vector<CoeffType>& b_agg_partial,
    const std::vector<Polynomial<CoeffType>>& y_vec_h,
    const std::vector<Polynomial<CoeffType>>& t_vec_h,
    const Polynomial<CoeffType>& a_tilde_agg, const GadgetParams& gadget_params,
    FftContext& ctx);

// MatrixPack uses the `PreprocessMatrixPackOutput` matrix to
// efficiently compute the aggregation over `y_vec_g` through matrix-vector
// multiplication, aggregating that result with the remainder parts.
template <typename CoeffType, typename MatCoeffType = CoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>> MatrixPack(
    const RlweParams<CoeffType>& params, const std::vector<CoeffType>& b,
    const std::vector<Polynomial<CoeffType>>& y_vec_g,
    const std::vector<Polynomial<CoeffType>>& y_vec_h,
    const PreprocessMatrixPackOutput<CoeffType, MatCoeffType>&
        preprocess_output,
    const GadgetParams& gadget_params, FftContext& ctx);

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_LWES_TO_RLWE_H_
