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

#include "crypto/lwes_to_rlwe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/matrix.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "crypto/status_macros.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

namespace {

// Helper for cyclic shift multiplication by X^power.
// Shifts coefficients by `power` positions (cyclic shift). Negates if it wraps
// around.
template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> MultiplyByXPower(
    const Polynomial<CoeffType>& poly, int power) {
  const int n = poly.Len();
  if (power < 0 || power >= n) {
    return absl::InvalidArgumentError("Invalid power for cyclic shift.");
  }
  std::vector<CoeffType> result(n);
  const auto& coeffs = poly.Coeffs();
  for (int i = 0; i < n; ++i) {
    if (i < power) {
      result[i] = -coeffs[n - power + i];
    } else {
      result[i] = coeffs[i - power];
    }
  }
  return Polynomial<CoeffType>::Create(result);
}

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

// Computes 5^k % (2n)
int ComputePower(int k, int two_n) {
  int64_t res = 1;
  for (int i = 0; i < k; ++i) {
    res = (res * 5) % two_n;
  }
  return res;
}

// Evaluates the polynomial P(Z) = \sum_{i=0}^{d-1} c[i] Z^i at Z = X^{2k} for
// k = 0, ..., d-1 using the Cooley-Tukey FFT over the ring Z_q[X]/(X^d+1).
// The roots of unity are powers of Z = X^2.
template <typename CoeffType>
absl::Status EvaluateNttInPlace(std::vector<Polynomial<CoeffType>>& c) {
  const int d = c.size();
  const int log2_d = absl::bit_width(static_cast<uint32_t>(d - 1));

  // 1) Bit-reversal permutation
  for (int i = 0; i < d; ++i) {
    int rev = 0;
    for (int b = 0; b < log2_d; ++b) {
      if ((i >> b) & 1) {
        rev |= (1 << (log2_d - 1 - b));
      }
    }
    if (i < rev) {
      std::swap(c[i], c[rev]);
    }
  }

  // 2) Iterative FFT stages
  for (int len = 2; len <= d; len *= 2) {
    int half = len / 2;
    int step = d / len;
    for (int i = 0; i < d; i += len) {
      for (int j = 0; j < half; ++j) {
        Polynomial<CoeffType> u = c[i + j];
        // Multiply c[i + j + half] by the twiddle factor omega^j = X^{2 * j *
        // step}
        ASSIGN_OR_RETURN(Polynomial<CoeffType> v,
                         MultiplyByXPower(c[i + j + half], 2 * j * step));
        ASSIGN_OR_RETURN(c[i + j], u.Add(v));
        ASSIGN_OR_RETURN(Polynomial<CoeffType> v_neg, v.Negate());
        ASSIGN_OR_RETURN(c[i + j + half], u.Add(v_neg));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

// Constructs a matrix that represents the linear transformation applied to
// the polynomials `y_vec_g`. This allows computing the inner products and
// automorphisms via a single vector-matrix multiplication.
//
// Assume the number of limbs is 1. We want to transform the expression:
//   \sum_{j = 0}^{d/2-2} t_vec_g[j] * Aut(y_vec_g, 5^j)
//     + t_vec_gh[j] * Aut(y_vec_g, (2d-1)*5^j)
// into its matrix-vector multiplication equivalent:
//   M * v(y_vec_g) where v(y_vec_g) is the vector representation of the
// coefficients of y_vec_g.
//
// First, we note that this is possible because all operations involved are
// linear: polynomial additions and multiplications are clear, and automorphism
// is simply a signed permutation of the coefficients.
//
// Given this fact, we observe that the c-th column of M corresponds to the
// coefficients of the following expression where y_vec_g = X^c:
//   \sum_{j=0}^{d/2-2} t_vec_g[j] * Aut(X^c, 5^j)
//     + t_vec_gh[j] * Aut(X^c, (2d-1)*5^j)
//   = \sum_{j=0}^{d/2-2} t_vec_g[j] * X^{c*5^j}
//     + t_vec_gh[j] * X^{c*(2d-1)*5^j}
//
// By observing that the powers 5^j and (2d-1)*5^j modulo 2d distinctively
// map to the set of odd integers {2i+1 | i = 0, ..., d-1}, we can define
// t_i(X) as the corresponding coefficient polynomial t_vec_g[j] or t_vec_gh[j].
// This simplifies the summation bounds over all odd powers:
//   = \sum_{i=0}^{d-1} t_i(X) * X^{c*(2i+1)}
//   = X^c * \sum_{i=0}^{d-1} t_i(X) * (X^{2c})^i
//
// Expressing the summation part as a polynomial over a variable Z, we get:
//   P(Z) = \sum_{i=0}^{d-1} t_i(X) * Z^i
// Thus, the c-th column of M contains the coefficients of X^c * P(X^{2c}) for
// c = 0, ..., d-1. Since Z = X^2 acts as a d-th primitive root of unity in
// Z_q[X]/(X^d+1), we can efficiently evaluate P(X^{2c}) for all c using the
// FFT in O(d log d) ring operations.
template <typename CoeffType, typename MatCoeffType>
absl::StatusOr<Matrix<MatCoeffType>> BuildPreprocessMatrix(
    const PreprocessPackOutput<CoeffType>& preprocess_output, const int d,
    const int k, int log_q) {
  // `matrix_data` has `d` rows and `k * d` columns. It consists of `k` blocks,
  // each of size `d x d`.
  std::vector<std::vector<MatCoeffType>> matrix_data(
      d, std::vector<MatCoeffType>(k * d, 0));

  std::vector<CoeffType> zeros(d, 0);
  ASSIGN_OR_RETURN(Polynomial<CoeffType> zero_poly,
                   Polynomial<CoeffType>::Create(zeros));

  // Iterate over each of the `k` polynomials in `y_vec_g`. For each polynomial,
  // we compute a `d x d` block representing its contribution to the final
  // output.
  for (int l = 0; l < k; ++l) {
    std::vector<Polynomial<CoeffType>> t(d, zero_poly);

    // Map the packing polynomials t_vec_g and t_vec_gh to their corresponding
    // odd power index `i = (power - 1) / 2`.
    for (int j = 0; j <= d / 2 - 2; ++j) {
      const int power_g = ComputePower(j, 2 * d);
      const int index_g = (power_g - 1) / 2;
      t[index_g] = preprocess_output.t_vec_g[j][l];

      const int power_gh = (1LL * power_g * (2 * d - 1)) % (2 * d);
      const int index_gh = (power_gh - 1) / 2;
      t[index_gh] = preprocess_output.t_vec_gh[j][l];
    }

    RETURN_IF_ERROR(EvaluateNttInPlace(t));

    // Evaluate the c-th column of M by multiplying the FFT output P(X^{2c})
    // by X^c, and assign to the matrix block.
    for (int c = 0; c < d; ++c) {
      ASSIGN_OR_RETURN(Polynomial<CoeffType> final_poly,
                       MultiplyByXPower(t[c], c));
      const auto& coeffs = final_poly.Coeffs();
      for (int r = 0; r < d; ++r) {
        // Convert coefficients to centered representations modulo 2^log_q.
        // This reduces the magnitude of matrix coefficients allowing them to be
        // packed efficiently into smaller signed/unsigned container types like
        // int16_t or int32_t.
        //
        // To avoid undefined behavior when shifting by the full bit width of
        // CoeffType, we explicitly handle the threshold calculation based on
        // the log_q.
        CoeffType threshold =
            (log_q == 0 || log_q >= sizeof(CoeffType) * 8)
                ? (static_cast<CoeffType>(1) << (sizeof(CoeffType) * 8 - 1))
                : (static_cast<CoeffType>(1) << (log_q - 1));
        CoeffType val = coeffs[r];
        if (val >= threshold) {
          // For full bit widths, two's complement wrap around already acts as a
          // perfect automatic modulo subtraction.
          if (log_q < sizeof(CoeffType) * 8) {
            val = val - (static_cast<CoeffType>(1) << log_q);
          }
        }
        matrix_data[r][l * d + c] = static_cast<MatCoeffType>(val);
      }
    }
  }

  if constexpr (std::is_same_v<MatCoeffType, uint8_t> ||
                std::is_same_v<MatCoeffType, uint16_t> ||
                std::is_same_v<MatCoeffType, int32_t>) {
    return Matrix<MatCoeffType>::CreateCondensed(matrix_data);
  } else {
    return Matrix<MatCoeffType>::Create(matrix_data);
  }
}

template <typename CoeffType>
absl::StatusOr<PreprocessPackOutput<CoeffType>> PreprocessPack(
    const RlweParams<CoeffType>& params,
    const std::vector<std::vector<CoeffType>>& a_is,
    const std::vector<Polynomial<CoeffType>>& w_vec_g,
    const std::vector<Polynomial<CoeffType>>& w_vec_h,
    const GadgetParams& gadget_params, FftContext& ctx) {
  const int d = a_is.size();
  const int log_q = params.LogModulus();

  int log2_d = absl::bit_width(static_cast<uint32_t>(d - 1));

  // 1) Interpret each a_i as a_tildes_i = \sum_{j=0}^{d-1} a_i[j] * X^{-j}
  // 2) Right shift coefficients by log2(d)
  std::vector<Polynomial<CoeffType>> a_tildes(d);
  for (int i = 0; i < d; ++i) {
    std::vector<CoeffType> coeffs(d, 0);
    coeffs[0] = a_is[i][0];
    for (int j = 1; j < d; ++j) {
      coeffs[d - j] = -a_is[i][j];
    }
    ASSIGN_OR_RETURN(Polynomial<CoeffType> poly,
                     Polynomial<CoeffType>::Create(coeffs));
    ASSIGN_OR_RETURN(poly, poly.LowBits(log_q));
    ASSIGN_OR_RETURN(poly, poly.RightShift(log2_d));
    ASSIGN_OR_RETURN(a_tildes[i], poly.LowBits(log_q));
  }

  // 3) Construct a_hat_agg
  // We want to compute:
  //    a_hat_agg[k] = \sum_{j=0}^{d-1} a_tildes_j(X^{p_k}) * X^j
  // where p_k = 5^k for k < d/2 and p_k = (2d-1)5^{k-d/2} for k >= d/2.
  // The powers p_k are exactly all the odd integers modulo 2d.
  // We can rephrase the computation by evaluating at all odd powers X^{2i+1}:
  //    \sum_{j=0}^{d-1} a_tildes_j(X^{2i+1}) * X^j
  // By expanding a_tildes_j(X) = \sum_{k=0}^{d-1} a_tildes_{j,k} * X^k, we
  // have:
  //    = \sum_{j=0}^{d-1} \sum_{k=0}^{d-1} a_tildes_{j,k} * X^{(2i+1)k + j}
  //    = \sum_{k=0}^{d-1} (X^k \sum_{j=0}^{d-1} a_tildes_{j,k} * X^j) * X^{2ik}
  // Let P_k(X) = X^k \sum_{j=0}^{d-1} a_tildes_{j,k} * X^j.
  // Then:
  //    = \sum_{k=0}^{d-1} P_k(X) * (X^{2i})^k, which corresponds to
  // evaluating the polynomial P(Z) = \sum_{k=0}^{d-1} P_k(X) * Z^k at
  // Z = X^{2i} for i = 0, ..., d-1. This can be computed efficiently using the
  // FFT.

  std::vector<Polynomial<CoeffType>> a_hat_agg_prime(d);
  for (int k = 0; k < d; ++k) {
    std::vector<CoeffType> p_k_coeffs(d, 0);
    for (int j = 0; j < d; ++j) {
      p_k_coeffs[j] = a_tildes[j].Coeffs()[k];
    }
    ASSIGN_OR_RETURN(Polynomial<CoeffType> p_k,
                     Polynomial<CoeffType>::Create(p_k_coeffs));
    ASSIGN_OR_RETURN(a_hat_agg_prime[k], MultiplyByXPower(p_k, k));
  }

  RETURN_IF_ERROR(EvaluateNttInPlace(a_hat_agg_prime));

  std::vector<Polynomial<CoeffType>> a_hat_agg(d);
  for (int k = 0; k < d / 2; ++k) {
    int power = ComputePower(k, 2 * d);
    int idx = (power - 1) / 2;
    a_hat_agg[k] = a_hat_agg_prime[idx];
  }
  for (int k = d / 2; k < d; ++k) {
    int power = ComputePower(k - d / 2, 2 * d);
    power = (1LL * power * (2 * d - 1)) % (2 * d);
    int idx = (power - 1) / 2;
    a_hat_agg[k] = a_hat_agg_prime[idx];
  }

  // 4) Reduction step
  Polynomial<CoeffType> t_g = a_hat_agg[d / 2 - 1];
  Polynomial<CoeffType> t_gh = a_hat_agg[d - 1];

  std::vector<std::vector<Polynomial<CoeffType>>> t_vec_g(d / 2 - 1);
  std::vector<std::vector<Polynomial<CoeffType>>> t_vec_gh(d / 2 - 1);

  for (int i = d / 2 - 2; i >= 0; --i) {
    ASSIGN_OR_RETURN(Polynomial<CoeffType> t_g_mod, t_g.LowBits(log_q));
    ASSIGN_OR_RETURN(
        std::vector<Polynomial<CoeffType>> t_vec_g_i,
        t_g_mod.SignedGadgetInv(params.LogModulus(), gadget_params.log_digit,
                                gadget_params.num_digits));
    t_vec_g[i] = t_vec_g_i;

    const int power_g = ComputePower(i, 2 * d);
    std::vector<Polynomial<CoeffType>> w_vec_i_g(w_vec_g.size());
    for (int k = 0; k < w_vec_g.size(); ++k) {
      ASSIGN_OR_RETURN(w_vec_i_g[k], w_vec_g[k].Automorph(power_g));
    }

    ASSIGN_OR_RETURN(Polynomial<CoeffType> inner_g,
                     InnerProduct(t_vec_g_i, w_vec_i_g, ctx,
                                  gadget_params.log_digit, log_q, true));
    ASSIGN_OR_RETURN(t_g, a_hat_agg[i].Add(inner_g));

    const int power_gh = (1LL * power_g * (2 * d - 1)) % (2 * d);
    std::vector<Polynomial<CoeffType>> w_vec_i_gh(w_vec_g.size());
    for (int k = 0; k < w_vec_g.size(); ++k) {
      ASSIGN_OR_RETURN(w_vec_i_gh[k], w_vec_g[k].Automorph(power_gh));
    }

    ASSIGN_OR_RETURN(Polynomial<CoeffType> t_gh_mod, t_gh.LowBits(log_q));
    ASSIGN_OR_RETURN(
        std::vector<Polynomial<CoeffType>> t_vec_gh_i,
        t_gh_mod.SignedGadgetInv(params.LogModulus(), gadget_params.log_digit,
                                 gadget_params.num_digits));
    t_vec_gh[i] = t_vec_gh_i;

    ASSIGN_OR_RETURN(Polynomial<CoeffType> inner_gh,
                     InnerProduct(t_vec_gh_i, w_vec_i_gh, ctx,
                                  gadget_params.log_digit, log_q, true));
    ASSIGN_OR_RETURN(t_gh, a_hat_agg[d / 2 + i].Add(inner_gh));
  }

  ASSIGN_OR_RETURN(Polynomial<CoeffType> t_gh_mod, t_gh.LowBits(log_q));
  ASSIGN_OR_RETURN(
      std::vector<Polynomial<CoeffType>> t_vec_h,
      t_gh_mod.SignedGadgetInv(params.LogModulus(), gadget_params.log_digit,
                               gadget_params.num_digits));

  ASSIGN_OR_RETURN(Polynomial<CoeffType> inner_h,
                   InnerProduct(t_vec_h, w_vec_h, ctx, gadget_params.log_digit,
                                log_q, true));
  ASSIGN_OR_RETURN(Polynomial<CoeffType> a_tilde_agg, t_g.Add(inner_h));
  ASSIGN_OR_RETURN(a_tilde_agg, a_tilde_agg.LowBits(log_q));

  return PreprocessPackOutput<CoeffType>{a_tilde_agg, t_vec_g, t_vec_gh,
                                         t_vec_h};
}

template <typename CoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>> Pack(
    const RlweParams<CoeffType>& params, const std::vector<CoeffType>& b,
    const std::vector<Polynomial<CoeffType>>& y_vec_g,
    const std::vector<Polynomial<CoeffType>>& y_vec_h,
    const PreprocessPackOutput<CoeffType>& preprocess_output,
    const GadgetParams& gadget_params, FftContext& ctx) {
  const int log_q = params.LogModulus();
  const int d = b.size();

  // Calculate b_agg
  std::vector<CoeffType> b_coeffs = b;
  ASSIGN_OR_RETURN(Polynomial<CoeffType> b_poly,
                   Polynomial<CoeffType>::Create(b_coeffs));

  Polynomial<CoeffType> b_agg = b_poly;
  ASSIGN_OR_RETURN(b_agg, b_agg.LowBits(log_q));

  for (int j = 0; j <= d / 2 - 2; ++j) {
    int power_g = ComputePower(j, 2 * d);
    std::vector<Polynomial<CoeffType>> y_vec_g_j(y_vec_g.size());
    for (int k = 0; k < y_vec_g.size(); ++k) {
      ASSIGN_OR_RETURN(y_vec_g_j[k], y_vec_g[k].Automorph(power_g));
    }
    ASSIGN_OR_RETURN(Polynomial<CoeffType> inner_g,
                     InnerProduct(preprocess_output.t_vec_g[j], y_vec_g_j, ctx,
                                  gadget_params.log_digit, log_q, true));
    ASSIGN_OR_RETURN(b_agg, b_agg.Add(inner_g));
    ASSIGN_OR_RETURN(b_agg, b_agg.LowBits(log_q));

    int power_gh = (1LL * power_g * (2 * d - 1)) % (2 * d);
    std::vector<Polynomial<CoeffType>> y_vec_gh_j(y_vec_g.size());
    for (int k = 0; k < y_vec_g.size(); ++k) {
      ASSIGN_OR_RETURN(y_vec_gh_j[k], y_vec_g[k].Automorph(power_gh));
    }
    ASSIGN_OR_RETURN(Polynomial<CoeffType> inner_gh,
                     InnerProduct(preprocess_output.t_vec_gh[j], y_vec_gh_j,
                                  ctx, gadget_params.log_digit, log_q, true));
    ASSIGN_OR_RETURN(b_agg, b_agg.Add(inner_gh));
    ASSIGN_OR_RETURN(b_agg, b_agg.LowBits(log_q));
  }

  ASSIGN_OR_RETURN(Polynomial<CoeffType> inner_h,
                   InnerProduct(preprocess_output.t_vec_h, y_vec_h, ctx,
                                gadget_params.log_digit, log_q, true));
  ASSIGN_OR_RETURN(b_agg, b_agg.Add(inner_h));
  ASSIGN_OR_RETURN(b_agg, b_agg.LowBits(log_q));

  return RlweCiphertext<CoeffType>{preprocess_output.a_tilde_agg, b_agg};
}

template <typename CoeffType, typename MatCoeffType>
absl::StatusOr<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
PreprocessMatrixPack(const RlweParams<CoeffType>& params,
                     const std::vector<std::vector<CoeffType>>& a_is,
                     const std::vector<Polynomial<CoeffType>>& w_vec_g,
                     const std::vector<Polynomial<CoeffType>>& w_vec_h,
                     const GadgetParams& gadget_params, FftContext& ctx) {
  int d = a_is.size();

  // 1-5) Re-use PreprocessPack up to this point.
  ASSIGN_OR_RETURN(
      auto preprocess_output,
      PreprocessPack(params, a_is, w_vec_g, w_vec_h, gadget_params, ctx));

  // Construct the matrix from t_vec_g and t_vec_gh.
  // The matrix converts polynomial multiplication & summation & automorphism
  // into matrix operations. This enables factoring out Y_g and represent the
  // entire operation as a single matrix-vector multiplication.
  const int k = w_vec_g.size();
  ASSIGN_OR_RETURN(auto matrix,
                   (BuildPreprocessMatrix<CoeffType, MatCoeffType>(
                       preprocess_output, d, k, params.LogModulus())));

  return PreprocessMatrixPackOutput<CoeffType, MatCoeffType>{
      preprocess_output.a_tilde_agg, matrix, preprocess_output.t_vec_h};
}

template <typename CoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>> FinalizeMatrixPack(
    const RlweParams<CoeffType>& params, const std::vector<CoeffType>& b,
    const std::vector<CoeffType>& b_agg_partial,
    const std::vector<Polynomial<CoeffType>>& y_vec_h,
    const std::vector<Polynomial<CoeffType>>& t_vec_h,
    const Polynomial<CoeffType>& a_tilde_agg, const GadgetParams& gadget_params,
    FftContext& ctx) {
  int d = b.size();

  if (b_agg_partial.size() != d) {
    return absl::InvalidArgumentError("b_agg_partial size mismatch.");
  }

  // Incorporate b_agg_partial into b_agg
  std::vector<CoeffType> b_coeffs = b;
  for (int m = 0; m < d; ++m) {
    b_coeffs[m] += b_agg_partial[m];
  }
  ASSIGN_OR_RETURN(Polynomial<CoeffType> b_agg,
                   Polynomial<CoeffType>::Create(b_coeffs));
  ASSIGN_OR_RETURN(b_agg, b_agg.LowBits(params.LogModulus()));

  // Add InnerProduct(y_vec_h, t_vec_h)
  ASSIGN_OR_RETURN(Polynomial<CoeffType> inner_h,
                   InnerProduct(t_vec_h, y_vec_h, ctx, gadget_params.log_digit,
                                params.LogModulus(), true));
  ASSIGN_OR_RETURN(b_agg, b_agg.Add(inner_h));
  ASSIGN_OR_RETURN(b_agg, b_agg.LowBits(params.LogModulus()));

  return RlweCiphertext<CoeffType>{a_tilde_agg, b_agg};
}

template <typename CoeffType, typename MatCoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>> MatrixPack(
    const RlweParams<CoeffType>& params, const std::vector<CoeffType>& b,
    const std::vector<Polynomial<CoeffType>>& y_vec_g,
    const std::vector<Polynomial<CoeffType>>& y_vec_h,
    const PreprocessMatrixPackOutput<CoeffType, MatCoeffType>&
        preprocess_output,
    const GadgetParams& gadget_params, FftContext& ctx) {
  int d = b.size();
  int k = y_vec_g.size();

  // Vectorize Y_g
  std::vector<CoeffType> y_vec_g_vec(k * d);
  for (int l = 0; l < k; ++l) {
    for (int m = 0; m < d; ++m) {
      y_vec_g_vec[l * d + m] = y_vec_g[l].Coeffs()[m];
    }
  }

  // Compute inner products using matrix multiplication.
  ASSIGN_OR_RETURN(
      std::vector<CoeffType> b_agg_partial,
      preprocess_output.matrix.template Multiply<CoeffType>(y_vec_g_vec));

  return FinalizeMatrixPack(params, b, b_agg_partial, y_vec_h,
                            preprocess_output.t_vec_h,
                            preprocess_output.a_tilde_agg, gadget_params, ctx);
}

template absl::StatusOr<PreprocessPackOutput<uint32_t>> PreprocessPack(
    const RlweParams<uint32_t>&, const std::vector<std::vector<uint32_t>>&,
    const std::vector<Polynomial<uint32_t>>&,
    const std::vector<Polynomial<uint32_t>>&, const GadgetParams&, FftContext&);

template absl::StatusOr<RlweCiphertext<uint32_t>> Pack(
    const RlweParams<uint32_t>&, const std::vector<uint32_t>&,
    const std::vector<Polynomial<uint32_t>>&,
    const std::vector<Polynomial<uint32_t>>&,
    const PreprocessPackOutput<uint32_t>&, const GadgetParams&, FftContext&);

template absl::StatusOr<PreprocessPackOutput<uint64_t>> PreprocessPack(
    const RlweParams<uint64_t>&, const std::vector<std::vector<uint64_t>>&,
    const std::vector<Polynomial<uint64_t>>&,
    const std::vector<Polynomial<uint64_t>>&, const GadgetParams&, FftContext&);

template absl::StatusOr<RlweCiphertext<uint64_t>> Pack(
    const RlweParams<uint64_t>&, const std::vector<uint64_t>&,
    const std::vector<Polynomial<uint64_t>>&,
    const std::vector<Polynomial<uint64_t>>&,
    const PreprocessPackOutput<uint64_t>&, const GadgetParams&, FftContext&);

// Instantiations for PreprocessMatrixPack
template absl::StatusOr<PreprocessMatrixPackOutput<uint32_t, uint32_t>>
PreprocessMatrixPack(const RlweParams<uint32_t>&,
                     const std::vector<std::vector<uint32_t>>&,
                     const std::vector<Polynomial<uint32_t>>&,
                     const std::vector<Polynomial<uint32_t>>&,
                     const GadgetParams&, FftContext&);

template absl::StatusOr<PreprocessMatrixPackOutput<uint64_t, uint64_t>>
PreprocessMatrixPack(const RlweParams<uint64_t>&,
                     const std::vector<std::vector<uint64_t>>&,
                     const std::vector<Polynomial<uint64_t>>&,
                     const std::vector<Polynomial<uint64_t>>&,
                     const GadgetParams&, FftContext&);

template absl::StatusOr<PreprocessMatrixPackOutput<uint32_t, int16_t>>
PreprocessMatrixPack(const RlweParams<uint32_t>&,
                     const std::vector<std::vector<uint32_t>>&,
                     const std::vector<Polynomial<uint32_t>>&,
                     const std::vector<Polynomial<uint32_t>>&,
                     const GadgetParams&, FftContext&);

template absl::StatusOr<PreprocessMatrixPackOutput<uint64_t, int16_t>>
PreprocessMatrixPack(const RlweParams<uint64_t>&,
                     const std::vector<std::vector<uint64_t>>&,
                     const std::vector<Polynomial<uint64_t>>&,
                     const std::vector<Polynomial<uint64_t>>&,
                     const GadgetParams&, FftContext&);

template absl::StatusOr<PreprocessMatrixPackOutput<uint32_t, int32_t>>
PreprocessMatrixPack(const RlweParams<uint32_t>&,
                     const std::vector<std::vector<uint32_t>>&,
                     const std::vector<Polynomial<uint32_t>>&,
                     const std::vector<Polynomial<uint32_t>>&,
                     const GadgetParams&, FftContext&);

template absl::StatusOr<PreprocessMatrixPackOutput<uint64_t, int32_t>>
PreprocessMatrixPack(const RlweParams<uint64_t>&,
                     const std::vector<std::vector<uint64_t>>&,
                     const std::vector<Polynomial<uint64_t>>&,
                     const std::vector<Polynomial<uint64_t>>&,
                     const GadgetParams&, FftContext&);

// Instantiations for FinalizeMatrixPack
template absl::StatusOr<RlweCiphertext<uint32_t>> FinalizeMatrixPack(
    const RlweParams<uint32_t>&, const std::vector<uint32_t>&,
    const std::vector<uint32_t>&, const std::vector<Polynomial<uint32_t>>&,
    const std::vector<Polynomial<uint32_t>>&, const Polynomial<uint32_t>&,
    const GadgetParams&, FftContext&);

template absl::StatusOr<RlweCiphertext<uint64_t>> FinalizeMatrixPack(
    const RlweParams<uint64_t>&, const std::vector<uint64_t>&,
    const std::vector<uint64_t>&, const std::vector<Polynomial<uint64_t>>&,
    const std::vector<Polynomial<uint64_t>>&, const Polynomial<uint64_t>&,
    const GadgetParams&, FftContext&);

// Instantiations for MatrixPack
template absl::StatusOr<RlweCiphertext<uint32_t>> MatrixPack(
    const RlweParams<uint32_t>&, const std::vector<uint32_t>&,
    const std::vector<Polynomial<uint32_t>>&,
    const std::vector<Polynomial<uint32_t>>&,
    const PreprocessMatrixPackOutput<uint32_t, uint32_t>&, const GadgetParams&,
    FftContext&);

template absl::StatusOr<RlweCiphertext<uint64_t>> MatrixPack(
    const RlweParams<uint64_t>&, const std::vector<uint64_t>&,
    const std::vector<Polynomial<uint64_t>>&,
    const std::vector<Polynomial<uint64_t>>&,
    const PreprocessMatrixPackOutput<uint64_t, uint64_t>&, const GadgetParams&,
    FftContext&);

template absl::StatusOr<RlweCiphertext<uint32_t>> MatrixPack(
    const RlweParams<uint32_t>&, const std::vector<uint32_t>&,
    const std::vector<Polynomial<uint32_t>>&,
    const std::vector<Polynomial<uint32_t>>&,
    const PreprocessMatrixPackOutput<uint32_t, int16_t>&, const GadgetParams&,
    FftContext&);

template absl::StatusOr<RlweCiphertext<uint64_t>> MatrixPack(
    const RlweParams<uint64_t>&, const std::vector<uint64_t>&,
    const std::vector<Polynomial<uint64_t>>&,
    const std::vector<Polynomial<uint64_t>>&,
    const PreprocessMatrixPackOutput<uint64_t, int16_t>&, const GadgetParams&,
    FftContext&);

template absl::StatusOr<RlweCiphertext<uint32_t>> MatrixPack(
    const RlweParams<uint32_t>&, const std::vector<uint32_t>&,
    const std::vector<Polynomial<uint32_t>>&,
    const std::vector<Polynomial<uint32_t>>&,
    const PreprocessMatrixPackOutput<uint32_t, int32_t>&, const GadgetParams&,
    FftContext&);

template absl::StatusOr<RlweCiphertext<uint64_t>> MatrixPack(
    const RlweParams<uint64_t>&, const std::vector<uint64_t>&,
    const std::vector<Polynomial<uint64_t>>&,
    const std::vector<Polynomial<uint64_t>>&,
    const PreprocessMatrixPackOutput<uint64_t, int32_t>&, const GadgetParams&,
    FftContext&);

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
