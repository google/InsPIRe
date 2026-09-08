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

#include "crypto/external_product.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "crypto/status_macros.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename CoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>> ExternalProduct(
    const RlweParams<CoeffType>& params, const GadgetParams& gadget_params,
    const RlweCiphertext<CoeffType>& ct,
    const RgswCiphertext<CoeffType>& rgsw_ct, FftContext& ctx) {
  const int d = params.Degree();
  const CoeffType q = params.Modulus();

  if (ct.a.Len() != d || ct.b.Len() != d) {
    return absl::InvalidArgumentError("ct lengths must match d.");
  }
  if (rgsw_ct.u.size() != gadget_params.num_digits ||
      rgsw_ct.v.size() != gadget_params.num_digits) {
    return absl::InvalidArgumentError("rgsw_ct sizes must match num_digits.");
  }
  for (int i = 0; i < gadget_params.num_digits; ++i) {
    if (rgsw_ct.u[i].a.Len() != d || rgsw_ct.u[i].b.Len() != d ||
        rgsw_ct.v[i].a.Len() != d || rgsw_ct.v[i].b.Len() != d) {
      return absl::InvalidArgumentError(
          "rgsw_ct element lengths must match d.");
    }
  }

  int log_q = (q == 0) ? (sizeof(CoeffType) * 8) : absl::bit_width(q - 1);

  // Compute g^-1(a)
  auto a_mod = ct.a;
  ASSIGN_OR_RETURN(a_mod, a_mod.LowBits(log_q));
  ASSIGN_OR_RETURN(auto a_inv,
                   a_mod.GadgetInv(params.LogModulus(), gadget_params.log_digit,
                                   gadget_params.num_digits));

  // Compute g^-1(b)
  auto b_mod = ct.b;
  ASSIGN_OR_RETURN(b_mod, b_mod.LowBits(log_q));
  ASSIGN_OR_RETURN(auto b_inv,
                   b_mod.GadgetInv(params.LogModulus(), gadget_params.log_digit,
                                   gadget_params.num_digits));

  // Compute inner products:
  // (a', b') = (<g^-1(a), rgsw_ct.u.a> + <g^-1(b), rgsw_ct.v.a>,
  //             <g^-1(a), rgsw_ct.u.b> + <g^-1(b), rgsw_ct.v.b>)
  std::vector<Polynomial<CoeffType>> u_a_joined, v_a_joined;
  std::vector<Polynomial<CoeffType>> u_b_joined, v_b_joined;
  u_a_joined.reserve(2 * gadget_params.num_digits);
  v_a_joined.reserve(2 * gadget_params.num_digits);
  u_b_joined.reserve(2 * gadget_params.num_digits);
  v_b_joined.reserve(2 * gadget_params.num_digits);

  for (int i = 0; i < gadget_params.num_digits; ++i) {
    u_a_joined.push_back(a_inv[i]);
    v_a_joined.push_back(rgsw_ct.u[i].a);

    u_b_joined.push_back(a_inv[i]);
    v_b_joined.push_back(rgsw_ct.u[i].b);
  }

  for (int i = 0; i < gadget_params.num_digits; ++i) {
    u_a_joined.push_back(b_inv[i]);
    v_a_joined.push_back(rgsw_ct.v[i].a);

    u_b_joined.push_back(b_inv[i]);
    v_b_joined.push_back(rgsw_ct.v[i].b);
  }

  int max_allowed_chunk = static_cast<int>(
      (53.0 - std::log2(2 * gadget_params.num_digits) - std::log2(d)) / 2.0);
  int chunk_bits = std::min(20, max_allowed_chunk);

  ASSIGN_OR_RETURN(auto a_prime,
                   Polynomial<CoeffType>::InnerProductFft(
                       u_a_joined, v_a_joined, ctx, gadget_params.log_digit,
                       log_q, chunk_bits));
  ASSIGN_OR_RETURN(auto b_prime,
                   Polynomial<CoeffType>::InnerProductFft(
                       u_b_joined, v_b_joined, ctx, gadget_params.log_digit,
                       log_q, chunk_bits));

  return RlweCiphertext<CoeffType>{std::move(a_prime), std::move(b_prime)};
}

template absl::StatusOr<RlweCiphertext<uint32_t>> ExternalProduct<uint32_t>(
    const RlweParams<uint32_t>&, const GadgetParams&,
    const RlweCiphertext<uint32_t>&, const RgswCiphertext<uint32_t>&,
    FftContext&);
template absl::StatusOr<RlweCiphertext<uint64_t>> ExternalProduct<uint64_t>(
    const RlweParams<uint64_t>&, const GadgetParams&,
    const RlweCiphertext<uint64_t>&, const RgswCiphertext<uint64_t>&,
    FftContext&);

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
