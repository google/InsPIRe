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

#include "crypto/encryption.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "crypto/polynomial.h"
#include "absl/numeric/bits.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "crypto/status_macros.h"
#include "absl/status/statusor.h"
#include "shell_encryption/integral_types.h"
#include "shell_encryption/montgomery.h"
#include "shell_encryption/prng/prng.h"
#include "shell_encryption/sample_error.h"
#include "shell_encryption/sampler/uniform_ternary.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename CoeffType>
absl::StatusOr<RlweParams<CoeffType>> RlweParams<CoeffType>::Create(
    int degree, CoeffType modulus, int variance) {
  if (!absl::has_single_bit(static_cast<uint64_t>(degree))) {
    return absl::InvalidArgumentError("Ring degree must be a power of 2.");
  }
  if (modulus != 0 && !absl::has_single_bit(modulus)) {
    return absl::InvalidArgumentError(
        "Coefficient modulus must be a power of 2 or 0.");
  }
  if (variance <= 0) {
    return absl::InvalidArgumentError("variance must be positive.");
  }
  return RlweParams(degree, modulus, variance);
}

template <typename CoeffType>
absl::StatusOr<RlweParams<CoeffType>> RlweParams<CoeffType>::Create(
    int degree, CoeffType modulus, CoeffType modulus1_after_switch,
    CoeffType modulus2_after_switch, int variance) {
  if (!absl::has_single_bit(static_cast<uint64_t>(degree))) {
    return absl::InvalidArgumentError("Ring degree must be a power of 2.");
  }
  if (modulus != 0 && !absl::has_single_bit(modulus)) {
    return absl::InvalidArgumentError(
        "Coefficient modulus must be a power of 2 or 0.");
  }

  if (modulus1_after_switch != 0 &&
      !absl::has_single_bit(modulus1_after_switch)) {
    return absl::InvalidArgumentError(
        "Coefficient modulus1_after_switch must be a power of 2 or 0.");
  }
  if (modulus2_after_switch != 0 &&
      !absl::has_single_bit(modulus2_after_switch)) {
    return absl::InvalidArgumentError(
        "Coefficient modulus2_after_switch must be a power of 2 or 0.");
  }

  if (modulus == 0) {
    if (modulus1_after_switch < modulus2_after_switch) {
      return absl::InvalidArgumentError(
          "modulus1_after_switch must be greater than or equal to "
          "modulus2_after_switch when modulus is 0.");
    }
  } else {
    if (modulus1_after_switch > modulus ||
        modulus2_after_switch > modulus1_after_switch) {
      return absl::InvalidArgumentError(
          "mod1_after_switch and modulus2_after_switch must be less than or "
          "equal to modulus when modulus is not 0.");
    }
  }

  if (variance <= 0) {
    return absl::InvalidArgumentError("variance must be positive.");
  }
  return RlweParams(degree, modulus, modulus1_after_switch,
                    modulus2_after_switch, variance);
}

template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> SampleSecretKey(
    const RlweParams<CoeffType>& params, ::rlwe::SecurePrng& prng) {
  const int d = params.Degree();
  const CoeffType q = params.Modulus();

  using ModularInt = ::rlwe::MontgomeryInt<::rlwe::Uint32>;
  ASSIGN_OR_RETURN(auto mod_params, ModularInt::Params::Create(3));

  ASSIGN_OR_RETURN(
      std::vector<ModularInt> ternary_coeffs,
      ::rlwe::SampleFromUniformTernary<ModularInt>(d, mod_params.get(), &prng));

  std::vector<CoeffType> coeffs(d);
  for (int i = 0; i < d; ++i) {
    uint32_t val = ternary_coeffs[i].ExportInt(mod_params.get());
    if (val > 2) {
      return absl::InternalError("Invalid ternary sample from rlwe library.");
    }
    // Constant-time mapping of {2, 0, 1} to {-1, 0, 1} mod q.
    CoeffType is_one_mask = -static_cast<CoeffType>(val == 1);
    CoeffType is_two_mask = -static_cast<CoeffType>(val == 2);
    coeffs[i] = (is_one_mask & 1) | (is_two_mask & (q - 1));
  }

  return Polynomial<CoeffType>::Create(std::move(coeffs));
}

template <typename CoeffType>
absl::StatusOr<std::vector<Polynomial<CoeffType>>> SampleAComponents(
    const RlweParams<CoeffType>& params, int num_a_components,
    ::rlwe::SecurePrng& prng) {
  const int d = params.Degree();
  const CoeffType q = params.Modulus();
  if (num_a_components <= 0) {
    return absl::InvalidArgumentError("num_a_components must be positive.");
  }

  std::vector<Polynomial<CoeffType>> a_components(num_a_components);
  for (int i = 0; i < num_a_components; ++i) {
    std::vector<CoeffType> coeffs(d);
    for (int j = 0; j < d; ++j) {
      ASSIGN_OR_RETURN(auto r64, prng.Rand64());
      coeffs[j] = static_cast<CoeffType>(r64) & (q - 1);
    }
    ASSIGN_OR_RETURN(a_components[i],
                     Polynomial<CoeffType>::Create(std::move(coeffs)));
  }

  return a_components;
}

template <typename CoeffType>
absl::StatusOr<std::vector<RlweSample<CoeffType>>> GenerateRlweSamples(
    const RlweParams<CoeffType>& params,
    const NttPolynomial& secret_key_ntt,
    const std::vector<Polynomial<CoeffType>>& a_components,
    ::rlwe::SecurePrng& prng,
    const Context& ctx) {
  const int d = params.Degree();
  const CoeffType q = params.Modulus();
  const int variance = params.Variance();

  using ModularInt = ::rlwe::MontgomeryInt<::rlwe::Uint32>;
  ASSIGN_OR_RETURN(auto mod_params_large,
                   ModularInt::Params::Create((1u << 30) - 1));

  std::vector<RlweSample<CoeffType>> samples;
  samples.reserve(a_components.size());

  for (const auto& a : a_components) {
    if (a.Len() != d) {
      return absl::InvalidArgumentError(
          "a_component length must match degree.");
    }

    ASSIGN_OR_RETURN(auto as, a.Mult(secret_key_ntt, ctx));

    ASSIGN_OR_RETURN(std::vector<ModularInt> e_coeffs_large,
                     ::rlwe::SampleFromErrorDistribution<ModularInt>(
                         d, variance, &prng, mod_params_large.get()));

    const ::rlwe::Uint32 modulus = mod_params_large->modulus;
    std::vector<CoeffType> b_coeffs(d);
    for (int i = 0; i < d; ++i) {
      ::rlwe::Uint32 val = e_coeffs_large[i].ExportInt(mod_params_large.get());
      // Constant-time representation mapping to avoid branching on secret
      // noise.
      bool is_negative = val > (modulus >> 1);
      CoeffType mask = -static_cast<CoeffType>(is_negative);
      CoeffType e_val_neg = q - (modulus - val);
      CoeffType e_val_pos = val;
      CoeffType e_val = (mask & e_val_neg) | (~mask & e_val_pos);

      // mod q is equivalent to bitwise AND with q - 1 since q is a power of
      // two.
      b_coeffs[i] = (e_val - as.Coeffs()[i]) & (q - 1);
    }

    ASSIGN_OR_RETURN(auto b,
                     Polynomial<CoeffType>::Create(std::move(b_coeffs)));
    samples.push_back({a, b});
  }

  return samples;
}

template <typename CoeffType>
absl::StatusOr<std::vector<RlweSample<CoeffType>>> GenerateRlweSamples(
    const RlweParams<CoeffType>& params,
    const Polynomial<CoeffType>& secret_key,
    const std::vector<Polynomial<CoeffType>>& a_components,
    ::rlwe::SecurePrng& prng) {
  const int d = params.Degree();
  if (secret_key.Len() != d) {
    return absl::InvalidArgumentError("secret_key length must match degree.");
  }
  ASSIGN_OR_RETURN(
      auto ctx,
      Context::CreateForTernary(absl::bit_width(static_cast<uint32_t>(d)) - 1));
  ASSIGN_OR_RETURN(auto secret_key_ntt,
                   secret_key.ToNtt(ctx, /*is_ternary=*/true));
  return GenerateRlweSamples(params, secret_key_ntt, a_components, prng, ctx);
}

template <typename CoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>> EncryptFromRlweSample(
    const RlweParams<CoeffType>& params, const CoeffType plaintext_modulus,
    const RlweSample<CoeffType>& rlwe_sample,
    const std::vector<CoeffType>& message) {
  const int d = params.Degree();
  const CoeffType q = params.Modulus();
  if (plaintext_modulus <= 0) {
    return absl::InvalidArgumentError("Plaintext modulus must be positive.");
  }
  if (rlwe_sample.a.Len() != d || rlwe_sample.b.Len() != d) {
    return absl::InvalidArgumentError("sample lengths must match degree.");
  }
  if (message.size() != d) {
    return absl::InvalidArgumentError("message length must match degree.");
  }

  absl::uint128 q_128 = absl::uint128{1} << params.LogModulus();
  absl::uint128 p_128 = plaintext_modulus;

  std::vector<CoeffType> b_coeffs = rlwe_sample.b.Coeffs();
  for (int i = 0; i < d; ++i) {
    if (message[i] >= plaintext_modulus) {
      return absl::InvalidArgumentError(
          "Message components must be less than plaintext_modulus.");
    }
    absl::uint128 m_128 = message[i];
    CoeffType scaled_m =
        static_cast<CoeffType>((m_128 * q_128 + p_128 / 2) / p_128);
    b_coeffs[i] = (b_coeffs[i] + scaled_m) & (q - 1);
  }

  ASSIGN_OR_RETURN(auto b_prime,
                   Polynomial<CoeffType>::Create(std::move(b_coeffs)));

  return RlweCiphertext<CoeffType>{rlwe_sample.a, b_prime};
}

template <typename CoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>> EncryptRlwe(
    const RlweParams<CoeffType>& params, const CoeffType plaintext_modulus,
    const Polynomial<CoeffType>& secret_key,
    const std::vector<CoeffType>& message, ::rlwe::SecurePrng& prng) {
  if (plaintext_modulus <= 0) {
    return absl::InvalidArgumentError("Plaintext modulus must be positive.");
  }

  ASSIGN_OR_RETURN(auto a_components, SampleAComponents(params, 1, prng));
  ASSIGN_OR_RETURN(auto rlwe_sample,
                   GenerateRlweSamples(params, secret_key, a_components, prng));
  return EncryptFromRlweSample(params, plaintext_modulus, rlwe_sample[0],
                               message);
}

template <typename CoeffType>
absl::StatusOr<std::vector<CoeffType>> Decrypt(
    const RlweParams<CoeffType>& params, const CoeffType plaintext_modulus,
    const RlweCiphertext<CoeffType>& ciphertext,
    const NttPolynomial& secret_key_ntt,
    const Context& ctx) {
  const int d = params.Degree();
  const CoeffType q = params.Modulus();
  const int log_q = params.LogModulus();
  if (plaintext_modulus <= 0) {
    return absl::InvalidArgumentError("Plaintext modulus must be positive.");
  }
  if (ciphertext.a.Len() != d || ciphertext.b.Len() != d) {
    return absl::InvalidArgumentError("ciphertext lengths must match degree.");
  }

  ASSIGN_OR_RETURN(auto as, ciphertext.a.Mult(secret_key_ntt, ctx));
  ASSIGN_OR_RETURN(auto noisy_m, ciphertext.b.Add(as));
  ASSIGN_OR_RETURN(noisy_m, noisy_m.LowBits(log_q));

  std::vector<CoeffType> m_coeffs(d);
  absl::uint128 q_128 = absl::uint128{1} << log_q;
  absl::uint128 p_128 = plaintext_modulus;

  for (int i = 0; i < d; ++i) {
    CoeffType val = noisy_m.Coeffs()[i] & (q - 1);
    // Decode with exact fraction round(c * P / Q)
    absl::uint128 v_128 = val;
    m_coeffs[i] = static_cast<CoeffType>((v_128 * p_128 + q_128 / 2) / q_128 %
                                         plaintext_modulus);
  }

  return m_coeffs;
}

template <typename CoeffType>
absl::StatusOr<std::vector<CoeffType>> Decrypt(
    const RlweParams<CoeffType>& params, const CoeffType plaintext_modulus,
    const RlweCiphertext<CoeffType>& ciphertext,
    const Polynomial<CoeffType>& secret_key) {
  const int d = params.Degree();
  if (secret_key.Len() != d) {
    return absl::InvalidArgumentError("secret_key length must match degree.");
  }
  ASSIGN_OR_RETURN(
      auto ctx,
      Context::CreateForTernary(absl::bit_width(static_cast<uint32_t>(d)) - 1));
  ASSIGN_OR_RETURN(auto secret_key_ntt,
                   secret_key.ToNtt(ctx, /*is_ternary=*/true));
  return Decrypt(params, plaintext_modulus, ciphertext, secret_key_ntt, ctx);
}

template <typename CoeffType>
absl::StatusOr<std::vector<CoeffType>> DecryptAfterModulusSwitch(
    const RlweParams<CoeffType>& params, const CoeffType plaintext_modulus,
    const RlweCiphertext<CoeffType>& ciphertext,
    const NttPolynomial& secret_key_ntt,
    const Context& ctx) {
  const int d = params.Degree();
  if (plaintext_modulus <= 0) {
    return absl::InvalidArgumentError("Plaintext modulus must be positive.");
  }
  if (ciphertext.a.Len() != d || ciphertext.b.Len() != d) {
    return absl::InvalidArgumentError("ciphertext lengths must match degree.");
  }

  ASSIGN_OR_RETURN(auto as, ciphertext.a.Mult(secret_key_ntt, ctx));
  // mod by q1
  ASSIGN_OR_RETURN(as, as.LowBits(params.LogModulus1AfterSwitch()));
  // scale as from q1 to q2
  ASSIGN_OR_RETURN(as, as.Rescale(params.LogModulus1AfterSwitch(),
                                  params.LogModulus2AfterSwitch()));

  ASSIGN_OR_RETURN(auto noisy_m, ciphertext.b.Add(as));
  // mod by q2
  ASSIGN_OR_RETURN(noisy_m, noisy_m.LowBits(params.LogModulus2AfterSwitch()));

  CoeffType modulus2 = 1;
  if (params.LogModulus2AfterSwitch() < 8 * sizeof(CoeffType)) {
    modulus2 = CoeffType{1} << params.LogModulus2AfterSwitch();
  } else {
    modulus2 = 0;
  }

  std::vector<CoeffType> m_coeffs(d);
  absl::uint128 q_128 = absl::uint128{1} << params.LogModulus2AfterSwitch();
  absl::uint128 p_128 = plaintext_modulus;

  for (int i = 0; i < d; ++i) {
    CoeffType val = noisy_m.Coeffs()[i] & (modulus2 - 1);
    absl::uint128 v_128 = val;
    m_coeffs[i] = static_cast<CoeffType>((v_128 * p_128 + q_128 / 2) / q_128 %
                                         plaintext_modulus);
  }

  return m_coeffs;
}

template <typename CoeffType>
absl::StatusOr<std::vector<CoeffType>> DecryptAfterModulusSwitch(
    const RlweParams<CoeffType>& params, const CoeffType plaintext_modulus,
    const RlweCiphertext<CoeffType>& ciphertext,
    const Polynomial<CoeffType>& secret_key) {
  const int d = params.Degree();
  if (secret_key.Len() != d) {
    return absl::InvalidArgumentError("secret_key length must match degree.");
  }

  ASSIGN_OR_RETURN(
      auto ctx,
      Context::CreateForTernary(absl::bit_width(static_cast<uint32_t>(d)) - 1));
  ASSIGN_OR_RETURN(auto secret_key_ntt,
                   secret_key.ToNtt(ctx, /*is_ternary=*/true));
  return DecryptAfterModulusSwitch(params, plaintext_modulus, ciphertext,
                                   secret_key_ntt, ctx);
}

template <typename CoeffType>
absl::StatusOr<std::vector<RlweCiphertext<CoeffType>>>
EncryptGadgetCiphertextFromRlweSamples(
    const RlweParams<CoeffType>& params, const GadgetParams& gadget_params,
    const std::vector<RlweSample<CoeffType>>& rlwe_samples,
    const std::vector<CoeffType>& message) {
  const int d = params.Degree();
  const CoeffType q = params.Modulus();
  if (message.size() != d) {
    return absl::InvalidArgumentError("message length must match d.");
  }
  if (rlwe_samples.size() != gadget_params.num_digits) {
    return absl::InvalidArgumentError(
        "rlwe_samples length must match num_digits.");
  }

  std::vector<RlweCiphertext<CoeffType>> output;
  output.reserve(gadget_params.num_digits);

  for (int i = 0; i < gadget_params.num_digits; ++i) {
    auto w_i = rlwe_samples[i].a;
    auto b_i = rlwe_samples[i].b;

    if (w_i.Len() != d || b_i.Len() != d) {
      return absl::InvalidArgumentError(
          "rlwe_samples element length must match d.");
    }

    std::vector<CoeffType> y_coeffs(d);
    int shift_amount = i * gadget_params.log_digit;

    // Indicates the number of bits we are excluding during the approximate
    // gadget decomposition. In the case of exact gadget decomposition, this
    // value should be 0.
    int offset_bits =
        std::max(0, params.LogModulus() -
                        (gadget_params.num_digits * gadget_params.log_digit));
    for (int j = 0; j < d; ++j) {
      const CoeffType target_val = message[j];
      CoeffType scaled_target = 0;
      if (shift_amount + offset_bits < sizeof(CoeffType) * 8) {
        scaled_target = target_val << (shift_amount + offset_bits);
      }

      // y_i = b_i + scaled_target (mod q)
      y_coeffs[j] = (b_i.Coeffs()[j] + scaled_target) & (q - 1);
    }

    ASSIGN_OR_RETURN(auto y_i,
                     Polynomial<CoeffType>::Create(std::move(y_coeffs)));
    output.push_back({w_i, y_i});
  }

  return output;
}

template <typename CoeffType>
absl::StatusOr<RgswCiphertext<CoeffType>> EncryptRgswCiphertextFromRlweSamples(
    const RlweParams<CoeffType>& params, const GadgetParams& gadget_params,
    const std::vector<RlweSample<CoeffType>>& u_samples,
    const std::vector<RlweSample<CoeffType>>& v_samples,
    const Polynomial<CoeffType>& secret_key,
    const std::vector<CoeffType>& message) {
  const int d = params.Degree();
  const CoeffType q = params.Modulus();
  if (secret_key.Len() != d) {
    return absl::InvalidArgumentError("secret_key length must match d.");
  }
  if (message.size() != d) {
    return absl::InvalidArgumentError("message length must match d.");
  }

  // Fast-path: check if message is a monomial (+/- X^k), which is standard for
  // PIR second dimension query evaluation.
  int non_zero_count = 0;
  int monomial_idx = 0;
  CoeffType monomial_val = 0;
  for (int i = 0; i < d; ++i) {
    if (message[i] != 0) {
      non_zero_count++;
      monomial_idx = i;
      monomial_val = message[i];
    }
  }

  Polynomial<CoeffType> m_s = secret_key;
  if (non_zero_count == 0) {
    ASSIGN_OR_RETURN(m_s, Polynomial<CoeffType>::CreateZero(d));
  } else if (non_zero_count == 1 &&
             (monomial_val == 1 || monomial_val == (q - 1))) {
    // O(d) negacyclic shift for m(X) * s(X) mod (X^d + 1).
    std::vector<CoeffType> m_s_coeffs(d);
    const auto& sk_coeffs = secret_key.Coeffs();
    const bool is_neg = (monomial_val == (q - 1));
    for (int j = 0; j < d; ++j) {
      int target_idx = j + monomial_idx;
      bool wrap = (target_idx >= d);
      if (wrap) target_idx -= d;
      bool negate = wrap ^ is_neg;
      CoeffType val = sk_coeffs[j];
      m_s_coeffs[target_idx] = negate ? (-val & (q - 1)) : val;
    }
    ASSIGN_OR_RETURN(m_s, Polynomial<CoeffType>::Create(std::move(m_s_coeffs)));
  } else {
    ASSIGN_OR_RETURN(
        auto ctx,
        Context::CreateForTernary(
            absl::bit_width(static_cast<uint32_t>(d)) - 1));
    ASSIGN_OR_RETURN(auto s_ntt, secret_key.ToNtt(ctx, /*is_ternary=*/true));
    ASSIGN_OR_RETURN(auto m_poly, Polynomial<CoeffType>::Create(message));
    ASSIGN_OR_RETURN(m_s, m_poly.Mult(s_ntt, ctx));
  }

  ASSIGN_OR_RETURN(auto u, EncryptGadgetCiphertextFromRlweSamples(
                               params, gadget_params, u_samples, m_s.Coeffs()));
  ASSIGN_OR_RETURN(auto v, EncryptGadgetCiphertextFromRlweSamples(
                               params, gadget_params, v_samples, message));

  return RgswCiphertext<CoeffType>{std::move(u), std::move(v)};
}

template absl::StatusOr<RlweParams<uint32_t>> RlweParams<uint32_t>::Create(
    int, uint32_t, int);
template absl::StatusOr<RlweParams<uint64_t>> RlweParams<uint64_t>::Create(
    int, uint64_t, int);

template absl::StatusOr<RlweParams<uint32_t>> RlweParams<uint32_t>::Create(
    int, uint32_t, uint32_t, uint32_t, int);
template absl::StatusOr<RlweParams<uint64_t>> RlweParams<uint64_t>::Create(
    int, uint64_t, uint64_t, uint64_t, int);

template absl::StatusOr<Polynomial<uint32_t>> SampleSecretKey<uint32_t>(
    const RlweParams<uint32_t>&, ::rlwe::SecurePrng&);
template absl::StatusOr<Polynomial<uint64_t>> SampleSecretKey<uint64_t>(
    const RlweParams<uint64_t>&, ::rlwe::SecurePrng&);

template absl::StatusOr<std::vector<Polynomial<uint32_t>>>
SampleAComponents<uint32_t>(const RlweParams<uint32_t>&, int,
                            ::rlwe::SecurePrng&);
template absl::StatusOr<std::vector<Polynomial<uint64_t>>>
SampleAComponents<uint64_t>(const RlweParams<uint64_t>&, int,
                            ::rlwe::SecurePrng&);

template absl::StatusOr<std::vector<RlweSample<uint32_t>>>
GenerateRlweSamples<uint32_t>(const RlweParams<uint32_t>&,
                              const Polynomial<uint32_t>&,
                              const std::vector<Polynomial<uint32_t>>&,
                              ::rlwe::SecurePrng&);
template absl::StatusOr<std::vector<RlweSample<uint64_t>>>
GenerateRlweSamples<uint64_t>(const RlweParams<uint64_t>&,
                              const Polynomial<uint64_t>&,
                              const std::vector<Polynomial<uint64_t>>&,
                              ::rlwe::SecurePrng&);
template absl::StatusOr<std::vector<RlweSample<uint32_t>>>
GenerateRlweSamples<uint32_t>(const RlweParams<uint32_t>&,
                              const NttPolynomial&,
                              const std::vector<Polynomial<uint32_t>>&,
                              ::rlwe::SecurePrng&,
                              const Context&);
template absl::StatusOr<std::vector<RlweSample<uint64_t>>>
GenerateRlweSamples<uint64_t>(const RlweParams<uint64_t>&,
                              const NttPolynomial&,
                              const std::vector<Polynomial<uint64_t>>&,
                              ::rlwe::SecurePrng&,
                              const Context&);

template absl::StatusOr<RlweCiphertext<uint32_t>>
EncryptFromRlweSample<uint32_t>(const RlweParams<uint32_t>&, uint32_t,
                                const RlweSample<uint32_t>&,
                                const std::vector<uint32_t>&);
template absl::StatusOr<RlweCiphertext<uint64_t>>
EncryptFromRlweSample<uint64_t>(const RlweParams<uint64_t>&, uint64_t,
                                const RlweSample<uint64_t>&,
                                const std::vector<uint64_t>&);

template absl::StatusOr<RlweCiphertext<uint32_t>> EncryptRlwe<uint32_t>(
    const RlweParams<uint32_t>&, uint32_t, const Polynomial<uint32_t>&,
    const std::vector<uint32_t>&, ::rlwe::SecurePrng&);
template absl::StatusOr<RlweCiphertext<uint64_t>> EncryptRlwe<uint64_t>(
    const RlweParams<uint64_t>&, uint64_t, const Polynomial<uint64_t>&,
    const std::vector<uint64_t>&, ::rlwe::SecurePrng&);

template absl::StatusOr<std::vector<uint32_t>> Decrypt<uint32_t>(
    const RlweParams<uint32_t>&, uint32_t, const RlweCiphertext<uint32_t>&,
    const Polynomial<uint32_t>&);
template absl::StatusOr<std::vector<uint64_t>> Decrypt<uint64_t>(
    const RlweParams<uint64_t>&, uint64_t, const RlweCiphertext<uint64_t>&,
    const Polynomial<uint64_t>&);
template absl::StatusOr<std::vector<uint32_t>> Decrypt<uint32_t>(
    const RlweParams<uint32_t>&, uint32_t, const RlweCiphertext<uint32_t>&,
    const NttPolynomial&, const Context&);
template absl::StatusOr<std::vector<uint64_t>> Decrypt<uint64_t>(
    const RlweParams<uint64_t>&, uint64_t, const RlweCiphertext<uint64_t>&,
    const NttPolynomial&, const Context&);

template absl::StatusOr<std::vector<uint32_t>>
DecryptAfterModulusSwitch<uint32_t>(const RlweParams<uint32_t>&, uint32_t,
                                    const RlweCiphertext<uint32_t>&,
                                    const Polynomial<uint32_t>&);
template absl::StatusOr<std::vector<uint64_t>>
DecryptAfterModulusSwitch<uint64_t>(const RlweParams<uint64_t>&, uint64_t,
                                    const RlweCiphertext<uint64_t>&,
                                    const Polynomial<uint64_t>&);
template absl::StatusOr<std::vector<uint32_t>>
DecryptAfterModulusSwitch<uint32_t>(const RlweParams<uint32_t>&, uint32_t,
                                    const RlweCiphertext<uint32_t>&,
                                    const NttPolynomial&, const Context&);
template absl::StatusOr<std::vector<uint64_t>>
DecryptAfterModulusSwitch<uint64_t>(const RlweParams<uint64_t>&, uint64_t,
                                    const RlweCiphertext<uint64_t>&,
                                    const NttPolynomial&, const Context&);

template absl::StatusOr<std::vector<RlweCiphertext<uint32_t>>>
EncryptGadgetCiphertextFromRlweSamples<uint32_t>(
    const RlweParams<uint32_t>&, const GadgetParams&,
    const std::vector<RlweSample<uint32_t>>&, const std::vector<uint32_t>&);
template absl::StatusOr<std::vector<RlweCiphertext<uint64_t>>>
EncryptGadgetCiphertextFromRlweSamples<uint64_t>(
    const RlweParams<uint64_t>&, const GadgetParams&,
    const std::vector<RlweSample<uint64_t>>&, const std::vector<uint64_t>&);

template absl::StatusOr<RgswCiphertext<uint32_t>>
EncryptRgswCiphertextFromRlweSamples<uint32_t>(
    const RlweParams<uint32_t>&, const GadgetParams&,
    const std::vector<RlweSample<uint32_t>>&,
    const std::vector<RlweSample<uint32_t>>&, const Polynomial<uint32_t>&,
    const std::vector<uint32_t>&);
template absl::StatusOr<RgswCiphertext<uint64_t>>
EncryptRgswCiphertextFromRlweSamples<uint64_t>(
    const RlweParams<uint64_t>&, const GadgetParams&,
    const std::vector<RlweSample<uint64_t>>&,
    const std::vector<RlweSample<uint64_t>>&, const Polynomial<uint64_t>&,
    const std::vector<uint64_t>&);

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
