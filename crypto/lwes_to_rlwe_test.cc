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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/random/random.h"
#include "absl/status/statusor.h"
#include "crypto/status_macros.h"
#include "shell_encryption/prng/prng.h"
#include "shell_encryption/prng/single_thread_chacha_prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using ::testing::Test;
using ::testing::Types;

template <typename CoeffType>
class LwestoRlweTest : public Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(6));
    ctx_ = std::make_unique<Context>(std::move(ctx));
    gadget_params_ = {.log_digit = 8,
                      .num_digits = static_cast<int>(sizeof(CoeffType))};
    variance_ = 8;
    ASSERT_OK_AND_ASSIGN(auto seed,
                         ::rlwe::SingleThreadChaChaPrng::GenerateSeed());
    ASSERT_OK_AND_ASSIGN(prng_, ::rlwe::SingleThreadChaChaPrng::Create(seed));
    ASSERT_OK_AND_ASSIGN(fft_ctx_, FftContext::Create(6));
  }
  // scheme.

  CoeffType GetMask() const {
    if (log_q_ >= sizeof(CoeffType) * 8) return ~static_cast<CoeffType>(0);
    return (static_cast<CoeffType>(1) << log_q_) - 1;
  }

  absl::StatusOr<Polynomial<CoeffType>> SampleSmall(int bound) {
    std::vector<CoeffType> coeffs(d_);
    CoeffType mask = this->GetMask();
    for (int i = 0; i < d_; ++i) {
      int val = absl::Uniform<int>(absl::IntervalClosedClosed, this->bitgen_,
                                   -bound, bound);
      coeffs[i] = static_cast<CoeffType>(val) & mask;
    }
    return Polynomial<CoeffType>::Create(coeffs);
  }

  absl::StatusOr<std::pair<std::vector<Polynomial<CoeffType>>,
                           std::vector<Polynomial<CoeffType>>>>
  EncryptKeySwitchingKey(const RlweParams<CoeffType>& params,
                         const Polynomial<CoeffType>& s,
                         const Polynomial<CoeffType>& s_target) {
    ASSIGN_OR_RETURN(
        auto w_vec,
        SampleAComponents(params, gadget_params_.num_digits, *prng_));
    std::vector<Polynomial<CoeffType>> y_vec(gadget_params_.num_digits);

    int offset_bits = params.LogModulus() -
                      (gadget_params_.num_digits * gadget_params_.log_digit);
    if (offset_bits < 0) {
      offset_bits = 0;
    }

    for (int i = 0; i < gadget_params_.num_digits; ++i) {
      ASSIGN_OR_RETURN(auto e, SampleSmall(2));

      std::vector<CoeffType> target_coeffs(d_);
      for (int j = 0; j < d_; ++j) {
        target_coeffs[j] = s_target.Coeffs()[j]
                           << (i * gadget_params_.log_digit + offset_bits);
      }
      ASSIGN_OR_RETURN(auto target_poly,
                       Polynomial<CoeffType>::Create(target_coeffs));

      ASSIGN_OR_RETURN(auto ws, w_vec[i].Mult(s, *ctx_));
      ASSIGN_OR_RETURN(auto minus_ws, ws.Negate());

      ASSIGN_OR_RETURN(auto y, minus_ws.Add(e));
      ASSIGN_OR_RETURN(y_vec[i], y.Add(target_poly));
    }

    return std::make_pair(w_vec, y_vec);
  }

  absl::StatusOr<
      std::pair<std::vector<std::vector<CoeffType>>, std::vector<CoeffType>>>
  ExtractLwes(const RlweCiphertext<CoeffType>& ciphertext) {
    const Polynomial<CoeffType>& a = ciphertext.a;
    const Polynomial<CoeffType>& b = ciphertext.b;

    std::vector<std::vector<CoeffType>> a_is(d_, std::vector<CoeffType>(d_));
    std::vector<CoeffType> b_is = b.Coeffs();
    for (int k = 0; k < d_; ++k) {
      for (int j = 0; j < d_; ++j) {
        if (j <= k) {
          a_is[k][j] = a.Coeffs()[k - j];
        } else {
          a_is[k][j] = -a.Coeffs()[d_ + k - j];
        }
      }
    }
    return std::make_pair(a_is, b_is);
  }

  int d_ = 64;  // Small d for testing to avoid timeout
  std::unique_ptr<Context> ctx_;
  std::unique_ptr<FftContext> fft_ctx_;
  GadgetParams gadget_params_;
  absl::BitGen bitgen_;

  int variance_ = 8;
  std::unique_ptr<::rlwe::SecurePrng> prng_;

  int log_q_ =
      static_cast<int>(sizeof(CoeffType) * 8);  // Default log_q initially
};

using CoeffTypes = Types<uint32_t, uint64_t>;
TYPED_TEST_SUITE(LwestoRlweTest, CoeffTypes);

TYPED_TEST(LwestoRlweTest, PackEndToEndWorks) {
  std::vector<int> log_qs;
  if (sizeof(TypeParam) == 4) {
    log_qs = {27, 29, 32};
  } else {
    log_qs = {45, 54, 64};
  }

  for (int test_log_q : log_qs) {
    this->log_q_ = test_log_q;
    this->gadget_params_.log_digit = 4;
    this->gadget_params_.num_digits =
        (test_log_q + this->gadget_params_.log_digit - 1) /
        this->gadget_params_.log_digit;

    // plaintext modulus
    int t = 8;

    TypeParam mask = this->GetMask();
    TypeParam delta;
    if (test_log_q == sizeof(TypeParam) * 8) {
      delta = (mask / t) + 1;
    } else {
      delta = (mask + 1) / t;
    }

    TypeParam modulus = mask + 1;
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(
                                          this->d_, modulus, this->variance_));

    // 1. Generate secret key
    ASSERT_OK_AND_ASSIGN(auto s, SampleSecretKey(params, *this->prng_));

    // 2. Encrypt Key Switching Keys
    ASSERT_OK_AND_ASSIGN(auto s_g, s.Automorph(5));
    ASSERT_OK_AND_ASSIGN(auto ksk_g,
                         this->EncryptKeySwitchingKey(params, s, s_g));
    auto w_vec_g = ksk_g.first;
    auto y_vec_g = ksk_g.second;

    ASSERT_OK_AND_ASSIGN(auto s_h, s.Automorph(2 * this->d_ - 1));
    ASSERT_OK_AND_ASSIGN(auto ksk_h,
                         this->EncryptKeySwitchingKey(params, s, s_h));
    auto w_vec_h = ksk_h.first;
    auto y_vec_h = ksk_h.second;

    // 3. Generate random message {0, 1}^d
    std::vector<TypeParam> m_coeffs(this->d_);
    for (int i = 0; i < this->d_; ++i) {
      m_coeffs[i] = absl::Uniform<int>(this->bitgen_, 0, t);
    }
    ASSERT_OK_AND_ASSIGN(auto m, Polynomial<TypeParam>::Create(m_coeffs));

    // 4. Encrypt RLWE and extract LWEs
    ASSERT_OK_AND_ASSIGN(auto rlwe_ct,
                         EncryptRlwe(params, static_cast<TypeParam>(t), s,
                                     m.Coeffs(), *this->prng_));
    ASSERT_OK_AND_ASSIGN(auto extracted, this->ExtractLwes(rlwe_ct));
    auto a_is = extracted.first;
    auto b_is = extracted.second;

    // Ensure that extracted LWE ciphertexts decrypt securely
    for (int k = 0; k < this->d_; ++k) {
      TypeParam val = b_is[k];
      for (int j = 0; j < this->d_; ++j) {
        val += a_is[k][j] * s.Coeffs()[j];  // a*s + b = e + m*delta
      }
      val &= mask;
      TypeParam m_k_extracted = 0;
      TypeParam min_dist = static_cast<TypeParam>(-1);
      for (int msg = 0; msg < t; ++msg) {
        // calculate distance to scaled message
        TypeParam scaled_msg = (static_cast<TypeParam>(msg) * delta) & mask;
        TypeParam dist =
            (val >= scaled_msg) ? (val - scaled_msg) : (scaled_msg - val);
        dist &= mask;
        if (dist > (mask >> 1)) {
          dist = mask - dist + 1;
        }
        if (dist < min_dist) {
          min_dist = dist;
          m_k_extracted = static_cast<TypeParam>(msg);
        }
      }
      EXPECT_EQ(m_k_extracted, m_coeffs[k])
          << "Failed LWE extraction at log_q=" << test_log_q << " index=" << k;
    }

    // 5. PreprocessPack and Pack
    ASSERT_OK_AND_ASSIGN(
        auto preprocess_output,
        (PreprocessPack<TypeParam>(params, a_is, w_vec_g, w_vec_h,
                                   this->gadget_params_, *this->fft_ctx_)));

    ASSERT_OK_AND_ASSIGN(
        auto packed,
        Pack<TypeParam>(params, b_is, y_vec_g, y_vec_h, preprocess_output,
                        this->gadget_params_, *this->fft_ctx_));

    // 6. Decrypt packed RLWE
    ASSERT_OK_AND_ASSIGN(
        auto m_dec,
        Decrypt<TypeParam>(params, static_cast<TypeParam>(t), packed, s));

    EXPECT_EQ(m_dec, m.Coeffs()) << "Failed at log_q=" << test_log_q;
  }
}

TYPED_TEST(LwestoRlweTest, PackEndToEndWithApproximateGadgetDecompWorks) {
  std::vector<int> log_qs;
  if (sizeof(TypeParam) == 4) {
    log_qs = {27, 29, 32};
  } else {
    log_qs = {45, 54, 64};
  }

  for (int test_log_q : log_qs) {
    this->log_q_ = test_log_q;
    this->gadget_params_.log_digit = 4;
    this->gadget_params_.num_digits = 4;

    // plaintext modulus
    int t = 8;

    TypeParam mask = this->GetMask();
    TypeParam delta;
    if (test_log_q == sizeof(TypeParam) * 8) {
      delta = (mask / t) + 1;
    } else {
      delta = (mask + 1) / t;
    }

    TypeParam modulus = mask + 1;
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(
                                          this->d_, modulus, this->variance_));

    // 1. Generate secret key
    ASSERT_OK_AND_ASSIGN(auto s, SampleSecretKey(params, *this->prng_));

    // 2. Encrypt Key Switching Keys
    ASSERT_OK_AND_ASSIGN(auto s_g, s.Automorph(5));
    ASSERT_OK_AND_ASSIGN(auto ksk_g,
                         this->EncryptKeySwitchingKey(params, s, s_g));
    auto w_vec_g = ksk_g.first;
    auto y_vec_g = ksk_g.second;

    ASSERT_OK_AND_ASSIGN(auto s_h, s.Automorph(2 * this->d_ - 1));
    ASSERT_OK_AND_ASSIGN(auto ksk_h,
                         this->EncryptKeySwitchingKey(params, s, s_h));
    auto w_vec_h = ksk_h.first;
    auto y_vec_h = ksk_h.second;

    // 3. Generate random message {0, 1}^d
    std::vector<TypeParam> m_coeffs(this->d_);
    for (int i = 0; i < this->d_; ++i) {
      m_coeffs[i] = absl::Uniform<int>(this->bitgen_, 0, t);
    }
    ASSERT_OK_AND_ASSIGN(auto m, Polynomial<TypeParam>::Create(m_coeffs));

    // 4. Encrypt RLWE and extract LWEs
    ASSERT_OK_AND_ASSIGN(auto rlwe_ct,
                         EncryptRlwe(params, static_cast<TypeParam>(t), s,
                                     m.Coeffs(), *this->prng_));
    ASSERT_OK_AND_ASSIGN(auto extracted, this->ExtractLwes(rlwe_ct));
    auto a_is = extracted.first;
    auto b_is = extracted.second;

    // Ensure that extracted LWE ciphertexts decrypt securely
    for (int k = 0; k < this->d_; ++k) {
      TypeParam val = b_is[k];
      for (int j = 0; j < this->d_; ++j) {
        val += a_is[k][j] * s.Coeffs()[j];  // a*s + b = e + m*delta
      }
      val &= mask;
      TypeParam m_k_extracted = 0;
      TypeParam min_dist = static_cast<TypeParam>(-1);
      for (int msg = 0; msg < t; ++msg) {
        // calculate distance to scaled message
        TypeParam scaled_msg = (static_cast<TypeParam>(msg) * delta) & mask;
        TypeParam dist =
            (val >= scaled_msg) ? (val - scaled_msg) : (scaled_msg - val);
        dist &= mask;
        if (dist > (mask >> 1)) {
          dist = mask - dist + 1;
        }
        if (dist < min_dist) {
          min_dist = dist;
          m_k_extracted = static_cast<TypeParam>(msg);
        }
      }
      EXPECT_EQ(m_k_extracted, m_coeffs[k])
          << "Failed LWE extraction at log_q=" << test_log_q << " index=" << k;
    }

    // 5. PreprocessPack and Pack
    ASSERT_OK_AND_ASSIGN(
        auto preprocess_output,
        (PreprocessPack<TypeParam>(params, a_is, w_vec_g, w_vec_h,
                                   this->gadget_params_, *this->fft_ctx_)));

    ASSERT_OK_AND_ASSIGN(
        auto packed,
        Pack<TypeParam>(params, b_is, y_vec_g, y_vec_h, preprocess_output,
                        this->gadget_params_, *this->fft_ctx_));

    // 6. Decrypt packed RLWE
    ASSERT_OK_AND_ASSIGN(
        auto m_dec,
        Decrypt<TypeParam>(params, static_cast<TypeParam>(t), packed, s));

    EXPECT_EQ(m_dec, m.Coeffs()) << "Failed at log_q=" << test_log_q;
  }
}

TYPED_TEST(LwestoRlweTest, PackWithMatrixEndToEndWorks) {
  std::vector<int> log_qs;
  if (sizeof(TypeParam) == 4) {
    log_qs = {27, 29, 32};
  } else {
    log_qs = {45, 54, 64};
  }

  for (int test_log_q : log_qs) {
    this->log_q_ = test_log_q;
    this->gadget_params_.log_digit = 4;
    int max_num_digits = (test_log_q + this->gadget_params_.log_digit - 1) /
                         this->gadget_params_.log_digit;
    this->gadget_params_.num_digits = max_num_digits - 2;

    int t = 8;

    TypeParam mask = this->GetMask();
    TypeParam delta;
    if (test_log_q == sizeof(TypeParam) * 8) {
      delta = (mask / t) + 1;
    } else {
      delta = (mask + 1) / t;
    }

    TypeParam modulus = mask + 1;
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(
                                          this->d_, modulus, this->variance_));

    // 1. Generate secret key
    ASSERT_OK_AND_ASSIGN(auto s, SampleSecretKey(params, *this->prng_));

    // 2. Encrypt Key Switching Keys
    ASSERT_OK_AND_ASSIGN(auto s_g, s.Automorph(5));
    ASSERT_OK_AND_ASSIGN(auto ksk_g,
                         this->EncryptKeySwitchingKey(params, s, s_g));
    auto w_vec_g = ksk_g.first;
    auto y_vec_g = ksk_g.second;

    ASSERT_OK_AND_ASSIGN(auto s_h, s.Automorph(2 * this->d_ - 1));
    ASSERT_OK_AND_ASSIGN(auto ksk_h,
                         this->EncryptKeySwitchingKey(params, s, s_h));
    auto w_vec_h = ksk_h.first;
    auto y_vec_h = ksk_h.second;

    // 3. Generate random message {0, 1}^d
    std::vector<TypeParam> m_coeffs(this->d_);
    for (int i = 0; i < this->d_; ++i) {
      m_coeffs[i] = absl::Uniform<int>(this->bitgen_, 0, t);
    }
    ASSERT_OK_AND_ASSIGN(auto m, Polynomial<TypeParam>::Create(m_coeffs));

    // 4. Encrypt RLWE and extract LWEs
    ASSERT_OK_AND_ASSIGN(auto rlwe_ct,
                         EncryptRlwe(params, static_cast<TypeParam>(t), s,
                                     m.Coeffs(), *this->prng_));
    ASSERT_OK_AND_ASSIGN(auto extracted, this->ExtractLwes(rlwe_ct));
    auto a_is = extracted.first;
    auto b_is = extracted.second;

    // 5. PreprocessPackWithMatrix and PackWithMatrix
    ASSERT_OK_AND_ASSIGN(auto preprocess_output,
                         (PreprocessMatrixPack<TypeParam>(
                             params, a_is, w_vec_g, w_vec_h,
                             this->gadget_params_, *this->fft_ctx_)));

    int d = b_is.size();
    int k = y_vec_g.size();
    std::vector<TypeParam> y_vec_g_vec(k * d);
    for (int l = 0; l < k; ++l) {
      for (int m = 0; m < d; ++m) {
        y_vec_g_vec[l * d + m] = y_vec_g[l].Coeffs()[m];
      }
    }
    ASSERT_OK_AND_ASSIGN(
        std::vector<TypeParam> b_agg_partial,
        preprocess_output.matrix.template Multiply<TypeParam>(y_vec_g_vec));

    ASSERT_OK_AND_ASSIGN(
        auto packed_precomputed,
        FinalizeMatrixPack<TypeParam>(
            params, b_is, b_agg_partial, y_vec_h, preprocess_output.t_vec_h,
            preprocess_output.a_tilde_agg, this->gadget_params_,
            *this->fft_ctx_));

    ASSERT_OK_AND_ASSIGN(
        auto packed_direct,
        (MatrixPack<TypeParam, TypeParam>(
            params, b_is, y_vec_g, y_vec_h, preprocess_output,
            this->gadget_params_, *this->fft_ctx_)));

    // Verify both yield same result
    EXPECT_EQ(packed_precomputed.a.Coeffs(), packed_direct.a.Coeffs());
    EXPECT_EQ(packed_precomputed.b.Coeffs(), packed_direct.b.Coeffs());

    auto packed = packed_direct;

    // 6. Decrypt packed RLWE
    ASSERT_OK_AND_ASSIGN(
        auto m_dec,
        Decrypt<TypeParam>(params, static_cast<TypeParam>(t), packed, s));

    EXPECT_EQ(m_dec, m.Coeffs()) << "Failed at log_q=" << test_log_q;
  }
}

TYPED_TEST(LwestoRlweTest, PreprocessMatrixNormBounded) {
  std::vector<int> log_qs;
  if (sizeof(TypeParam) == 4) {
    log_qs = {27, 29, 32};
  } else {
    log_qs = {45, 54, 64};
  }

  for (int test_log_q : log_qs) {
    this->log_q_ = test_log_q;
    this->gadget_params_.log_digit = 4;
    this->gadget_params_.num_digits =
        (test_log_q + this->gadget_params_.log_digit - 1) /
        this->gadget_params_.log_digit;

    TypeParam mask = this->GetMask();
    TypeParam modulus = mask + 1;
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(
                                          this->d_, modulus, this->variance_));

    ASSERT_OK_AND_ASSIGN(auto s, SampleSecretKey(params, *this->prng_));

    ASSERT_OK_AND_ASSIGN(auto s_g, s.Automorph(5));
    ASSERT_OK_AND_ASSIGN(auto ksk_g,
                         this->EncryptKeySwitchingKey(params, s, s_g));
    auto w_vec_g = ksk_g.first;

    ASSERT_OK_AND_ASSIGN(auto s_h, s.Automorph(2 * this->d_ - 1));
    ASSERT_OK_AND_ASSIGN(auto ksk_h,
                         this->EncryptKeySwitchingKey(params, s, s_h));
    auto w_vec_h = ksk_h.first;

    std::vector<std::vector<TypeParam>> a_is(
        this->d_, std::vector<TypeParam>(this->d_, 0));
    for (int i = 0; i < this->d_; ++i) {
      for (int j = 0; j < this->d_; ++j) {
        a_is[i][j] = absl::Uniform<TypeParam>(this->bitgen_, 0, mask);
      }
    }

    ASSERT_OK_AND_ASSIGN(auto preprocess_output,
                         (PreprocessMatrixPack<TypeParam, int32_t>(
                             params, a_is, w_vec_g, w_vec_h,
                             this->gadget_params_, *this->fft_ctx_)));

    int32_t max_val = 0;
    for (int32_t val : preprocess_output.matrix.Data()) {
      max_val = std::max(max_val, std::abs(val));
    }

    double empirical_bound = max_val;
    double expected_bound =
        4.0 * std::sqrt(this->d_) * (1 << this->gadget_params_.log_digit);
    EXPECT_LT(empirical_bound, expected_bound)
        << "Failed at log_q=" << test_log_q;
  }
}

TYPED_TEST(LwestoRlweTest,
           PackWithMatrixEndToEndWithDifferentMatCoeffTypeWorks) {
  std::vector<int> log_qs;
  if (sizeof(TypeParam) == 4) {
    log_qs = {27};
  } else {
    log_qs = {45};
  }

  for (int test_log_q : log_qs) {
    this->log_q_ = test_log_q;
    this->gadget_params_.log_digit = 4;
    int max_num_digits = (test_log_q + this->gadget_params_.log_digit - 1) /
                         this->gadget_params_.log_digit;
    this->gadget_params_.num_digits = max_num_digits - 2;

    int t = 8;

    TypeParam mask = this->GetMask();
    TypeParam modulus = mask + 1;
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(
                                          this->d_, modulus, this->variance_));

    ASSERT_OK_AND_ASSIGN(auto s, SampleSecretKey(params, *this->prng_));

    ASSERT_OK_AND_ASSIGN(auto s_g, s.Automorph(5));
    ASSERT_OK_AND_ASSIGN(auto ksk_g,
                         this->EncryptKeySwitchingKey(params, s, s_g));
    auto w_vec_g = ksk_g.first;
    auto y_vec_g = ksk_g.second;

    ASSERT_OK_AND_ASSIGN(auto s_h, s.Automorph(2 * this->d_ - 1));
    ASSERT_OK_AND_ASSIGN(auto ksk_h,
                         this->EncryptKeySwitchingKey(params, s, s_h));
    auto w_vec_h = ksk_h.first;
    auto y_vec_h = ksk_h.second;

    std::vector<TypeParam> m_coeffs(this->d_);
    for (int i = 0; i < this->d_; ++i) {
      m_coeffs[i] = absl::Uniform<int>(this->bitgen_, 0, t);
    }
    ASSERT_OK_AND_ASSIGN(auto m, Polynomial<TypeParam>::Create(m_coeffs));

    ASSERT_OK_AND_ASSIGN(auto rlwe_ct,
                         EncryptRlwe(params, static_cast<TypeParam>(t), s,
                                     m.Coeffs(), *this->prng_));
    ASSERT_OK_AND_ASSIGN(auto extracted, this->ExtractLwes(rlwe_ct));
    auto a_is = extracted.first;
    auto b_is = extracted.second;

    using MatCoeffType = int16_t;

    ASSERT_OK_AND_ASSIGN(auto preprocess_output,
                         (PreprocessMatrixPack<TypeParam, MatCoeffType>(
                             params, a_is, w_vec_g, w_vec_h,
                             this->gadget_params_, *this->fft_ctx_)));

    int d = b_is.size();
    int k = y_vec_g.size();
    std::vector<TypeParam> y_vec_g_vec(k * d);
    for (int l = 0; l < k; ++l) {
      for (int m = 0; m < d; ++m) {
        y_vec_g_vec[l * d + m] = y_vec_g[l].Coeffs()[m];
      }
    }
    ASSERT_OK_AND_ASSIGN(
        std::vector<TypeParam> b_agg_partial,
        preprocess_output.matrix.template Multiply<TypeParam>(y_vec_g_vec));

    ASSERT_OK_AND_ASSIGN(
        auto packed_precomputed,
        FinalizeMatrixPack<TypeParam>(
            params, b_is, b_agg_partial, y_vec_h, preprocess_output.t_vec_h,
            preprocess_output.a_tilde_agg, this->gadget_params_,
            *this->fft_ctx_));

    ASSERT_OK_AND_ASSIGN(
        auto packed_direct,
        (MatrixPack<TypeParam, MatCoeffType>(
            params, b_is, y_vec_g, y_vec_h, preprocess_output,
            this->gadget_params_, *this->fft_ctx_)));

    // Verify both yield same result
    EXPECT_EQ(packed_precomputed.a.Coeffs(), packed_direct.a.Coeffs());
    EXPECT_EQ(packed_precomputed.b.Coeffs(), packed_direct.b.Coeffs());

    auto packed = packed_direct;

    ASSERT_OK_AND_ASSIGN(
        auto m_dec,
        Decrypt<TypeParam>(params, static_cast<TypeParam>(t), packed, s));

    EXPECT_EQ(m_dec, m.Coeffs())
        << "Failed with int32_t MatCoeffType at log_q=" << test_log_q;
  }
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
