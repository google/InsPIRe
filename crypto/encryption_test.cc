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

#include <cstdint>
#include <vector>

#include "crypto/external_product.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "shell_encryption/integral_types.h"
#include "shell_encryption/prng/prng.h"
#include "shell_encryption/prng/single_thread_chacha_prng.h"
#include "shell_encryption/testing/status_testing.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using ::rlwe::testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

template <typename T>
class EncryptionTest : public ::testing::Test {};

using CoeffTypes = ::testing::Types<uint32_t, uint64_t>;
TYPED_TEST_SUITE(EncryptionTest, CoeffTypes);

// An insecure PRNG that returns a predictable sequence to allow testing
// behaviors of sampler implementations.
class FakePrng : public ::rlwe::SecurePrng {
 public:
  FakePrng() : counter_(0) {}

  absl::StatusOr<::rlwe::Uint8> Rand8() override {
    ::rlwe::Uint8 val = ++counter_;
    return static_cast<::rlwe::Uint8>(val * 4321321ULL);
  }

  absl::StatusOr<::rlwe::Uint64> Rand64() override {
    ::rlwe::Uint64 val = ++counter_;
    return val * 123456789101112ULL;
  }

 private:
  uint64_t counter_;
  std::vector<::rlwe::Uint8> rand8_seq_;
  std::vector<::rlwe::Uint64> rand64_seq_;
};

TYPED_TEST(EncryptionTest, RlweParamsCreateFailsIfDegreeNotPowerOf2) {
  EXPECT_THAT(RlweParams<TypeParam>::Create(-1, 1024, 8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Ring degree must be a power of 2")));
  EXPECT_THAT(RlweParams<TypeParam>::Create(0, 1024, 8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Ring degree must be a power of 2")));
  EXPECT_THAT(RlweParams<TypeParam>::Create(3, 1024, 8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Ring degree must be a power of 2")));
}

TYPED_TEST(EncryptionTest, RlweParamsCreateFailsIfModulusNotPowerOf2) {
  EXPECT_THAT(RlweParams<TypeParam>::Create(4, 3, 8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("power of 2 or 0")));
  EXPECT_THAT(RlweParams<TypeParam>::Create(4, 1023, 8),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("power of 2 or 0")));
}

TYPED_TEST(EncryptionTest, RlweParamsCreateFailsIfVarianceNotPositive) {
  EXPECT_THAT(RlweParams<TypeParam>::Create(4, 1024, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("variance must be positive")));
  EXPECT_THAT(RlweParams<TypeParam>::Create(4, 1024, -1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("variance must be positive")));
}

TYPED_TEST(EncryptionTest, SampleAComponentsFailsIfNumComponentsNotPositive) {
  FakePrng prng;
  ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, 1024, 8));
  EXPECT_THAT(SampleAComponents<TypeParam>(params, 0, prng),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("num_a_components must be positive")));
  EXPECT_THAT(SampleAComponents<TypeParam>(params, -1, prng),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("num_a_components must be positive")));
}

TYPED_TEST(EncryptionTest, GenerateRlweSamplesFailsIfSecretKeyLengthMismatch) {
  FakePrng prng;
  std::vector<TypeParam> coeffs(2, 0);
  ASSERT_OK_AND_ASSIGN(auto s, Polynomial<TypeParam>::Create(coeffs));
  std::vector<Polynomial<TypeParam>> a_components;
  ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, 1024, 8));
  EXPECT_THAT(GenerateRlweSamples<TypeParam>(params, s, a_components, prng),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("secret_key length must match degree")));
}

TYPED_TEST(EncryptionTest, GenerateRlweSamplesFailsIfAComponentLengthMismatch) {
  FakePrng prng;
  std::vector<TypeParam> s_coeffs(4, 0);
  ASSERT_OK_AND_ASSIGN(auto s, Polynomial<TypeParam>::Create(s_coeffs));

  std::vector<TypeParam> a_coeffs(2, 0);
  ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
  std::vector<Polynomial<TypeParam>> a_components = {a};

  ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, 1024, 8));
  EXPECT_THAT(GenerateRlweSamples<TypeParam>(params, s, a_components, prng),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("a_component length must match degree")));
}

TYPED_TEST(EncryptionTest, SampleSecretKeySucceeds) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    FakePrng prng;
    const int degree = 4;
    ASSERT_OK_AND_ASSIGN(auto params,
                         RlweParams<TypeParam>::Create(degree, q, 8));
    ASSERT_OK_AND_ASSIGN(auto s_poly, SampleSecretKey<TypeParam>(params, prng));
    EXPECT_EQ(s_poly.Len(), degree);

    std::vector<int> signed_secret_key;
    for (int i = 0; i < degree; ++i) {
      TypeParam current = s_poly.Coeffs()[i];
      TypeParam mask = (q == 0) ? ~static_cast<TypeParam>(0) : (q - 1);
      signed_secret_key.push_back((current == mask) ? -1 : current);
    }
    EXPECT_THAT(signed_secret_key, ElementsAre(1, -1, 0, 1));
  }
}

TYPED_TEST(EncryptionTest, SampleAComponentsSucceeds) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    FakePrng prng;
    const int degree = 4;
    ASSERT_OK_AND_ASSIGN(auto params,
                         RlweParams<TypeParam>::Create(degree, q, 8));
    ASSERT_OK_AND_ASSIGN(auto a_components,
                         SampleAComponents<TypeParam>(params, 2, prng));

    EXPECT_EQ(a_components.size(), 2);

    FakePrng expected_prng;
    TypeParam mask = (q == 0) ? ~static_cast<TypeParam>(0) : (q - 1);
    for (int i = 0; i < 2; ++i) {
      std::vector<TypeParam> expected_coeffs;
      for (int j = 0; j < degree; ++j) {
        expected_coeffs.push_back(
            static_cast<TypeParam>(expected_prng.Rand64().value()) & mask);
      }
      EXPECT_THAT(a_components[i].Coeffs(),
                  ::testing::ElementsAreArray(expected_coeffs));
    }
  }
}

TYPED_TEST(EncryptionTest, GenerateRlweSamplesSucceeds) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    FakePrng prng;
    const int degree = 4;
    const int variance = 8;
    ASSERT_OK_AND_ASSIGN(auto params,
                         RlweParams<TypeParam>::Create(degree, q, variance));

    // 1. Generate a secret key.
    ASSERT_OK_AND_ASSIGN(auto s, SampleSecretKey<TypeParam>(params, prng));

    // 2. Generate a_components.
    ASSERT_OK_AND_ASSIGN(auto a_components,
                         SampleAComponents<TypeParam>(params, 2, prng));

    ASSERT_OK_AND_ASSIGN(auto result, GenerateRlweSamples<TypeParam>(
                                          params, s, a_components, prng));

    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].a.Len(), degree);
    EXPECT_EQ(result[0].b.Len(), degree);

    // Verify that b + as = e (mod q)  where e is small
    int log_degree = absl::bit_width(static_cast<uint32_t>(degree)) - 1;
    ASSERT_OK_AND_ASSIGN(
        auto ctx, private_membership::rlwe::v2::Context::Create(log_degree));

    TypeParam mask = (q == 0) ? ~static_cast<TypeParam>(0) : (q - 1);
    TypeParam half_q = (q == 0) ? (mask >> 1) + 1 : q / 2;

    std::vector<std::vector<int64_t>> signed_errors_vec;
    for (int i = 0; i < 2; ++i) {
      ASSERT_OK_AND_ASSIGN(auto as, result[i].a.Mult(s, ctx));
      std::vector<int64_t> signed_errors;
      for (int j = 0; j < degree; ++j) {
        TypeParam error = (result[i].b.Coeffs()[j] + as.Coeffs()[j]) & mask;

        // Map error back to [-q/2, q/2) to check its size
        int64_t signed_error;
        if (error > half_q) {
          signed_error = -static_cast<int64_t>((mask - error) + 1);
        } else {
          signed_error = static_cast<int64_t>(error);
        }
        signed_errors.push_back(signed_error);
      }
      signed_errors_vec.push_back(signed_errors);
    }
    EXPECT_THAT(signed_errors_vec,
                ElementsAre(ElementsAre(1, 1, 4, -3), ElementsAre(5, 1, 6, 1)));
  }
}

TYPED_TEST(EncryptionTest,
           EncryptFromRlweSampleFailsIfPlaintextModulusNotPositive) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> b_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto b, Polynomial<TypeParam>::Create(b_coeffs));
    std::vector<TypeParam> m(4, 0);
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));

    EXPECT_THAT(EncryptFromRlweSample<TypeParam>(
                    params, 0, RlweSample<TypeParam>{a, b}, m),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Plaintext modulus must be positive")));
  }
}

TYPED_TEST(EncryptionTest, EncryptFromRlweSampleFailsIfALengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    std::vector<TypeParam> a_coeffs(2, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> b_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto b, Polynomial<TypeParam>::Create(b_coeffs));
    std::vector<TypeParam> m(4, 0);
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));

    EXPECT_THAT(EncryptFromRlweSample<TypeParam>(
                    params, 2, RlweSample<TypeParam>{a, b}, m),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("sample lengths must match degree")));
  }
}

TYPED_TEST(EncryptionTest, EncryptFromRlweSampleFailsIfBLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> b_coeffs(2, 0);
    ASSERT_OK_AND_ASSIGN(auto b, Polynomial<TypeParam>::Create(b_coeffs));
    std::vector<TypeParam> m(4, 0);
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));

    EXPECT_THAT(EncryptFromRlweSample<TypeParam>(
                    params, 2, RlweSample<TypeParam>{a, b}, m),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("sample lengths must match degree")));
  }
}

TYPED_TEST(EncryptionTest, EncryptFromRlweSampleFailsIfMessageLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> b_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto b, Polynomial<TypeParam>::Create(b_coeffs));
    std::vector<TypeParam> m(2, 0);
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));

    EXPECT_THAT(EncryptFromRlweSample<TypeParam>(
                    params, 2, RlweSample<TypeParam>{a, b}, m),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("message length must match degree")));
  }
}

TYPED_TEST(EncryptionTest,
           EncryptFromRlweSampleFailsIfMessageComponentTooLarge) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> b_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto b, Polynomial<TypeParam>::Create(b_coeffs));
    std::vector<TypeParam> m = {0, 1, 2, 0};
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));

    EXPECT_THAT(
        EncryptFromRlweSample<TypeParam>(params, 2, RlweSample<TypeParam>{a, b},
                                         m),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr(
                "Message components must be less than plaintext_modulus")));
  }
}

TYPED_TEST(EncryptionTest, DecryptFailsIfPlaintextModulusNotPositive) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> b_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto b, Polynomial<TypeParam>::Create(b_coeffs));
    std::vector<TypeParam> s_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto s, Polynomial<TypeParam>::Create(s_coeffs));
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    RlweCiphertext<TypeParam> ciphertext = {a, b};

    EXPECT_THAT(Decrypt<TypeParam>(params, 0, ciphertext, s),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Plaintext modulus must be positive")));
  }
}

TYPED_TEST(EncryptionTest, DecryptFailsIfALengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    std::vector<TypeParam> a_coeffs(2, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> b_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto b, Polynomial<TypeParam>::Create(b_coeffs));
    std::vector<TypeParam> s_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto s, Polynomial<TypeParam>::Create(s_coeffs));
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    RlweCiphertext<TypeParam> ciphertext = {a, b};

    EXPECT_THAT(Decrypt<TypeParam>(params, 2, ciphertext, s),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("ciphertext lengths must match degree")));
  }
}

TYPED_TEST(EncryptionTest, DecryptFailsIfBLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> b_coeffs(2, 0);
    ASSERT_OK_AND_ASSIGN(auto b, Polynomial<TypeParam>::Create(b_coeffs));
    std::vector<TypeParam> s_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto s, Polynomial<TypeParam>::Create(s_coeffs));
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    RlweCiphertext<TypeParam> ciphertext = {a, b};

    EXPECT_THAT(Decrypt<TypeParam>(params, 2, ciphertext, s),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("ciphertext lengths must match degree")));
  }
}

TYPED_TEST(EncryptionTest, DecryptFailsIfSecretKeyLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> b_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto b, Polynomial<TypeParam>::Create(b_coeffs));
    std::vector<TypeParam> s_coeffs(2, 0);
    ASSERT_OK_AND_ASSIGN(auto s, Polynomial<TypeParam>::Create(s_coeffs));
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    RlweCiphertext<TypeParam> ciphertext = {a, b};

    EXPECT_THAT(Decrypt<TypeParam>(params, 2, ciphertext, s),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("secret_key length must match degree")));
  }
}

TYPED_TEST(EncryptionTest, EncryptDecryptSucceeds) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  const int d = 32;
  const int p = 9;
  ASSERT_OK_AND_ASSIGN(auto seed,
                       ::rlwe::SingleThreadChaChaPrng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(auto prng, ::rlwe::SingleThreadChaChaPrng::Create(seed));
  for (TypeParam q : moduli) {
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(d, q, 1));
    ASSERT_OK_AND_ASSIGN(auto s, SampleSecretKey<TypeParam>(params, *prng));
    ASSERT_OK_AND_ASSIGN(
        auto a_components,
        SampleAComponents<TypeParam>(params, /*num_a_components=*/1, *prng));
    ASSERT_OK_AND_ASSIGN(
        auto rlwe_samples,
        GenerateRlweSamples<TypeParam>(params, s, a_components, *prng));

    std::vector<TypeParam> message(d);
    for (int k = 0; k < d; ++k) {
      message[k] = k % p;
    }

    ASSERT_OK_AND_ASSIGN(
        auto ciphertext,
        EncryptFromRlweSample<TypeParam>(params, p, rlwe_samples[0], message));
    ASSERT_OK_AND_ASSIGN(auto decrypted_message,
                         Decrypt<TypeParam>(params, p, ciphertext, s));

    EXPECT_EQ(decrypted_message, message);
  }
}

TYPED_TEST(EncryptionTest,
           EncryptGadgetCiphertextFromRlweSamplesFailsIfMessageLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    GadgetParams gadget_params = {2, 5};
    std::vector<TypeParam> message(2, 0);

    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<RlweSample<TypeParam>> rlwe_samples(5, {a, a});

    EXPECT_THAT(EncryptGadgetCiphertextFromRlweSamples<TypeParam>(
                    params, gadget_params, rlwe_samples, message),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("message length must match d.")));
  }
}

TYPED_TEST(
    EncryptionTest,
    EncryptGadgetCiphertextFromRlweSamplesFailsIfRlweSamplesLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    GadgetParams gadget_params = {2, 5};
    std::vector<TypeParam> message(4, 0);

    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<RlweSample<TypeParam>> rlwe_samples(3, {a, a});

    EXPECT_THAT(
        EncryptGadgetCiphertextFromRlweSamples<TypeParam>(
            params, gadget_params, rlwe_samples, message),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("rlwe_samples length must match num_digits.")));
  }
}

TYPED_TEST(EncryptionTest,
           EncryptGadgetCiphertextFromRlweSamplesFailsIfLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    GadgetParams gadget_params = {2, 5};
    std::vector<TypeParam> message(4, 0);

    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> a_short_coeffs(2, 0);
    ASSERT_OK_AND_ASSIGN(auto a_short,
                         Polynomial<TypeParam>::Create(a_short_coeffs));
    std::vector<RlweSample<TypeParam>> rlwe_samples(5, {a, a});
    rlwe_samples[0] = {a, a_short};

    EXPECT_THAT(
        EncryptGadgetCiphertextFromRlweSamples<TypeParam>(
            params, gadget_params, rlwe_samples, message),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("rlwe_samples element length must match d.")));
  }
}

TYPED_TEST(EncryptionTest, EncryptGadgetCiphertextFromRlweSamplesSucceeds) {
  // Tests by performing a key switching operation.
  std::vector<TypeParam> moduli = {
      0, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};

  ASSERT_OK_AND_ASSIGN(auto prng_seed,
                       ::rlwe::SingleThreadChaChaPrng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(auto prng,
                       ::rlwe::SingleThreadChaChaPrng::Create(prng_seed));

  const std::vector<int> degrees = {16, 32, 64, 2048};
  const std::vector<int> plaintext_moduli = {3, 17, 257, 4095};
  const int variance = 8;

  for (TypeParam q : moduli) {
    for (int d : degrees) {
      for (int p : plaintext_moduli) {
        int log_q = (q == 0) ? (sizeof(TypeParam) * 8) : absl::bit_width(q - 1);
        int log_digit = 4;
        int num_digits = (log_q + log_digit - 1) / log_digit;
        GadgetParams gadget_params = {log_digit, num_digits};

        ASSERT_OK_AND_ASSIGN(auto params,
                             RlweParams<TypeParam>::Create(d, q, variance));

        ASSERT_OK_AND_ASSIGN(
            auto ctx,
            Context::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));

        // s_source
        ASSERT_OK_AND_ASSIGN(auto s_source,
                             SampleSecretKey<TypeParam>(params, *prng));
        // s_target
        ASSERT_OK_AND_ASSIGN(auto s_target,
                             SampleSecretKey<TypeParam>(params, *prng));
        std::vector<TypeParam> s_target_vec = s_target.Coeffs();

        // w_vec
        ASSERT_OK_AND_ASSIGN(
            std::vector<Polynomial<TypeParam>> w_vec,
            SampleAComponents<TypeParam>(params, num_digits, *prng));

        // Generate RLWE samples (-a*s + e) for w_vec and s_source
        ASSERT_OK_AND_ASSIGN(
            auto rlwe_samples,
            GenerateRlweSamples<TypeParam>(params, s_source, w_vec, *prng));

        // Generate KeySwitching key under s_source for message
        ASSERT_OK_AND_ASSIGN(
            auto ks_key,
            EncryptGadgetCiphertextFromRlweSamples<TypeParam>(
                params, gadget_params, rlwe_samples, s_target_vec));

        // Generate an arbitrary message
        std::vector<TypeParam> m_coeffs(d);
        for (int k = 0; k < d; ++k) {
          m_coeffs[k] = (k * 987654321ULL) % p;
        }
        ASSERT_OK_AND_ASSIGN(auto m, Polynomial<TypeParam>::Create(m_coeffs));

        // Sample a_components
        std::vector<Polynomial<TypeParam>> a_components;
        std::vector<TypeParam> a_coeffs(d, 1);
        ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
        a_components.push_back(a);

        // Encrypt plaintext under s_target.
        ASSERT_OK_AND_ASSIGN(
            auto samples, GenerateRlweSamples<TypeParam>(params, s_target,
                                                         a_components, *prng));

        // First, encrypt the plaintext under s_target.
        ASSERT_OK_AND_ASSIGN(auto ct, EncryptFromRlweSample<TypeParam>(
                                          params, p, samples[0], m_coeffs));

        // The gadget ciphertext stores (w_i, y_i) where:
        //   y_i = -s * w_i + e + message * B^i
        //
        // Given ciphertext (a, b) encrypting m under message:
        //   b = -a * message + e_ct + m * Delta
        //
        // To decrypt under the original secret key s, we compute:
        //   a' = <w, g^-1(a)>
        //   b' = b + <y, g^-1(a)>
        //
        // This results in a valid ciphertext (a', b') of m under s because:
        //   b' =  b + <-s * w + e + message * B, g^-1(a)>
        //      = -a * message + e_ct + Delta * m - s * a' + <e, g^-1(a)> + a *
        //          message
        //      = -s * a' + Delta * m + e_ct + <e, g^-1(a)>

        auto a_mod = ct.a;
        ASSERT_OK_AND_ASSIGN(a_mod, a_mod.LowBits(log_q));
        ASSERT_OK_AND_ASSIGN(auto a_inv,
                             a_mod.GadgetInv(log_q, log_digit, num_digits));

        // Compute inner products:
        // a_prime = <w, a_inv>
        // term_y = <y, a_inv>
        ASSERT_OK_AND_ASSIGN(auto a_prime, a_inv[0].Mult(ks_key[0].a, ctx));
        ASSERT_OK_AND_ASSIGN(auto term_y, a_inv[0].Mult(ks_key[0].b, ctx));

        for (int i = 1; i < num_digits; ++i) {
          ASSERT_OK_AND_ASSIGN(auto temp_a, a_inv[i].Mult(ks_key[i].a, ctx));
          ASSERT_OK_AND_ASSIGN(a_prime, a_prime.Add(temp_a));

          ASSERT_OK_AND_ASSIGN(auto temp_y, a_inv[i].Mult(ks_key[i].b, ctx));
          ASSERT_OK_AND_ASSIGN(term_y, term_y.Add(temp_y));
        }

        // b' = b + <y, a_inv>
        ASSERT_OK_AND_ASSIGN(auto b_prime, ct.b.Add(term_y));

        // Decrypt under s_source
        ASSERT_OK_AND_ASSIGN(
            auto m_dec,
            Decrypt<TypeParam>(params, p, {a_prime, b_prime}, s_source));
        EXPECT_EQ(m_dec, m_coeffs);
      }
    }
  }
}

TYPED_TEST(EncryptionTest,
           EncryptRgswCiphertextFromRlweSamplesFailsIfSecretKeyLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    GadgetParams gadget_params = {2, 5};
    std::vector<TypeParam> s_coeffs(2, 0);
    ASSERT_OK_AND_ASSIGN(auto s_short, Polynomial<TypeParam>::Create(s_coeffs));

    std::vector<TypeParam> message(4, 0);
    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<RlweSample<TypeParam>> u_samples(5, {a, a});
    std::vector<RlweSample<TypeParam>> v_samples(5, {a, a});

    EXPECT_THAT(
        EncryptRgswCiphertextFromRlweSamples<TypeParam>(
            params, gadget_params, u_samples, v_samples, s_short, message),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("secret_key length must match d.")));
  }
}

TYPED_TEST(EncryptionTest,
           EncryptRgswCiphertextFromRlweSamplesFailsIfMessageLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    GadgetParams gadget_params = {2, 5};
    std::vector<TypeParam> s_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto s, Polynomial<TypeParam>::Create(s_coeffs));

    std::vector<TypeParam> message_short(2, 0);
    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<RlweSample<TypeParam>> u_samples(5, {a, a});
    std::vector<RlweSample<TypeParam>> v_samples(5, {a, a});

    EXPECT_THAT(
        EncryptRgswCiphertextFromRlweSamples<TypeParam>(
            params, gadget_params, u_samples, v_samples, s, message_short),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("message length must match d.")));
  }
}

TYPED_TEST(EncryptionTest, ExternalProductFailsIfCtLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    GadgetParams gadget_params = {2, 5};

    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> a_short_coeffs(2, 0);
    ASSERT_OK_AND_ASSIGN(auto a_short,
                         Polynomial<TypeParam>::Create(a_short_coeffs));

    RlweCiphertext<TypeParam> ct = {a, a_short};
    RgswCiphertext<TypeParam> rgsw_ct = {
        std::vector<RlweCiphertext<TypeParam>>(5, {a, a}),
        std::vector<RlweCiphertext<TypeParam>>(5, {a, a})};

    ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(2));
    EXPECT_THAT(ExternalProduct<TypeParam>(params, gadget_params, ct, rgsw_ct,
                                           *fft_ctx),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("ct lengths must match d.")));
  }
}

TYPED_TEST(EncryptionTest, ExternalProductFailsIfRgswCtElementLengthMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    GadgetParams gadget_params = {2, 5};

    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
    std::vector<TypeParam> a_short_coeffs(2, 0);
    ASSERT_OK_AND_ASSIGN(auto a_short,
                         Polynomial<TypeParam>::Create(a_short_coeffs));

    RlweCiphertext<TypeParam> ct = {a, a};
    RgswCiphertext<TypeParam> rgsw_ct = {
        std::vector<RlweCiphertext<TypeParam>>(5, {a, a}),
        std::vector<RlweCiphertext<TypeParam>>(5, {a, a})};
    rgsw_ct.u[0] = {a, a_short};

    ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(2));
    EXPECT_THAT(ExternalProduct<TypeParam>(params, gadget_params, ct, rgsw_ct,
                                           *fft_ctx),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("rgsw_ct element lengths must match d.")));
  }
}

TYPED_TEST(EncryptionTest, ExternalProductFailsIfRgswCtSizeMismatch) {
  std::vector<TypeParam> moduli = {
      0, 1024, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};
  for (TypeParam q : moduli) {
    ASSERT_OK_AND_ASSIGN(auto params, RlweParams<TypeParam>::Create(4, q, 8));
    GadgetParams gadget_params = {2, 5};

    std::vector<TypeParam> a_coeffs(4, 0);
    ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));

    RlweCiphertext<TypeParam> ct = {a, a};
    RgswCiphertext<TypeParam> rgsw_ct = {
        std::vector<RlweCiphertext<TypeParam>>(4, {a, a}),
        std::vector<RlweCiphertext<TypeParam>>(5, {a, a})};

    ASSERT_OK_AND_ASSIGN(auto fft_ctx, FftContext::Create(2));
    EXPECT_THAT(ExternalProduct<TypeParam>(params, gadget_params, ct, rgsw_ct,
                                           *fft_ctx),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("rgsw_ct sizes must match num_digits.")));
  }
}

TYPED_TEST(EncryptionTest,
           EncryptRgswCiphertextFromRlweSamplesAndExternalProductSucceeds) {
  std::vector<TypeParam> moduli = {
      0, static_cast<TypeParam>(1ULL << (sizeof(TypeParam) * 8 - 2))};

  ASSERT_OK_AND_ASSIGN(auto prng_seed,
                       ::rlwe::SingleThreadChaChaPrng::GenerateSeed());
  ASSERT_OK_AND_ASSIGN(auto prng,
                       ::rlwe::SingleThreadChaChaPrng::Create(prng_seed));

  const std::vector<int> degrees = {16, 32, 64, 2048};
  const std::vector<int> plaintext_moduli = {3, 17, 257, 4095};
  const int variance = 8;

  for (TypeParam q : moduli) {
    for (int d : degrees) {
      for (int p : plaintext_moduli) {
        int log_q = (q == 0) ? (sizeof(TypeParam) * 8) : absl::bit_width(q - 1);
        int log_digit = 2;
        // Use one less digit than the maximum possible to test approximate
        // gadget decomposition.
        int num_digits = (log_q + log_digit - 1) / log_digit - 1;
        GadgetParams gadget_params = {log_digit, num_digits};

        ASSERT_OK_AND_ASSIGN(auto params,
                             RlweParams<TypeParam>::Create(d, q, variance));

        ASSERT_OK_AND_ASSIGN(
            auto ctx,
            Context::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));

        // Secret key
        ASSERT_OK_AND_ASSIGN(auto s, SampleSecretKey<TypeParam>(params, *prng));

        // w_vec for u and v
        ASSERT_OK_AND_ASSIGN(
            std::vector<Polynomial<TypeParam>> w_vec_u,
            SampleAComponents<TypeParam>(params, num_digits, *prng));
        ASSERT_OK_AND_ASSIGN(
            std::vector<Polynomial<TypeParam>> w_vec_v,
            SampleAComponents<TypeParam>(params, num_digits, *prng));

        // Generate RLWE samples (-a*s + e) for w_vec_u and w_vec_v
        ASSERT_OK_AND_ASSIGN(
            auto rlwe_samples_u,
            GenerateRlweSamples<TypeParam>(params, s, w_vec_u, *prng));
        ASSERT_OK_AND_ASSIGN(
            auto rlwe_samples_v,
            GenerateRlweSamples<TypeParam>(params, s, w_vec_v, *prng));

        // Message message.
        std::vector<TypeParam> message(d, 0);
        message[1] = 1;  // encrypting X

        // Generate RGSW ciphertext
        ASSERT_OK_AND_ASSIGN(auto rgsw_ct,
                             EncryptRgswCiphertextFromRlweSamples<TypeParam>(
                                 params, gadget_params, rlwe_samples_u,
                                 rlwe_samples_v, s, message));

        // Generate an arbitrary message m for RLWE ciphertext
        std::vector<TypeParam> m_coeffs(d, 0);
        for (int k = 0; k < d; ++k) {
          m_coeffs[k] = (k * 987654321ULL) % p;
        }

        // Sample a_components for RLWE ciphertext
        std::vector<Polynomial<TypeParam>> a_components;
        std::vector<TypeParam> a_coeffs(d, 1);
        ASSERT_OK_AND_ASSIGN(auto a, Polynomial<TypeParam>::Create(a_coeffs));
        a_components.push_back(a);

        // Encrypt message M under s.
        ASSERT_OK_AND_ASSIGN(auto samples, GenerateRlweSamples<TypeParam>(
                                               params, s, a_components, *prng));

        ASSERT_OK_AND_ASSIGN(auto ct, EncryptFromRlweSample<TypeParam>(
                                          params, p, samples[0], m_coeffs));

        // Compute external product.
        ASSERT_OK_AND_ASSIGN(
            auto fft_ctx,
            FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));
        ASSERT_OK_AND_ASSIGN(
            auto ext_ct, ExternalProduct<TypeParam>(params, gadget_params, ct,
                                                    rgsw_ct, *fft_ctx));

        // Decrypt under s
        ASSERT_OK_AND_ASSIGN(auto m_dec,
                             Decrypt<TypeParam>(params, p, ext_ct, s));

        // Since m = X, the decrypted message should be a cyclic shift of m by
        // 1, with the first coefficient negated.
        for (int i = 1; i < d; ++i) {
          EXPECT_EQ(m_dec[i], m_coeffs[i - 1]);
        }
        // The first coefficient of m_dec is a negation of the last coefficient
        // of m.
        EXPECT_EQ(m_dec[0], (p - m_coeffs[d - 1]) % p);
      }
    }
  }
}

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
