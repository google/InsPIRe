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

#include "crypto/db_row_interpolation.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "common/modular_int/modular_int.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "crypto/status_macros.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

using ::private_membership::ModularInt;
using ::private_membership::ModularIntParams;

namespace {

// Multiplies the polynomial `poly` by X^power in the ring Z_p[X] / (X^d + 1).
// This is equivalent to a cyclic shift of the coefficients by `power`
// positions, where elements wrapped around the boundaries are negated modulo
// `modulus`.
//
// and extract as a testable utility function.
template <typename T>
std::vector<ModularInt<T>> MultiplyByXPower(
    const std::vector<ModularInt<T>>& poly, int power,
    const ModularIntParams<T>* params) {
  const int d = poly.size();
  std::vector<ModularInt<T>> result;
  result.reserve(d);
  power = power % (2 * d);
  if (power < 0) {
    power += 2 * d;
  }

  bool negate_all = false;
  if (power >= d) {
    negate_all = true;
    power -= d;
  }

  for (int i = 0; i < d; ++i) {
    ModularInt<T> val = ModularInt<T>::ImportZero(params);
    bool negate_this = negate_all;
    if (i < power) {
      val = poly[d - power + i];
      negate_this = !negate_this;
    } else {
      val = poly[i - power];
    }

    if (negate_this) {
      result.push_back(val.Negate(params));
    } else {
      result.push_back(val);
    }
  }
  return result;
}

// Computes the multiplicative inverse of `a` modulo `m` using the Extended
// Euclidean Algorithm. This works for any modulus `m` as long as `a` and `m`
// are coprime (e.g., when `a` is a power of 2 and `m` is odd).
// Returns an error if the inverse does not exist.
// untested cases (e.g. when a and m are not co-prime).
template <typename T>
absl::StatusOr<T> ModInverse(T a, T m) {
  int64_t t = 0;
  int64_t new_t = 1;
  int64_t r = m;
  int64_t new_r = a;

  while (new_r != 0) {
    int64_t quotient = r / new_r;
    int64_t temp_t = t - quotient * new_t;
    t = new_t;
    new_t = temp_t;

    int64_t temp_r = r - quotient * new_r;
    r = new_r;
    new_r = temp_r;
  }

  if (r > 1) {
    return absl::InvalidArgumentError(
        "a is not invertible modulo m (not coprime).");
  }
  if (t < 0) {
    t += m;
  }
  return static_cast<T>(t);
}

}  // namespace

template <typename T>
absl::StatusOr<std::vector<T>> Interpolate(const std::vector<T>& db_row,
                                           int ring_degree,
                                           int interpolation_degree,
                                           T plaintext_modulus) {
  const int d = ring_degree;
  const int t = interpolation_degree;

  if (db_row.size() != d * t) {
    return absl::InvalidArgumentError(
        "Length of evals must be ring_degree * interpolation_degree.");
  }
  if (!absl::has_single_bit(static_cast<unsigned int>(d))) {
    return absl::InvalidArgumentError("ring_degree must be a power of two.");
  }
  if (!absl::has_single_bit(static_cast<unsigned int>(t))) {
    return absl::InvalidArgumentError(
        "interpolation_degree must be a power of two.");
  }
  if (t > 2 * d) {
    return absl::InvalidArgumentError(
        "interpolation_degree must be less than or equal to 2 * ring_degree.");
  }
  if (t == 1) {
    return db_row;
  }
  if (plaintext_modulus % 2 == 0) {
    return absl::InvalidArgumentError("plaintext_modulus must be odd.");
  }

  ASSIGN_OR_RETURN(auto params,
                   ModularIntParams<uint32_t>::Create(plaintext_modulus));

  auto bit_reverse = [](int n, int bits) {
    int res = 0;
    for (int bit = 0; bit < bits; ++bit) {
      res = (res << 1) | ((n >> bit) & 1);
    }
    return res;
  };

  const int log_t = absl::countr_zero(static_cast<unsigned int>(t));

  // Parse db_row into t polynomials of degree d, placing them in bit reversed
  // order.
  std::vector<std::vector<ModularInt<uint32_t>>> polys(t);
  for (int i = 0; i < t; ++i) {
    int target_idx = bit_reverse(i, log_t);
    std::vector<ModularInt<uint32_t>> poly;
    poly.reserve(d);
    for (int j = 0; j < d; ++j) {
      ASSIGN_OR_RETURN(auto val, ModularInt<uint32_t>::ImportInt(
                                     db_row[i * d + j], params.get()));
      poly.push_back(std::move(val));
    }
    polys[target_idx] = std::move(poly);
  }

  // Perform an iterative Cooley-Tukey Inverse Fast Fourier Transform.
  //
  // We implement custom polynomial arithmetic (addition, subtraction, and
  // scalar mult) directly on the coefficients. Existing libraries are
  // unsuitable for this because:
  //   1. `rlwe::Polynomial` maintains its coefficients in NTT representation.
  //   2. `private_membership::rlwe::v2::Polynomial` only supports power of two
  //      moduli.
  //   3. `dft_transformation.h` cannot be used directly as this interpolation
  //      occurs over the specific cyclotomic ring Z_p[X] / (X^d + 1).
  for (int len = 2; len <= t; len <<= 1) {
    const int half = len / 2;
    // omega_len = omega^{-t/len} = X^{-2d/len} = X^{2d - 2d/len}
    // So the power of X is (2d - 2d/len)
    const int omega_power_step = (2 * d - 2 * d / len) % (2 * d);

    for (int i = 0; i < t; i += len) {
      // omega^0 = 1
      int current_omega_power = 0;
      for (int j = 0; j < half; ++j) {
        const auto& u = polys[i + j];
        auto v = MultiplyByXPower(polys[i + j + half], current_omega_power,
                                  params.get());

        ASSIGN_OR_RETURN(auto new_u,
                         ModularInt<uint32_t>::BatchAdd(u, v, params.get()));
        ASSIGN_OR_RETURN(auto new_v,
                         ModularInt<uint32_t>::BatchSub(u, v, params.get()));

        polys[i + j] = std::move(new_u);
        polys[i + j + half] = std::move(new_v);

        current_omega_power =
            (current_omega_power + omega_power_step) % (2 * d);
      }
    }
  }

  // Multiply by t^{-1} mod p
  ASSIGN_OR_RETURN(
      auto t_inv_int,
      ModInverse(static_cast<T>(t % plaintext_modulus), plaintext_modulus));
  ASSIGN_OR_RETURN(auto t_inv,
                   ModularInt<uint32_t>::ImportInt(t_inv_int, params.get()));

  std::vector<T> result;
  result.reserve(d * t);
  for (int i = 0; i < t; ++i) {
    for (int j = 0; j < d; ++j) {
      ModularInt<uint32_t> val = polys[i][j].Mul(t_inv, params.get());
      result.push_back(val.ExportInt(params.get()));
    }
  }

  return result;
}

template absl::StatusOr<std::vector<uint8_t>> Interpolate(
    const std::vector<uint8_t>&, int, int, uint8_t);
template absl::StatusOr<std::vector<uint16_t>> Interpolate(
    const std::vector<uint16_t>&, int, int, uint16_t);
template absl::StatusOr<std::vector<uint32_t>> Interpolate(
    const std::vector<uint32_t>&, int, int, uint32_t);

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
