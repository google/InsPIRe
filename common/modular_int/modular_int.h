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

#ifndef COMMON_MODULAR_INT_MODULAR_INT_H_
#define COMMON_MODULAR_INT_MODULAR_INT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "crypto/status_macros.h"
#include "shell_encryption/bits_util.h"
#include "shell_encryption/int256.h"
#include "shell_encryption/prng/prng.h"
#include "shell_encryption/serialization.pb.h"

namespace private_membership {

namespace internal {

// Struct to capture the "bigger int" type.
template <typename T>
struct BigInt;
// Specialization for uint8, uint16, uint32, uint64, and uint128.
template <>
struct BigInt<uint8_t> {
  typedef uint16_t value_type;
};
template <>
struct BigInt<uint16_t> {
  typedef uint32_t value_type;
};
template <>
struct BigInt<uint32_t> {
  typedef uint64_t value_type;
};
template <>
struct BigInt<uint64_t> {
  typedef absl::uint128 value_type;
};
template <>
struct BigInt<absl::uint128> {
  typedef ::rlwe::uint256 value_type;
};
}  // namespace internal

// Whether AVX2 implementation is enabled for ModularInt.
bool IsAvx2EnabledForModularInt();

// The parameters necessary for a modular integer. Note that the template
// parameters ensure that T is an unsigned integral of at least 8 bits.
template <typename T>
struct ModularIntParams {
  // Expose Int and its greater type. BigInt is required in order to multiply
  // two Int and ensure that no overflow occurs.
  //
  // Thread safe.
  using Int = T;
  using BigInt = typename internal::BigInt<Int>::value_type;
  static const size_t bitsize_int = sizeof(Int) * 8;

  // Factory function to create ModularIntParams.
  static absl::StatusOr<std::unique_ptr<const ModularIntParams>> Create(
      Int modulus);

  // The modulus over which these modular operations are being performed.
  const Int modulus;

  // The modulus over which these modular operations are being performed, cast
  // as a BigInt.
  const BigInt modulus_bigint;

  // The number of bits in the modulus.
  const unsigned int log_modulus;

  // The numerator used in the Barrett reduction.
  const BigInt barrett_numerator;         // = 2^(sizeof(Int)*8) / modulus
  const BigInt barrett_numerator_bigint;  // = 2^(sizeof(BigInt)*8-1) / modulus

  Int One() const { return 1; }
  Int Two() const { return 2; }

  // Functions to perform Barrett reduction. For more details, see
  // https://en.wikipedia.org/wiki/Barrett_reduction.
  // This function should remains in the header file to avoid performance
  // regressions.
  Int BarrettReduce(Int input) const {
    Int out =
        static_cast<Int>((this->barrett_numerator * input) >> bitsize_int);
    out = input - (out * this->modulus);
    // The steps above produce an integer that is in the range [0, 2N).
    // We now reduce to the range [0, N).
    return (out >= this->modulus) ? out - this->modulus : out;
  }

  Int BarrettReduceBigInt(BigInt input) const {
    using BigBigInt = typename internal::BigInt<BigInt>::value_type;
    Int out = static_cast<Int>(
        (static_cast<BigBigInt>(this->barrett_numerator_bigint) * input) >>
        (sizeof(BigInt) * 8 - 1));
    out = static_cast<Int>(input) - (out * this->modulus);
    // The steps above produce an integer that is in the range [0, 2N).
    // We now reduce to the range [0, N).
    return (out >= this->modulus) ? out - this->modulus : out;
  }

  // Identity operator.
  Int ExportInt(Int input) const { return input; }

  // Computes the serialized byte length of an integer.
  unsigned int SerializedSize() const { return (log_modulus + 7) / 8; }

  // Check whether (1 << log_n) fits into the underlying Int type.
  static bool DoesLogNFit(uint64_t log_n) { return (log_n < bitsize_int - 1); }

  // Transform an Int into a double. Note that as soon as value is more than
  // 2^53, this is a potentially lossy conversion.
  static double GetDouble(Int value) { return static_cast<double>(value); }

 private:
  explicit ModularIntParams(Int mod)
      : modulus(mod),
        modulus_bigint(static_cast<BigInt>(this->modulus)),
        log_modulus(::rlwe::internal::BitLength(this->modulus)),
        barrett_numerator((static_cast<BigInt>(1) << bitsize_int) /
                          this->modulus_bigint),
        barrett_numerator_bigint(
            (static_cast<BigInt>(1) << (sizeof(BigInt) * 8 - 1)) /
            this->modulus_bigint) {}
};

// Modular integer representation.
template <typename T>
class ModularInt {
 public:
  // Expose Int and its greater type. BigInt is required in order to multiply
  // two Int and ensure that no overflow occurs. This should also be used by
  // external classes.
  using Int = T;
  using BigInt = typename internal::BigInt<Int>::value_type;

  // Expose the parameter type.
  using Params = ModularIntParams<T>;

  // Static factory that converts a standard integer type into a modular
  // integer. Does not take ownership of params. i.e., import "a".
  static absl::StatusOr<ModularInt> ImportInt(Int n, const Params* params);

  // Static factory that converts a vector of standard integer types into
  // modular integer types. Does not take ownership of params.
  static absl::StatusOr<std::vector<ModularInt>> BatchImportInts(
      const std::vector<Int>& ints, const Params* params);

  explicit ModularInt(Int n) : n_(n) {}

  // Static functions to create a ModularInt of 0 and 1.
  static ModularInt ImportZero(const Params* params);
  static ModularInt ImportOne(const Params* params);

  // Import a random integer using entropy from specified prng. Does not take
  // ownership of params or prng.
  template <typename Prng = ::rlwe::SecurePrng>
  static absl::StatusOr<ModularInt> ImportRandom(Prng* prng,
                                                 const Params* params) {
    // In order to generate unbiased randomness, we uniformly and randomly
    // sample integers in [0, 2^params->log_modulus) until the generated integer
    // is less than the modulus (i.e., we perform rejection sampling).
    ASSIGN_OR_RETURN(Int random_int,
                     GenerateRandomInt(params->log_modulus, prng));
    while (random_int >= params->modulus) {
      ASSIGN_OR_RETURN(random_int,
                       GenerateRandomInt(params->log_modulus, prng));
    }
    return ModularInt(random_int);
  }

  static BigInt DivAndTruncate(BigInt dividend, BigInt divisor);

  // No default constructor.
  ModularInt() = delete;

  // Default copy constructor.
  ModularInt(const ModularInt& that) = default;
  ModularInt& operator=(const ModularInt& that) = default;

  // Convert a Modular integer representation  back to the underlying integer.
  // i.e., export "a".
  Int ExportInt(const Params* params) const { return params->ExportInt(n_); }

  // Returns the least significant 64 bits of n.
  static uint64_t ExportUInt64(Int n) { return static_cast<uint64_t>(n); }

  // Serialization.
  absl::StatusOr<std::string> Serialize(const Params* params) const;
  static absl::StatusOr<std::string> SerializeVector(
      const std::vector<ModularInt>& coeffs, const Params* params);

  // Deserialization.
  static absl::StatusOr<ModularInt> Deserialize(absl::string_view payload,
                                                const Params* params);
  static absl::StatusOr<std::vector<ModularInt>> DeserializeVector(
      int num_coeffs, absl::string_view serialized, const Params* params);

  // Modular multiplication.
  // Perform a multiply followed by a Barrett reduction. Produces an output
  // in the range [0, N).
  ModularInt Mul(const ModularInt& that, const Params* params) const {
    ModularInt out(*this);
    return out.MulInPlace(that, params);
  }

  // This function should remains in the header file to avoid performance
  // regressions.
  ModularInt& MulInPlace(const ModularInt& that, const Params* params) {
    BigInt m = static_cast<BigInt>(n_) * that.n_;
    n_ = params->BarrettReduceBigInt(m);
    return *this;
  }

  // The function GetConstant() returns a tuple (constant, constant_barrett):
  //       constant = ExportInt(params),
  // and
  //       constant_barrett = (constant << bitsize_int) / modulus
  std::tuple<Int, Int> GetConstant(const Params* params) const;

  ModularInt MulConstant(const Int& constant, const Int& constant_barrett,
                         const Params* params) const {
    ModularInt out(*this);
    return out.MulConstantInPlace(constant, constant_barrett, params);
  }

  // This function should remains in the header file to avoid performance
  // regressions.
  // The value this->n_ can be in [0, 4*modulus] since this operation is
  // actually a multiplication over a BigInt followed by a Barrett reduction.
  ModularInt& MulConstantInPlace(const Int& constant,
                                 const Int& constant_barrett,
                                 const Params* params) {
    Int out = static_cast<Int>((static_cast<BigInt>(constant_barrett) * n_) >>
                               Params::bitsize_int);
    n_ = n_ * constant - out * params->modulus;
    // The steps above produce an integer that is in the range [0, 2N).
    // We now reduce to the range [0, N).
    n_ -= (n_ >= params->modulus) ? params->modulus : 0;
    return *this;
  }

  // Modular int addition.
  ModularInt Add(const ModularInt& that, const Params* params) const {
    ModularInt out(*this);
    return out.AddInPlace(that, params);
  }

  // This function should remains in the header file to avoid performance
  // regressions.
  // The sum of this->n_ + that.n_ needs to be in [0, 4 * modulus].
  ModularInt& AddInPlace(const ModularInt& that, const Params* params) {
    // We can use Barrett reduction because n_ <= modulus < Max(Int)/4.
    n_ = params->BarrettReduce(n_ + that.n_);
    return *this;
  }

  // Modular negation.
  ModularInt Negate(const Params* params) const {
    ModularInt out(*this);
    return out.NegateInPlace(params);
  }

  // This function should remains in the header file to avoid performance
  // regressions.
  ModularInt& NegateInPlace(const Params* params) {
    n_ = n_ == 0 ? 0 : params->modulus - n_;
    return *this;
  }

  // Modular subtraction.
  ModularInt Sub(const ModularInt& that, const Params* params) const {
    ModularInt out(*this);
    return out.SubInPlace(that, params);
  }

  // This function should remain in the header file to avoid performance
  // regressions.
  // The value this->n_ can be in [0, 3*modulus] and that.n_ needs to be in
  // [0, modulus].
  ModularInt& SubInPlace(const ModularInt& that, const Params* params) {
    // We can use Barrett reduction because n_ <= modulus < Max(Int)/4.
    n_ = params->BarrettReduce(n_ + (params->modulus - that.n_));
    return *this;
  }

  // We enable lazy additions and lazy multiplications, where the output is not
  // assured to be in [0, modulus]. A lazy addition adds the underlying
  // Montgomery integers, whereas a lazy addition adds the left hand side with
  // modulus minus the right hand side.
  // These operations can be used in place of addition and subtraction, as long
  // as a Barrett reduction is performed on the value before the ModularInt
  // is used in the rest of the library.
  //
  // As an example, these operations are used when performing an NTT operation
  // on all odd layers.
  //
  // These functions should remains in the header file to avoid performance
  // regressions.

  // The sum of this->n_ + that.n_ needs to be in [0, 4 * modulus].
  ModularInt& LazyAddInPlace(const ModularInt& that, const Params* params) {
    n_ += that.n_;
    return *this;
  }

  // The value this->n_ can be in [0, 3*modulus] and that.n_ needs to be in
  // [0, modulus].
  ModularInt& LazySubInPlace(const ModularInt& that, const Params* params) {
    n_ += (params->modulus - that.n_);
    return *this;
  }

  ModularInt& FusedMulAddInPlace(const ModularInt& a, const ModularInt& b,
                                 const Params* params) {
    BigInt m = static_cast<BigInt>(n_) + static_cast<BigInt>(a.n_) * b.n_;
    n_ = params->BarrettReduceBigInt(m);
    return *this;
  }

  ModularInt& FusedMulConstantAddInPlace(const ModularInt& a,
                                         const Int& constant,
                                         const Int& constant_barrett,
                                         const Params* params) {
    // Compute this += a * constant, where this is multiplied by
    // the Barrett numerator.
    // Denote:
    // - m = modulus
    // - n = this
    // - b = constant
    // - k = bitsize_int
    // - CB = constant_barrett = (b << k) / m = floor(b * 2^k/m),
    // - BN = barrett_numerator = (1 << k) / m = floor(2^k/m),
    //
    // We recall that m <= k/4. The function below will compute
    //           a * b + n - floor((CB * a + BN * n)/2^k) * m
    // and we will prove that this is smaller than 2 * m, which enables to
    // perform a unique conditional subtraction by m to reduce the final result
    // in [0, m].
    //
    // We first express the floor as a difference,
    //   a * b + n - floor((CB * a + BN * n)/2^k) * m
    //   = ab + n - m * (CB * a + BN * n - (a * CB + n * BN mod 2^k))/2^k
    // and we bound m * (a * CB + n * BN mod 2^k)/2^k by m, which yields:
    //   <= ab + n - (CB*a + BN*n) * m/2^k + m.
    //
    // We know expand CB and BN with their definition,
    //   = ab+n - m /2^k *(floor(b * 2^k/m) * a + floor(2^k/m) * n) + m
    //   = ab+n - ma/2^k *(b2^k-(b2^k mod m))/m - mn/2^k*(2^k-(2^k mod m))/m + m
    //   = ab+n - a /2^k *(b2^k-(b2^k mod m))   - n /2^k*(2^k-(2^k mod m)) + m
    //   = ab+n - a*b + a*(b2^k mod m)/2^k - n + n(2^k mod m)/2^k + m
    //   = a*(b2^k mod m)/2^k + n(2^k mod m)/2^k + m
    //
    // Finally, we note that a, n <= m, and we recall that m <= k/4. We get
    //   <= 2m^2 / 2^k + m <= 2m^2/(4m) + m <= m/2+m < 2*m
    Int out = static_cast<Int>((static_cast<BigInt>(constant_barrett) * a.n_ +
                                params->barrett_numerator * n_) >>
                               Params::bitsize_int);
    n_ += a.n_ * constant - out * params->modulus;
    n_ -= (n_ >= params->modulus) ? params->modulus : 0;
    return *this;
  }

  // Batch addition of two vectors.
  static absl::StatusOr<std::vector<ModularInt>> BatchAdd(
      const std::vector<ModularInt>& in1, const std::vector<ModularInt>& in2,
      const Params* params);
  static absl::Status BatchAddInPlace(std::vector<ModularInt>* in1,
                                      const std::vector<ModularInt>& in2,
                                      const Params* params);

  // Batch addition of one vector with a scalar.
  static absl::StatusOr<std::vector<ModularInt>> BatchAdd(
      const std::vector<ModularInt>& in1, const ModularInt& in2,
      const Params* params);
  static absl::Status BatchAddInPlace(std::vector<ModularInt>* in1,
                                      const ModularInt& in2,
                                      const Params* params);

  // Batch subtraction of two vectors.
  static absl::StatusOr<std::vector<ModularInt>> BatchSub(
      const std::vector<ModularInt>& in1, const std::vector<ModularInt>& in2,
      const Params* params);
  static absl::Status BatchSubInPlace(std::vector<ModularInt>* in1,
                                      const std::vector<ModularInt>& in2,
                                      const Params* params);

  // Batch subtraction of one vector with a scalar.
  static absl::StatusOr<std::vector<ModularInt>> BatchSub(
      const std::vector<ModularInt>& in1, const ModularInt& in2,
      const Params* params);
  static absl::Status BatchSubInPlace(std::vector<ModularInt>* in1,
                                      const ModularInt& in2,
                                      const Params* params);

  // Batch multiplication of two vectors.
  static absl::StatusOr<std::vector<ModularInt>> BatchMul(
      const std::vector<ModularInt>& in1, const std::vector<ModularInt>& in2,
      const Params* params);
  static absl::Status BatchMulInPlace(std::vector<ModularInt>* in1,
                                      const std::vector<ModularInt>& in2,
                                      const Params* params);

  // Batch fused multiply add of two vectors.
  static absl::StatusOr<std::vector<ModularInt>> BatchFusedMulAdd(
      const std::vector<ModularInt>& in1, const std::vector<ModularInt>& in2,
      const std::vector<ModularInt>& in3, const Params* params);
  static absl::Status BatchFusedMulAddInPlace(
      std::vector<ModularInt>* in1, const std::vector<ModularInt>& in2,
      const std::vector<ModularInt>& in3, const Params* params);

  // Batch fused multiply add of two vectors, one being constant.
  static absl::StatusOr<std::vector<ModularInt>> BatchFusedMulConstantAdd(
      const std::vector<ModularInt>& in1, const std::vector<ModularInt>& in2,
      const std::vector<Int>& constant,
      const std::vector<Int>& constant_barrett, const Params* params);
  static absl::Status BatchFusedMulConstantAddInPlace(
      std::vector<ModularInt>* in1, const std::vector<ModularInt>& in2,
      const std::vector<Int>& constant,
      const std::vector<Int>& constant_barrett, const Params* params);

  static absl::StatusOr<std::vector<ModularInt>> BatchMulConstant(
      const std::vector<ModularInt>& in1, const std::vector<Int>& in2_constant,
      const std::vector<Int>& in2_constant_barrett, const Params* params);
  static absl::Status BatchMulConstantInPlace(
      std::vector<ModularInt>* in1, const std::vector<Int>& in2_constant,
      const std::vector<Int>& in2_constant_barrett, const Params* params);

  // Batch multiplication of a vector with a scalar.
  static absl::StatusOr<std::vector<ModularInt>> BatchMul(
      const std::vector<ModularInt>& in1, const ModularInt& in2,
      const Params* params);
  static absl::Status BatchMulInPlace(std::vector<ModularInt>* in1,
                                      const ModularInt& in2,
                                      const Params* params);

  // Batch multiplication of a vector with a constant scalar.
  static absl::StatusOr<std::vector<ModularInt>> BatchMulConstant(
      const std::vector<ModularInt>& in1, const Int& constant,
      const Int& constant_barrett, const Params* params);
  static absl::Status BatchMulConstantInPlace(std::vector<ModularInt>* in1,
                                              const Int& constant,
                                              const Int& constant_barrett,
                                              const Params* params);

  // Equality.
  bool operator==(const ModularInt& that) const { return (n_ == that.n_); }
  bool operator!=(const ModularInt& that) const { return !(*this == that); }

  // Modular exponentiation.
  ModularInt ModExp(Int exponent, const Params* params) const;

  // Inverse.
  ModularInt MultiplicativeInverse(const Params* params) const;

 private:
  template <typename Prng = ::rlwe::SecurePrng>
  static absl::StatusOr<Int> GenerateRandomInt(int log_modulus, Prng* prng) {
    // Generate a random Int. As the modulus is always smaller than max(Int),
    // there will be no issues with overflow.
    int max_bits_per_step = std::min((int)Params::bitsize_int, (int)64);
    auto bits_required = log_modulus;
    Int rand = 0;
    while (bits_required > 0) {
      Int rand_bits = 0;
      if (bits_required <= 8) {
        // Generate 8 bits of randomness.
        ASSIGN_OR_RETURN(rand_bits, prng->Rand8());

        // Extract bits_required bits and add them to rand.
        Int needed_bits =
            rand_bits & ((static_cast<Int>(1) << bits_required) - 1);
        rand = (rand << bits_required) + needed_bits;
        break;
      } else {
        // Generate 64 bits of randomness.
        ASSIGN_OR_RETURN(rand_bits, prng->Rand64());

        // Extract min(64, bits in Int, bits_required) bits and add them to rand
        int bits_to_extract = std::min(bits_required, max_bits_per_step);
        Int needed_bits =
            rand_bits & ((static_cast<Int>(1) << bits_to_extract) - 1);
        rand = (rand << bits_to_extract) + needed_bits;
        bits_required -= bits_to_extract;
      }
    }
    return rand;
  }

  Int n_;
};

// Instantiations of ModularInt and ModularIntParams with specific
// integral types.
extern template struct ModularIntParams<uint16_t>;
extern template struct ModularIntParams<uint32_t>;
extern template struct ModularIntParams<uint64_t>;
extern template class ModularInt<uint16_t>;
extern template class ModularInt<uint32_t>;
extern template class ModularInt<uint64_t>;

}  // namespace private_membership

#endif  // COMMON_MODULAR_INT_MODULAR_INT_H_
