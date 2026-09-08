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

#include "common/modular_int/modular_int.h"

#include <cstring>
#include <string>
#include <vector>

#include "common/modular_int/batch_modular_int_avx2.h"
#include "common/modular_int/batch_modular_int_hardware_acceleration.h"
#include "common/modular_int/batch_modular_int_hardware_acceleration_impl.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "crypto/status_macros.h"
#include "shell_encryption/transcription.h"

namespace private_membership {

constexpr uint64_t kMaxNumCoeffs = 1L << 15;

bool IsAvx2EnabledForModularInt() {
#if MODULAR_INT_AVX2_ENABLED
  return true;
#else
  return false;
#endif
}

template <typename T>
absl::StatusOr<std::unique_ptr<const ModularIntParams<T>>>
ModularIntParams<T>::Create(Int modulus) {
  // Check that the modulus is smaller than max(Int) / 4.
  Int most_significant_bit = modulus >> (bitsize_int - 2);
  if (most_significant_bit != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The modulus should be less than 2^", (bitsize_int - 2), "."));
  }
  if ((modulus % 2) == 0) {
    return absl::InvalidArgumentError("The modulus should be odd.");
  }
  return absl::WrapUnique<const ModularIntParams>(
      new ModularIntParams(modulus));
}

template <typename T>
absl::StatusOr<ModularInt<T>> ModularInt<T>::ImportInt(Int n,
                                                       const Params* params) {
  n = params->BarrettReduce(n);
  return ModularInt(n);
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::BatchImportInts(
    const std::vector<Int>& ints, const Params* params) {
  std::vector<ModularInt<T>> result(ints.size(),
                                    ModularInt<T>::ImportZero(params));
  memcpy(result.data(), ints.data(), ints.size() * sizeof(Int));
  for (int i = 0; i < result.size(); ++i) {
    if (ints[i] >= params->modulus) {
      result[i] = ModularInt(params->BarrettReduce(ints[i]));
    }
  }
  return result;
}

template <typename T>
ModularInt<T> ModularInt<T>::ImportZero(const Params* params) {
  return ModularInt(0);
}

template <typename T>
ModularInt<T> ModularInt<T>::ImportOne(const Params* params) {
  return ModularInt(1);
}

template <typename T>
typename internal::BigInt<T>::value_type ModularInt<T>::DivAndTruncate(
    BigInt dividend, BigInt divisor) {
  return dividend / divisor;
}

template <typename T>
absl::StatusOr<std::string> ModularInt<T>::Serialize(
    const Params* params) const {
  // Use transcription to transform all the LogModulus() bits of input into a
  // vector of unsigned char.
  ASSIGN_OR_RETURN(
      auto v, (::rlwe::TranscribeBits<Int, uint8_t>(
                  {this->n_}, params->log_modulus, params->log_modulus, 8)));
  // Return a string
  return std::string(std::make_move_iterator(v.begin()),
                     std::make_move_iterator(v.end()));
}

template <typename T>
absl::StatusOr<std::string> ModularInt<T>::SerializeVector(
    const std::vector<ModularInt>& coeffs, const Params* params) {
  if (coeffs.size() > kMaxNumCoeffs) {
    return absl::InvalidArgumentError(
        absl::StrCat("Number of coefficients, ", coeffs.size(),
                     ", cannot be larger than ", kMaxNumCoeffs, "."));
  } else if (coeffs.empty()) {
    return absl::InvalidArgumentError("Cannot serialize an empty vector.");
  }
  // Bits required to represent modulus.
  int bit_size = params->log_modulus;
  // Extract the values
  std::vector<Int> coeffs_values;
  coeffs_values.reserve(coeffs.size());
  for (const auto& c : coeffs) {
    coeffs_values.push_back(c.n_);
  }
  // Use transcription to transform all the bit_size bits of input into a
  // vector of unsigned char.
  ASSIGN_OR_RETURN(auto v, (::rlwe::TranscribeBits<Int, uint8_t>(
                               coeffs_values, coeffs_values.size() * bit_size,
                               bit_size, 8)));
  // Return a string
  return std::string(std::make_move_iterator(v.begin()),
                     std::make_move_iterator(v.end()));
}

template <typename T>
absl::StatusOr<ModularInt<T>> ModularInt<T>::Deserialize(
    absl::string_view payload, const Params* params) {
  // Parse the string as unsigned char
  std::vector<uint8_t> input(payload.begin(), payload.end());
  // Bits required to represent modulus.
  int bit_size = params->log_modulus;
  // Recover the coefficients from the input stream.
  ASSIGN_OR_RETURN(auto coeffs_values, (::rlwe::TranscribeBits<uint8_t, Int>(
                                           input, bit_size, 8, bit_size)));
  // There will be at least one coefficient in coeff_values because bit_size
  // is always expected to be positive.
  return ModularInt(coeffs_values[0]);
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::DeserializeVector(
    int num_coeffs, absl::string_view serialized, const Params* params) {
  if (num_coeffs < 0) {
    return absl::InvalidArgumentError(
        "Number of coefficients must be non-negative.");
  }
  if (num_coeffs > static_cast<int>(kMaxNumCoeffs)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Number of coefficients, ", num_coeffs, ", cannot be ",
                     "larger than ", kMaxNumCoeffs, "."));
  }
  // Parse the string as unsigned char
  std::vector<uint8_t> input(serialized.begin(), serialized.end());
  // Bits required to represent modulus.
  int bit_size = params->log_modulus;
  // Recover the coefficients from the input stream.
  ASSIGN_OR_RETURN(auto coeffs_values,
                   (::rlwe::TranscribeBits<uint8_t, Int>(
                       input, bit_size * num_coeffs, 8, bit_size)));
  // Check that the number of coefficients recovered is at least what is
  // expected.
  if (coeffs_values.size() < static_cast<size_t>(num_coeffs)) {
    return absl::InvalidArgumentError("Given serialization is invalid.");
  }
  // Create a vector of ModularInt from the values.
  std::vector<ModularInt> coeffs;
  coeffs.reserve(num_coeffs);
  for (int i = 0; i < num_coeffs; i++) {
    coeffs.push_back(ModularInt(coeffs_values[i]));
  }
  return coeffs;
}

template <typename T>
std::tuple<T, T> ModularInt<T>::GetConstant(const Params* params) const {
  Int constant = ExportInt(params);
  Int constant_barrett = static_cast<Int>(
      (static_cast<BigInt>(constant) << params->bitsize_int) / params->modulus);
  return std::make_tuple(constant, constant_barrett);
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::BatchAdd(
    const std::vector<ModularInt>& in1, const std::vector<ModularInt>& in2,
    const Params* params) {
  std::vector<ModularInt> out = in1;
  RETURN_IF_ERROR(BatchAddInPlace(&out, in2, params));
  return out;
}

template <typename T>
absl::Status ModularInt<T>::BatchAddInPlace(std::vector<ModularInt>* in1,
                                            const std::vector<ModularInt>& in2,
                                            const Params* params) {
  // If the input vectors' sizes don't match, return an error.
  if (in1->size() != in2.size()) {
    return absl::InvalidArgumentError("Input vectors are not of same size");
  }
  int i = internal::MaximumBatchAddInPlaceWithHardware<T>(in1, in2, params);
  for (; i < in1->size(); i++) {
    (*in1)[i].AddInPlace(in2[i], params);
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::BatchAdd(
    const std::vector<ModularInt>& in1, const ModularInt& in2,
    const Params* params) {
  std::vector<ModularInt> out = in1;
  RETURN_IF_ERROR(BatchAddInPlace(&out, in2, params));
  return out;
}

template <typename T>
absl::Status ModularInt<T>::BatchAddInPlace(std::vector<ModularInt>* in1,
                                            const ModularInt& in2,
                                            const Params* params) {
  int i = internal::MaximumBatchAddInPlaceWithHardware<T>(in1, in2, params);
  for (; i < in1->size(); i++) {
    (*in1)[i].AddInPlace(in2, params);
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::BatchSub(
    const std::vector<ModularInt>& in1, const std::vector<ModularInt>& in2,
    const Params* params) {
  std::vector<ModularInt> out = in1;
  RETURN_IF_ERROR(BatchSubInPlace(&out, in2, params));
  return out;
}

template <typename T>
absl::Status ModularInt<T>::BatchSubInPlace(std::vector<ModularInt>* in1,
                                            const std::vector<ModularInt>& in2,
                                            const Params* params) {
  // If the input vectors' sizes don't match, return an error.
  if (in1->size() != in2.size()) {
    return absl::InvalidArgumentError("Input vectors are not of same size");
  }
  int i = internal::MaximumBatchSubInPlaceWithHardware<T>(in1, in2, params);
  for (; i < in1->size(); i++) {
    (*in1)[i].SubInPlace(in2[i], params);
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::BatchSub(
    const std::vector<ModularInt>& in1, const ModularInt& in2,
    const Params* params) {
  std::vector<ModularInt> out = in1;
  RETURN_IF_ERROR(BatchSubInPlace(&out, in2, params));
  return out;
}

template <typename T>
absl::Status ModularInt<T>::BatchSubInPlace(std::vector<ModularInt>* in1,
                                            const ModularInt& in2,
                                            const Params* params) {
  int i = internal::MaximumBatchSubInPlaceWithHardware<T>(in1, in2, params);
  for (; i < in1->size(); i++) {
    (*in1)[i].SubInPlace(in2, params);
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::BatchMulConstant(
    const std::vector<ModularInt>& in1, const std::vector<Int>& constant,
    const std::vector<Int>& constant_barrett, const Params* params) {
  std::vector<ModularInt> out = in1;
  RETURN_IF_ERROR(
      BatchMulConstantInPlace(&out, constant, constant_barrett, params));
  return out;
}

template <typename T>
absl::Status ModularInt<T>::BatchMulConstantInPlace(
    std::vector<ModularInt>* in1, const std::vector<Int>& constant,
    const std::vector<Int>& constant_barrett, const Params* params) {
  // If the input vectors' sizes don't match, return an error.
  if (in1->size() != constant.size() ||
      constant.size() != constant_barrett.size()) {
    return absl::InvalidArgumentError("Input vectors are not of same size");
  }
  for (int i = 0; i < in1->size(); i++) {
    (*in1)[i].MulConstantInPlace(constant[i], constant_barrett[i], params);
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::BatchMulConstant(
    const std::vector<ModularInt>& in1, const Int& constant,
    const Int& constant_barrett, const Params* params) {
  std::vector<ModularInt> out = in1;
  RETURN_IF_ERROR(
      BatchMulConstantInPlace(&out, constant, constant_barrett, params));
  return out;
}

template <typename T>
absl::Status ModularInt<T>::BatchMulConstantInPlace(
    std::vector<ModularInt>* in1, const Int& constant,
    const Int& constant_barrett, const Params* params) {
  for (int i = 0; i < in1->size(); i++) {
    (*in1)[i].MulConstantInPlace(constant, constant_barrett, params);
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::BatchMul(
    const std::vector<ModularInt>& in1, const std::vector<ModularInt>& in2,
    const Params* params) {
  std::vector<ModularInt> out = in1;
  RETURN_IF_ERROR(BatchMulInPlace(&out, in2, params));
  return out;
}

template <typename T>
absl::Status ModularInt<T>::BatchMulInPlace(std::vector<ModularInt>* in1,
                                            const std::vector<ModularInt>& in2,
                                            const Params* params) {
  // If the input vectors' sizes don't match, return an error.
  if (in1->size() != in2.size()) {
    return absl::InvalidArgumentError("Input vectors are not of same size");
  }
  int i = internal::MaximumBatchMulInPlaceWithHardware<T>(in1, in2, params);
  for (; i < in1->size(); i++) {
    (*in1)[i].MulInPlace(in2[i], params);
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::BatchMul(
    const std::vector<ModularInt>& in1, const ModularInt& in2,
    const Params* params) {
  std::vector<ModularInt> out = in1;
  RETURN_IF_ERROR(BatchMulInPlace(&out, in2, params));
  return out;
}

template <typename T>
absl::Status ModularInt<T>::BatchMulInPlace(std::vector<ModularInt>* in1,
                                            const ModularInt& in2,
                                            const Params* params) {
  for (int i = 0; i < in1->size(); i++) {
    (*in1)[i].MulInPlace(in2, params);
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>> ModularInt<T>::BatchFusedMulAdd(
    const std::vector<ModularInt>& in1, const std::vector<ModularInt>& in2,
    const std::vector<ModularInt>& in3, const Params* params) {
  std::vector<ModularInt> out = in1;
  RETURN_IF_ERROR(BatchFusedMulAddInPlace(&out, in2, in3, params));
  return out;
}

template <typename T>
absl::Status ModularInt<T>::BatchFusedMulAddInPlace(
    std::vector<ModularInt>* in1, const std::vector<ModularInt>& in2,
    const std::vector<ModularInt>& in3, const Params* params) {
  // If the input vectors' sizes don't match, return an error.
  if (in1->size() != in2.size() || in1->size() != in3.size()) {
    return absl::InvalidArgumentError("Input vectors are not of same size");
  }
  int i = internal::MaximumBatchFusedMulAddInPlaceWithHardware<T>(in1, in2, in3,
                                                                  params);
  for (; i < in1->size(); i++) {
    (*in1)[i].FusedMulAddInPlace(in2[i], in3[i], params);
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<std::vector<ModularInt<T>>>
ModularInt<T>::BatchFusedMulConstantAdd(
    const std::vector<ModularInt>& in1, const std::vector<ModularInt>& in2,
    const std::vector<Int>& constant, const std::vector<Int>& constant_barrett,
    const Params* params) {
  std::vector<ModularInt> out = in1;
  RETURN_IF_ERROR(BatchFusedMulConstantAddInPlace(&out, in2, constant,
                                                  constant_barrett, params));
  return out;
}

template <typename T>
absl::Status ModularInt<T>::BatchFusedMulConstantAddInPlace(
    std::vector<ModularInt>* in1, const std::vector<ModularInt>& in2,
    const std::vector<Int>& constant, const std::vector<Int>& constant_barrett,
    const Params* params) {
  // If the input vectors' sizes don't match, return an error.
  if (in1->size() != in2.size() || in1->size() != constant.size() ||
      in1->size() != constant_barrett.size()) {
    return absl::InvalidArgumentError("Input vectors are not of same size");
  }
  for (int i = 0; i < in1->size(); i++) {
    (*in1)[i].FusedMulConstantAddInPlace(in2[i], constant[i],
                                         constant_barrett[i], params);
  }
  return absl::OkStatus();
}

template <typename T>
ModularInt<T> ModularInt<T>::ModExp(Int exponent, const Params* params) const {
  ModularInt result = ModularInt::ImportOne(params);
  ModularInt base = *this;

  // Uses the bits of the exponent to gradually compute the result.
  // When bit k of the exponent is 1, the result is multiplied by
  // base^{2^k}.
  while (exponent > 0) {
    // If the current bit (bit k) is 1, multiply base^{2^k} into the result.
    if (exponent % 2 == 1) {
      result.MulInPlace(base, params);
    }

    // Update base from base^{2^k} to base^{2^{k+1}}.
    base.MulInPlace(base, params);
    exponent >>= 1;
  }

  return result;
}

template <typename T>
ModularInt<T> ModularInt<T>::MultiplicativeInverse(const Params* params) const {
  return (*this).ModExp(static_cast<Int>(params->modulus - 2), params);
}

// Instantiations of ModularInt and ModularIntParams with specific
// integral types.
template struct ModularIntParams<uint16_t>;
template struct ModularIntParams<uint32_t>;
template struct ModularIntParams<uint64_t>;
template class ModularInt<uint16_t>;
template class ModularInt<uint32_t>;
template class ModularInt<uint64_t>;

}  // namespace private_membership
