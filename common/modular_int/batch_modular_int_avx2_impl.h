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

#ifndef COMMON_MODULAR_INT_BATCH_MODULAR_INT_AVX2_IMPL_H_
#define COMMON_MODULAR_INT_BATCH_MODULAR_INT_AVX2_IMPL_H_

#include "common/modular_int/batch_modular_int_avx2.h"

#if MODULAR_INT_AVX2_ENABLED

namespace rlwe::internal {

template <>
__m256i BatchSetAvx2<private_membership::ModularInt<uint16_t>>(
    const private_membership::ModularInt<uint16_t>& a) {
  const uint16_t* ptr = reinterpret_cast<const uint16_t*>(&a);
  return internal::BatchSetAvx2<uint16_t>(*ptr);
}

template <>
__m256i BatchSetAvx2<private_membership::ModularInt<uint32_t>>(
    const private_membership::ModularInt<uint32_t>& a) {
  const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&a);
  return internal::BatchSetAvx2<uint32_t>(*ptr);
}

template <>
__m256i BatchSetAvx2<private_membership::ModularInt<uint64_t>>(
    const private_membership::ModularInt<uint64_t>& a) {
  const uint64_t* ptr = reinterpret_cast<const uint64_t*>(&a);
  return internal::BatchSetAvx2<uint64_t>(*ptr);
}

template <>
__m256i BatchLoadAvx2<private_membership::ModularInt<uint16_t>>(
    const private_membership::ModularInt<uint16_t>* ptr) {
  const __m256i* ptr_m256i = reinterpret_cast<const __m256i*>(ptr);
  return _mm256_loadu_si256(ptr_m256i);
}

template <>
__m256i BatchLoadAvx2<private_membership::ModularInt<uint32_t>>(
    const private_membership::ModularInt<uint32_t>* ptr) {
  const __m256i* ptr_m256i = reinterpret_cast<const __m256i*>(ptr);
  return _mm256_loadu_si256(ptr_m256i);
}

template <>
__m256i BatchLoadAvx2<private_membership::ModularInt<uint64_t>>(
    const private_membership::ModularInt<uint64_t>* ptr) {
  const __m256i* ptr_m256i = reinterpret_cast<const __m256i*>(ptr);
  return _mm256_loadu_si256(ptr_m256i);
}

template <>
void BatchStoreAvx2<private_membership::ModularInt<uint16_t>>(
    private_membership::ModularInt<uint16_t>* ptr, const __m256i& u) {
  __m256i* ptr_m256i = reinterpret_cast<__m256i*>(ptr);
  return _mm256_storeu_si256(ptr_m256i, u);
}

template <>
void BatchStoreAvx2<private_membership::ModularInt<uint32_t>>(
    private_membership::ModularInt<uint32_t>* ptr, const __m256i& u) {
  __m256i* ptr_m256i = reinterpret_cast<__m256i*>(ptr);
  return _mm256_storeu_si256(ptr_m256i, u);
}

template <>
void BatchStoreAvx2<private_membership::ModularInt<uint64_t>>(
    private_membership::ModularInt<uint64_t>* ptr, const __m256i& u) {
  __m256i* ptr_m256i = reinterpret_cast<__m256i*>(ptr);
  return _mm256_storeu_si256(ptr_m256i, u);
}

}  // namespace rlwe::internal

namespace private_membership::internal {

template <typename T>
int MaximumBatchAddInPlaceWithAvx2(std::vector<ModularInt<T>>* in1,
                                   const std::vector<ModularInt<T>>& in2,
                                   const ModularIntParams<T>* params) {
  int i = 0;
  const __m256i simd_modulus =
      ::rlwe::internal::BatchSetAvx2<T>(params->modulus);
  constexpr size_t batch_size = sizeof(__m256i) / sizeof(T);
  for (; i + batch_size <= in1->size(); i += batch_size) {
    // Load the next batch of ModularInt<Int>'s into 256-bit registers.
    const __m256i u =
        ::rlwe::internal::BatchLoadAvx2<ModularInt<T>>(in1->data() + i);
    const __m256i v =
        ::rlwe::internal::BatchLoadAvx2<ModularInt<T>>(in2.data() + i);
    // Store the modular modular addition in in1->data() + i.
    ::rlwe::internal::BatchStoreAvx2<ModularInt<T>>(
        in1->data() + i,
        ::rlwe::internal::BatchAddModAvx2<T>(u, v, simd_modulus));
  }
  return i;
}

template <typename T>
int MaximumBatchAddInPlaceWithAvx2(std::vector<ModularInt<T>>* in1,
                                   const ModularInt<T>& in2,
                                   const ModularIntParams<T>* params) {
  int i = 0;
  const __m256i simd_modulus =
      ::rlwe::internal::BatchSetAvx2<T>(params->modulus);
  const __m256i simd_in2 = ::rlwe::internal::BatchSetAvx2<ModularInt<T>>(in2);
  constexpr size_t batch_size = sizeof(__m256i) / sizeof(T);
  for (; i + batch_size <= in1->size(); i += batch_size) {
    const __m256i u =
        ::rlwe::internal::BatchLoadAvx2<ModularInt<T>>(in1->data() + i);
    ::rlwe::internal::BatchStoreAvx2<ModularInt<T>>(
        in1->data() + i,
        ::rlwe::internal::BatchAddModAvx2<T>(u, simd_in2, simd_modulus));
  }
  return i;
}

template <typename T>
int MaximumBatchSubInPlaceWithAvx2(std::vector<ModularInt<T>>* in1,
                                   const std::vector<ModularInt<T>>& in2,
                                   const ModularIntParams<T>* params) {
  int i = 0;
  const __m256i simd_modulus =
      ::rlwe::internal::BatchSetAvx2<T>(params->modulus);
  constexpr size_t batch_size = sizeof(__m256i) / sizeof(T);
  for (; i + batch_size <= in1->size(); i += batch_size) {
    const __m256i u =
        ::rlwe::internal::BatchLoadAvx2<ModularInt<T>>(in1->data() + i);
    const __m256i v =
        ::rlwe::internal::BatchLoadAvx2<ModularInt<T>>(in2.data() + i);
    ::rlwe::internal::BatchStoreAvx2<ModularInt<T>>(
        in1->data() + i,
        ::rlwe::internal::BatchSubModAvx2<T>(u, v, simd_modulus));
  }
  return i;
}

template <typename T>
int MaximumBatchSubInPlaceWithAvx2(std::vector<ModularInt<T>>* in1,
                                   const ModularInt<T>& in2,
                                   const ModularIntParams<T>* params) {
  int i = 0;
  const __m256i simd_modulus =
      ::rlwe::internal::BatchSetAvx2<T>(params->modulus);
  const __m256i simd_in2 = ::rlwe::internal::BatchSetAvx2<ModularInt<T>>(in2);
  constexpr size_t batch_size = sizeof(__m256i) / sizeof(T);
  for (; i + batch_size <= in1->size(); i += batch_size) {
    // Load the next batch of ModularInt<Int>'s into 256-bit registers.
    const __m256i u =
        ::rlwe::internal::BatchLoadAvx2<ModularInt<T>>(in1->data() + i);
    // Store the modular modular addition in in1->data() + i.
    ::rlwe::internal::BatchStoreAvx2<ModularInt<T>>(
        in1->data() + i,
        ::rlwe::internal::BatchSubModAvx2<T>(u, simd_in2, simd_modulus));
  }
  return i;
}

template <typename T>
int MaximumBatchMulInPlaceWithAvx2(std::vector<ModularInt<T>>* in1,
                                   const std::vector<ModularInt<T>>& in2,
                                   const ModularIntParams<T>* params) {
  return 0;
}

// Specialize for 64 bits.
template <>
int MaximumBatchMulInPlaceWithAvx2(std::vector<ModularInt<uint64_t>>* in1,
                                   const std::vector<ModularInt<uint64_t>>& in2,
                                   const ModularIntParams<uint64_t>* params) {
  int i = 0;
  // If <= 50 bit modulus, use floating point instruction AVX2 batch operation.
  if (params->modulus >> 50 == 0) {
    __m256d p = _mm256_set1_pd(params->modulus);
    __m256d p_inv = _mm256_div_pd(_mm256_set1_pd(1), p);
    // Process in batch
    constexpr size_t batch_size = sizeof(__m256i) / sizeof(uint64_t);
    for (; i + batch_size <= in1->size(); i += batch_size) {
      // Load the next batch of ModularInt<Int>'s into 256-bit registers.
      const __m256i u = ::rlwe::internal::BatchLoadAvx2<ModularInt<uint64_t>>(
          in1->data() + i);
      const __m256i v =
          ::rlwe::internal::BatchLoadAvx2<ModularInt<uint64_t>>(in2.data() + i);
      // Compute the product modulo modulus.
      const __m256i uv =
          ::rlwe::internal::BatchMulModAvx2Uint50(u, v, p, p_inv);
      // Store the modular modular addition in in1->data() + i.
      ::rlwe::internal::BatchStoreAvx2<ModularInt<uint64_t>>(in1->data() + i,
                                                             uv);
    }
  } else {
    const __m256i simd_modulus =
        ::rlwe::internal::BatchSetAvx2<uint64_t>(params->modulus);
    // Compute the parameters for the Barrett reduction.
    int r;
    __m256i q;
    ::rlwe::internal::ComputeBarrettParametersAvx2<uint64_t>(&r, &q,
                                                             params->modulus);
    // Process in batch
    constexpr size_t batch_size = sizeof(__m256i) / sizeof(uint64_t);
    for (; i + batch_size <= in1->size(); i += batch_size) {
      // Load the next batch of ModularInt<Int>'s into 256-bit registers.
      const __m256i u = ::rlwe::internal::BatchLoadAvx2<ModularInt<uint64_t>>(
          in1->data() + i);
      const __m256i v =
          ::rlwe::internal::BatchLoadAvx2<ModularInt<uint64_t>>(in2.data() + i);
      // Compute the product modulo modulus.
      const __m256i uv =
          ::rlwe::internal::BatchMulModAvx2<uint64_t>(u, v, simd_modulus, r, q);
      // Store the modular modular addition in in1->data() + i.
      ::rlwe::internal::BatchStoreAvx2<ModularInt<uint64_t>>(in1->data() + i,
                                                             uv);
    }
  }
  return i;
}

template <typename T>
int MaximumBatchFusedMulAddInPlaceWithAvx2(
    std::vector<ModularInt<T>>* in1, const std::vector<ModularInt<T>>& in2,
    const std::vector<ModularInt<T>>& in3, const ModularIntParams<T>* params) {
  return 0;
}

template <>
int MaximumBatchFusedMulAddInPlaceWithAvx2(
    std::vector<ModularInt<uint64_t>>* in1,
    const std::vector<ModularInt<uint64_t>>& in2,
    const std::vector<ModularInt<uint64_t>>& in3,
    const ModularIntParams<uint64_t>* params) {
  int i = 0;
  // If <= 50 bit modulus, use floating point instruction AVX2 batch operation.
  if (params->modulus >> 50 == 0) {
    __m256i p = ::rlwe::internal::BatchSetAvx2<uint64_t>(params->modulus);
    __m256d p_double = _mm256_set1_pd(params->modulus);
    __m256d p_inv_double = _mm256_div_pd(_mm256_set1_pd(1), p_double);
    // Process in batch
    constexpr size_t batch_size = sizeof(__m256i) / sizeof(uint64_t);
    for (; i + batch_size <= in1->size(); i += batch_size) {
      // Load the next batch of ModularInt<Int>'s into 256-bit registers.
      const __m256i u = ::rlwe::internal::BatchLoadAvx2<ModularInt<uint64_t>>(
          in1->data() + i);
      const __m256i v =
          ::rlwe::internal::BatchLoadAvx2<ModularInt<uint64_t>>(in2.data() + i);
      const __m256i w =
          ::rlwe::internal::BatchLoadAvx2<ModularInt<uint64_t>>(in3.data() + i);
      // Compute the product modulo modulus.
      const __m256i vw =
          ::rlwe::internal::BatchMulModAvx2Uint50(v, w, p_double, p_inv_double);
      const __m256i result =
          ::rlwe::internal::BatchAddModAvx2<uint64_t>(u, vw, p);
      // Store the modular modular addition in in1->data() + i.
      ::rlwe::internal::BatchStoreAvx2<ModularInt<uint64_t>>(in1->data() + i,
                                                             result);
    }
  } else {
    const __m256i simd_modulus =
        ::rlwe::internal::BatchSetAvx2<uint64_t>(params->modulus);
    // Compute the parameters for the Barrett reduction.
    int r;
    __m256i q;
    ::rlwe::internal::ComputeBarrettParametersAvx2<uint64_t>(&r, &q,
                                                           params->modulus);
    // Process in batch
    constexpr size_t batch_size = sizeof(__m256i) / sizeof(uint64_t);
    for (; i + batch_size <= in1->size(); i += batch_size) {
      // Load the next batch of ModularInt<Int>'s into 256-bit registers.
      const __m256i u = ::rlwe::internal::BatchLoadAvx2<ModularInt<uint64_t>>(
          in1->data() + i);
      const __m256i v =
          ::rlwe::internal::BatchLoadAvx2<ModularInt<uint64_t>>(in2.data() + i);
      const __m256i w =
          ::rlwe::internal::BatchLoadAvx2<ModularInt<uint64_t>>(in3.data() + i);
      // Compute the product modulo modulus.
      const __m256i vw =
          ::rlwe::internal::BatchMulModAvx2<uint64_t>(v, w, simd_modulus, r, q);
      const __m256i result =
          ::rlwe::internal::BatchAddModAvx2<uint64_t>(u, vw, simd_modulus);
      // Store the modular modular addition in in1->data() + i.
      ::rlwe::internal::BatchStoreAvx2<ModularInt<uint64_t>>(in1->data() + i,
                                                             result);
    }
  }
  return i;
}
}  // namespace private_membership::internal

#endif

#endif  // COMMON_MODULAR_INT_BATCH_MODULAR_INT_AVX2_IMPL_H_
