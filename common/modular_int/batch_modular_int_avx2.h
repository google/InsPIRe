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

#ifndef COMMON_MODULAR_INT_BATCH_MODULAR_INT_AVX2_H_
#define COMMON_MODULAR_INT_BATCH_MODULAR_INT_AVX2_H_

#include "common/modular_int/modular_int.h"

#if defined(__AVX2__) && defined(__FMA__)
#define MODULAR_INT_AVX2_ENABLED true
#else
#define MODULAR_INT_AVX2_ENABLED false
#endif

#if MODULAR_INT_AVX2_ENABLED
#include "common/modular_int/batch_operations_avx2.h"
#endif

#if MODULAR_INT_AVX2_ENABLED

namespace rlwe::internal {

template <>
__m256i BatchSetAvx2<private_membership::ModularInt<uint16_t>>(
    const private_membership::ModularInt<uint16_t>& a);
template <>
__m256i BatchSetAvx2<private_membership::ModularInt<uint32_t>>(
    const private_membership::ModularInt<uint32_t>& a);
template <>
__m256i BatchSetAvx2<private_membership::ModularInt<uint64_t>>(
    const private_membership::ModularInt<uint64_t>& a);

template <>
__m256i BatchLoadAvx2<private_membership::ModularInt<uint16_t>>(
    const private_membership::ModularInt<uint16_t>* ptr);
template <>
__m256i BatchLoadAvx2<private_membership::ModularInt<uint32_t>>(
    const private_membership::ModularInt<uint32_t>* ptr);
template <>
__m256i BatchLoadAvx2<private_membership::ModularInt<uint64_t>>(
    const private_membership::ModularInt<uint64_t>* ptr);

template <>
void BatchStoreAvx2<private_membership::ModularInt<uint16_t>>(
    private_membership::ModularInt<uint16_t>* ptr, const __m256i& u);
template <>
void BatchStoreAvx2<private_membership::ModularInt<uint32_t>>(
    private_membership::ModularInt<uint32_t>* ptr, const __m256i& u);
template <>
void BatchStoreAvx2<private_membership::ModularInt<uint64_t>>(
    private_membership::ModularInt<uint64_t>* ptr, const __m256i& u);
}  // namespace rlwe::internal

namespace private_membership::internal {

template <typename T>
int MaximumBatchAddInPlaceWithAvx2(std::vector<ModularInt<T>>* in1,
                                   const std::vector<ModularInt<T>>& in2,
                                   const ModularIntParams<T>* params);

template <typename T>
int MaximumBatchAddInPlaceWithAvx2(std::vector<ModularInt<T>>* in1,
                                   const ModularInt<T>& in2,
                                   const ModularIntParams<T>* params);

template <typename T>
int MaximumBatchSubInPlaceWithAvx2(std::vector<ModularInt<T>>* in1,
                                   const std::vector<ModularInt<T>>& in2,
                                   const ModularIntParams<T>* params);

template <typename T>
int MaximumBatchSubInPlaceWithAvx2(std::vector<ModularInt<T>>* in1,
                                   const ModularInt<T>& in2,
                                   const ModularIntParams<T>* params);

template <typename T>
int MaximumBatchMulInPlaceWithAvx2(std::vector<ModularInt<T>>* in1,
                                   const std::vector<ModularInt<T>>& in2,
                                   const ModularIntParams<T>* params);

template <typename T>
int MaximumBatchFusedMulAddInPlaceWithAvx2(
    std::vector<ModularInt<T>>* in1, const std::vector<ModularInt<T>>& in2,
    const std::vector<ModularInt<T>>& in3, const ModularIntParams<T>* params);

}  // namespace private_membership::internal

#endif

#include "common/modular_int/batch_modular_int_avx2_impl.h"

#endif  // COMMON_MODULAR_INT_BATCH_MODULAR_INT_AVX2_H_
