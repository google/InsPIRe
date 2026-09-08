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

#ifndef COMMON_MODULAR_INT_BATCH_MODULAR_INT_HARDWARE_ACCELERATION_H_
#define COMMON_MODULAR_INT_BATCH_MODULAR_INT_HARDWARE_ACCELERATION_H_

#include "common/modular_int/modular_int.h"

namespace private_membership::internal {

template <typename T>
int MaximumBatchAddInPlaceWithHardware(std::vector<ModularInt<T>>* in1,
                                       const std::vector<ModularInt<T>>& in2,
                                       const ModularIntParams<T>* params);

template <typename T>
int MaximumBatchAddInPlaceWithHardware(std::vector<ModularInt<T>>* in1,
                                       const ModularInt<T>& in2,
                                       const ModularIntParams<T>* params);

template <typename T>
int MaximumBatchSubInPlaceWithHardware(std::vector<ModularInt<T>>* in1,
                                       const std::vector<ModularInt<T>>& in2,
                                       const ModularIntParams<T>* params);
template <typename T>
int MaximumBatchSubInPlaceWithHardware(std::vector<ModularInt<T>>* in1,
                                       const ModularInt<T>& in2,
                                       const ModularIntParams<T>* params);
template <typename T>
int MaximumBatchMulInPlaceWithHardware(std::vector<ModularInt<T>>* in1,
                                       const std::vector<ModularInt<T>>& in2,
                                       const ModularIntParams<T>* params);

template <typename T>
int MaximumBatchFusedMulAddInPlaceWithHardware(
    std::vector<ModularInt<T>>* in1, const std::vector<ModularInt<T>>& in2,
    const std::vector<ModularInt<T>>& in3, const ModularIntParams<T>* params);

}  // namespace private_membership::internal

#include "common/modular_int/batch_modular_int_hardware_acceleration_impl.h"

#endif  // COMMON_MODULAR_INT_BATCH_MODULAR_INT_HARDWARE_ACCELERATION_H_
