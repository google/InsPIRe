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

#include "crypto/pir/pir_params.h"

#include <cstdint>

#include "crypto/encryption.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename CoeffType>
absl::StatusOr<PirParams<CoeffType>> PirParams<CoeffType>::Create(
    const RlweParams<CoeffType>& rlwe_params, const CoeffType plaintext_modulus,
    const GadgetParams& pack_gadget_params,
    const GadgetParams& poly_eval_gadget_params, const int num_entries,
    const int interpolation_degree, const int entry_size_multiple) {
  if (num_entries <= 0) {
    return absl::InvalidArgumentError("num_entries must be strictly positive.");
  }

  if (entry_size_multiple <= 0) {
    return absl::InvalidArgumentError(
        "entry_size_multiple must be strictly positive.");
  }

  if (interpolation_degree <= 0 ||
      absl::has_single_bit(static_cast<unsigned int>(interpolation_degree)) !=
          1) {
    return absl::InvalidArgumentError(
        "interpolation_degree must be a power of two.");
  }

  if (num_entries % interpolation_degree != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("interpolation_degree (", interpolation_degree,
                     ") must divide num_entries (", num_entries, ")."));
  }

  if (interpolation_degree > 2 * rlwe_params.Degree()) {
    return absl::InvalidArgumentError(
        absl::StrCat("interpolation_degree (", interpolation_degree,
                     ") must be less than or equal to 2 * ring degree (",
                     2 * rlwe_params.Degree(), ")."));
  }

  return PirParams<CoeffType>(rlwe_params, plaintext_modulus,
                              pack_gadget_params, poly_eval_gadget_params,
                              num_entries, interpolation_degree,
                              entry_size_multiple);
}

template class PirParams<uint32_t>;
template class PirParams<uint64_t>;

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
