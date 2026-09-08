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

#include "crypto/vpir/vpir_params.h"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include "crypto/encryption.h"
#include "crypto/pir/pir_params.h"
#include "absl/status/status.h"
#include "crypto/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename CoeffType, typename ZCoeffType>
absl::StatusOr<VpirParams<CoeffType, ZCoeffType>>
VpirParams<CoeffType, ZCoeffType>::Create(
    const RlweParams<CoeffType>& rlwe_params, const CoeffType plaintext_modulus,
    const GadgetParams& pack_gadget_params, const int num_entries,
    const int entry_size_multiple, int z_interpolation_degree,
    std::optional<GadgetParams> z_pack_gadget_params,
    std::optional<GadgetParams> z_poly_eval_gadget_params,
    std::optional<ZCoeffType> z_plaintext_modulus,
    std::optional<RlweParams<ZCoeffType>> z_rlwe_params,
    bool local_finalize) {
  if (num_entries <= 0) {
    return absl::InvalidArgumentError("num_entries must be strictly positive.");
  }

  if (entry_size_multiple <= 0) {
    return absl::InvalidArgumentError(
        "entry_size_multiple must be strictly positive.");
  }

  if (z_interpolation_degree > 1 && !z_poly_eval_gadget_params.has_value()) {
    return absl::InvalidArgumentError(
        "z_poly_eval_gadget_params must be provided if z_interpolation_degree "
        "> 1.");
  }

  const int rows = num_entries;
  if (rows % rlwe_params.Degree() != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "num_entries (", rows, ") must be divisible by the ring degree (",
        rlwe_params.Degree(), ")."));
  }

  const int z_num_entries = entry_size_multiple * rlwe_params.Degree();
  const int z_entry_size_multiple = num_entries / rlwe_params.Degree();

  ZCoeffType default_z_plaintext_modulus = 1073741824;  // 2^30 for 64-bit
  if constexpr (sizeof(ZCoeffType) == 4) {
    default_z_plaintext_modulus = 8192;  // 2^13 for 32-bit
  }
  ZCoeffType actual_z_plaintext_modulus = z_plaintext_modulus.has_value()
                                             ? *z_plaintext_modulus
                                             : default_z_plaintext_modulus;

  const GadgetParams& actual_z_pack_gadget_params =
      z_pack_gadget_params.has_value() ? *z_pack_gadget_params
                                       : pack_gadget_params;
  const GadgetParams& actual_z_poly_eval_gadget_params =
      z_poly_eval_gadget_params.has_value() ? *z_poly_eval_gadget_params
                                            : GadgetParams{};

  auto get_actual_z_rlwe_params =
      [&]() -> absl::StatusOr<RlweParams<ZCoeffType>> {
    if (z_rlwe_params.has_value()) {
      return *z_rlwe_params;
    }
    if constexpr (std::is_same_v<CoeffType, ZCoeffType>) {
      return rlwe_params;
    } else {
      return RlweParams<ZCoeffType>::Create(
          rlwe_params.Degree(),
          static_cast<ZCoeffType>(rlwe_params.Modulus()),
          static_cast<ZCoeffType>(rlwe_params.Modulus1AfterSwitch()),
          static_cast<ZCoeffType>(rlwe_params.Modulus2AfterSwitch()),
          rlwe_params.Variance());
    }
  };
  ASSIGN_OR_RETURN(auto actual_z_rlwe_params, get_actual_z_rlwe_params());

  ASSIGN_OR_RETURN(auto z_pir_params,
                   PirParams<ZCoeffType>::Create(
                       actual_z_rlwe_params, actual_z_plaintext_modulus,
                       actual_z_pack_gadget_params,
                       actual_z_poly_eval_gadget_params, z_num_entries,
                       z_interpolation_degree, z_entry_size_multiple));

  return VpirParams<CoeffType, ZCoeffType>(
      rlwe_params, plaintext_modulus, pack_gadget_params, num_entries,
      entry_size_multiple, std::move(z_pir_params), local_finalize);
}

template class VpirParams<uint32_t, uint32_t>;
template class VpirParams<uint64_t, uint64_t>;
template class VpirParams<uint32_t, uint64_t>;
template class VpirParams<uint64_t, uint32_t>;

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
