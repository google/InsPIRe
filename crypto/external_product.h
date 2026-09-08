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

#ifndef CRYPTO_EXTERNAL_PRODUCT_H_
#define CRYPTO_EXTERNAL_PRODUCT_H_

#include "crypto/encryption.h"
#include "crypto/polynomial_fft.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// Computes the external product between an RLWE ciphertext and an RGSW
// ciphertext. The resulting RLWE ciphertext encrypts the product of the two
// underlying messages.
template <typename CoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>> ExternalProduct(
    const RlweParams<CoeffType>& params, const GadgetParams& gadget_params,
    const RlweCiphertext<CoeffType>& ct,
    const RgswCiphertext<CoeffType>& rgsw_ct, FftContext& ctx);

extern template absl::StatusOr<RlweCiphertext<uint32_t>>
ExternalProduct<uint32_t>(const RlweParams<uint32_t>&, const GadgetParams&,
                          const RlweCiphertext<uint32_t>&,
                          const RgswCiphertext<uint32_t>&, FftContext&);
extern template absl::StatusOr<RlweCiphertext<uint64_t>>
ExternalProduct<uint64_t>(const RlweParams<uint64_t>&, const GadgetParams&,
                          const RlweCiphertext<uint64_t>&,
                          const RgswCiphertext<uint64_t>&, FftContext&);

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_EXTERNAL_PRODUCT_H_
