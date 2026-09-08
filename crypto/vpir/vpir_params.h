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

#ifndef CRYPTO_VPIR_VPIR_PARAMS_H_
#define CRYPTO_VPIR_VPIR_PARAMS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/pir/pir_params.h"
#include "crypto/polynomial.h"
#include "crypto/proto/pir.pb.h"
#include "absl/numeric/bits.h"
#include "crypto/status_macros.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename CoeffType, typename ZCoeffType = CoeffType>
class VpirParams {
 public:
  static absl::StatusOr<VpirParams<CoeffType, ZCoeffType>> Create(
      const RlweParams<CoeffType>& rlwe_params, CoeffType plaintext_modulus,
      const GadgetParams& pack_gadget_params, int num_entries,
      int entry_size_multiple = 1, int z_interpolation_degree = 1,
      std::optional<GadgetParams> z_pack_gadget_params = std::nullopt,
      std::optional<GadgetParams> z_poly_eval_gadget_params = std::nullopt,
      std::optional<ZCoeffType> z_plaintext_modulus = std::nullopt,
      std::optional<RlweParams<ZCoeffType>> z_rlwe_params = std::nullopt,
      bool local_finalize = false);

  const RlweParams<CoeffType>& RlweParameters() const { return rlwe_params_; }
  CoeffType PlaintextModulus() const { return plaintext_modulus_; }
  const GadgetParams& PackGadgetParams() const { return pack_gadget_params_; }
  int NumEntries() const { return num_entries_; }
  int EntrySizeMultiple() const { return entry_size_multiple_; }
  int Ell() const { return entry_size_multiple_; }
  bool LocalFinalize() const { return local_finalize_; }

  const PirParams<ZCoeffType>& ZPirParams() const { return z_pir_params_; }

 private:
  VpirParams(const RlweParams<CoeffType>& rlwe_params,
             CoeffType plaintext_modulus,
             const GadgetParams& pack_gadget_params, int num_entries,
             int entry_size_multiple, PirParams<ZCoeffType> z_pir_params,
             bool local_finalize = false)
      : rlwe_params_(rlwe_params),
        plaintext_modulus_(plaintext_modulus),
        pack_gadget_params_(pack_gadget_params),
        num_entries_(num_entries),
        entry_size_multiple_(entry_size_multiple),
        z_pir_params_(std::move(z_pir_params)),
        local_finalize_(local_finalize) {}

  const RlweParams<CoeffType> rlwe_params_;
  const CoeffType plaintext_modulus_;
  const GadgetParams pack_gadget_params_;
  const int num_entries_;
  const int entry_size_multiple_;
  PirParams<ZCoeffType> z_pir_params_;
  const bool local_finalize_;
};

extern template class VpirParams<uint32_t, uint32_t>;
extern template class VpirParams<uint64_t, uint64_t>;
extern template class VpirParams<uint32_t, uint64_t>;
extern template class VpirParams<uint64_t, uint32_t>;

template <typename CoeffType, typename ZCoeffType = CoeffType>
absl::StatusOr<proto::PirParams> SerializeVpirParams(
    const VpirParams<CoeffType, ZCoeffType>& params,
    proto::PirParams::DbValueType db_value_type,
    proto::PirParams::MatCoeffType mat_coeff_type) {
  proto::PirParams proto;

  auto* rlwe_proto = proto.mutable_rlwe_params();
  const auto& rlwe_params = params.RlweParameters();
  rlwe_proto->set_log_degree(
      absl::bit_width(static_cast<uint32_t>(rlwe_params.Degree() - 1)));
  rlwe_proto->set_log_modulus(rlwe_params.LogModulus());
  rlwe_proto->set_log_modulus1_after_switch(
      rlwe_params.LogModulus1AfterSwitch());
  rlwe_proto->set_log_modulus2_after_switch(
      rlwe_params.LogModulus2AfterSwitch());
  rlwe_proto->set_variance(rlwe_params.Variance());

  proto.set_plaintext_modulus(static_cast<int64_t>(params.PlaintextModulus()));

  auto* pack_proto = proto.mutable_pack_gadget_params();
  pack_proto->set_log_digit(params.PackGadgetParams().log_digit);
  pack_proto->set_num_digits(params.PackGadgetParams().num_digits);

  auto* eval_proto = proto.mutable_poly_eval_gadget_params();
  eval_proto->set_log_digit(0);
  eval_proto->set_num_digits(0);

  proto.set_num_entries(params.NumEntries());
  proto.set_interpolation_degree(1);
  proto.set_entry_size_multiple(params.EntrySizeMultiple());

  proto.set_db_value_type(db_value_type);
  if constexpr (std::is_same_v<CoeffType, uint32_t>) {
    proto.set_coeff_type(proto::PirParams::COEFF_TYPE_UINT32);
  } else if constexpr (std::is_same_v<CoeffType, uint64_t>) {
    proto.set_coeff_type(proto::PirParams::COEFF_TYPE_UINT64);
  }
  proto.set_mat_coeff_type(mat_coeff_type);

  return proto;
}

template <typename CoeffType, typename ZCoeffType = CoeffType>
absl::StatusOr<VpirParams<CoeffType, ZCoeffType>> DeserializeVpirParams(
    const proto::PirParams& proto) {
  const auto& rlwe_proto = proto.rlwe_params();
  int degree = 1 << rlwe_proto.log_degree();
  CoeffType modulus = rlwe_proto.log_modulus() == 8 * sizeof(CoeffType)
                          ? 0
                          : (CoeffType{1} << rlwe_proto.log_modulus());
  CoeffType modulus1 =
      rlwe_proto.log_modulus1_after_switch() == 8 * sizeof(CoeffType)
          ? 0
          : (CoeffType{1} << rlwe_proto.log_modulus1_after_switch());
  CoeffType modulus2 =
      rlwe_proto.log_modulus2_after_switch() == 8 * sizeof(CoeffType)
          ? 0
          : (CoeffType{1} << rlwe_proto.log_modulus2_after_switch());

  ASSIGN_OR_RETURN(auto rlwe_params, RlweParams<CoeffType>::Create(
                                         degree, modulus, modulus1, modulus2,
                                         rlwe_proto.variance()));

  GadgetParams pack_gadget_params{
      .log_digit = proto.pack_gadget_params().log_digit(),
      .num_digits = proto.pack_gadget_params().num_digits(),
  };

  return VpirParams<CoeffType, ZCoeffType>::Create(
      rlwe_params, static_cast<CoeffType>(proto.plaintext_modulus()),
      pack_gadget_params, proto.num_entries(), proto.entry_size_multiple());
}

template <typename CoeffType>
struct VpirRequest {
  std::vector<Polynomial<CoeffType>> first_dimension_query;
  std::vector<Polynomial<CoeffType>> packing_key;
};

template <typename CoeffType>
struct VpirResponse {
  std::vector<Polynomial<CoeffType>> b_responses;
};

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_VPIR_VPIR_PARAMS_H_
