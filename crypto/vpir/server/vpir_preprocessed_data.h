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

#ifndef CRYPTO_VPIR_SERVER_VPIR_PREPROCESSED_DATA_H_
#define CRYPTO_VPIR_SERVER_VPIR_PREPROCESSED_DATA_H_

#include <cstddef>
#include <string>
#include <vector>

#include "crypto/file_util.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/server/pir_preprocessed_data.h"
#include "crypto/polynomial.h"
#include "crypto/vpir/vpir_params.h"
#include "crypto/proto/pir_server.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "crypto/status_macros.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename DbValueType, typename CoeffType, typename MatCoeffType,
          typename ZDbValueType = DbValueType, typename ZCoeffType = CoeffType,
          typename ZMatCoeffType = MatCoeffType>
struct VpirPreprocessedData {
  Matrix<DbValueType> db_matrix;
  std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
      preprocessed_outputs;
  Matrix<CoeffType> hint_matrix;
  std::string prng_seed;
  PirPreprocessedData<ZDbValueType, ZCoeffType, ZMatCoeffType>
      z_preprocessed_data;
};

template <typename DbValueType, typename CoeffType, typename MatCoeffType,
          typename ZDbValueType = DbValueType, typename ZCoeffType = CoeffType,
          typename ZMatCoeffType = MatCoeffType>
absl::StatusOr<proto::PirPreprocessedData> SerializeVpirPreprocessedData(
    const VpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                               ZDbValueType, ZCoeffType, ZMatCoeffType>&
        preprocessed_data,
    const VpirParams<CoeffType, ZCoeffType>& params) {
  proto::PirPreprocessedData proto;
  const int log_modulus = params.RlweParameters().LogModulus();

  ASSIGN_OR_RETURN(auto matrix_proto, preprocessed_data.db_matrix.ToProto());
  *proto.add_db_matrices() = std::move(matrix_proto);

  for (const auto& output : preprocessed_data.preprocessed_outputs) {
    auto* shard_proto = proto.add_preprocessed_outputs();
    auto* output_proto = shard_proto->add_outputs();
    ASSIGN_OR_RETURN(auto a_tilde_agg_proto,
                     output.a_tilde_agg.ToProto(log_modulus));
    *output_proto->mutable_a_tilde_agg() = std::move(a_tilde_agg_proto);

    ASSIGN_OR_RETURN(auto mat_proto, output.matrix.ToProto());
    *output_proto->mutable_matrix() = std::move(mat_proto);

    for (const auto& t_h : output.t_vec_h) {
      ASSIGN_OR_RETURN(auto t_h_proto, t_h.ToProto(log_modulus));
      *output_proto->add_t_vec_h() = std::move(t_h_proto);
    }
  }

  return proto;
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType,
          typename ZDbValueType = DbValueType, typename ZCoeffType = CoeffType,
          typename ZMatCoeffType = MatCoeffType>
absl::StatusOr<VpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                                    ZDbValueType, ZCoeffType, ZMatCoeffType>>
DeserializeVpirPreprocessedData(
    const proto::PirPreprocessedData& proto,
    const VpirParams<CoeffType, ZCoeffType>& params) {
  const int log_modulus = params.RlweParameters().LogModulus();
  const int degree = params.RlweParameters().Degree();

  if (proto.db_matrices_size() == 0) {
    return absl::InvalidArgumentError(
        "Expected at least 1 db_matrix in proto.");
  }
  ASSIGN_OR_RETURN(auto db_matrix,
                   Matrix<DbValueType>::FromProto(proto.db_matrices(0)));

  std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
      preprocessed_outputs;
  preprocessed_outputs.reserve(proto.preprocessed_outputs_size());
  for (const auto& shard_proto : proto.preprocessed_outputs()) {
    if (shard_proto.outputs_size() != 1) {
      return absl::InvalidArgumentError(
          "Expected 1 output per entry block in VPIR.");
    }
    const auto& output_proto = shard_proto.outputs(0);
    ASSIGN_OR_RETURN(auto a_tilde_agg,
                     Polynomial<CoeffType>::CreateFromProto(
                         output_proto.a_tilde_agg(), degree, log_modulus));
    ASSIGN_OR_RETURN(auto matrix,
                     Matrix<MatCoeffType>::FromProto(output_proto.matrix()));
    std::vector<Polynomial<CoeffType>> t_vec_h;
    t_vec_h.reserve(output_proto.t_vec_h_size());
    for (const auto& t_h_proto : output_proto.t_vec_h()) {
      ASSIGN_OR_RETURN(auto t_h, Polynomial<CoeffType>::CreateFromProto(
                                     t_h_proto, degree, log_modulus));
      t_vec_h.push_back(std::move(t_h));
    }
    preprocessed_outputs.push_back(
        PreprocessMatrixPackOutput<CoeffType, MatCoeffType>{
            .a_tilde_agg = std::move(a_tilde_agg),
            .matrix = std::move(matrix),
            .t_vec_h = std::move(t_vec_h),
        });
  }

  return VpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                              ZDbValueType, ZCoeffType, ZMatCoeffType>{
      .db_matrix = std::move(db_matrix),
      .preprocessed_outputs = std::move(preprocessed_outputs),
  };
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType,
          typename ZDbValueType = DbValueType, typename ZCoeffType = CoeffType,
          typename ZMatCoeffType = MatCoeffType>
absl::Status SaveVpirPreprocessedDataToFile(
    const VpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                               ZDbValueType, ZCoeffType, ZMatCoeffType>&
        preprocessed_data,
    const VpirParams<CoeffType, ZCoeffType>& params,
    absl::string_view file_path) {
  ASSIGN_OR_RETURN(auto proto,
                   SerializeVpirPreprocessedData(preprocessed_data, params));
  return file_util::SetFileContents(file_path, proto.SerializeAsString());
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType,
          typename ZDbValueType = DbValueType, typename ZCoeffType = CoeffType,
          typename ZMatCoeffType = MatCoeffType>
absl::StatusOr<VpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                                    ZDbValueType, ZCoeffType, ZMatCoeffType>>
LoadVpirPreprocessedDataFromFile(
    const VpirParams<CoeffType, ZCoeffType>& params,
    absl::string_view file_path) {
  std::string contents;
  RETURN_IF_ERROR(file_util::GetFileContents(file_path, &contents));
  proto::PirPreprocessedData proto;
  if (!proto.ParseFromString(contents)) {
    return absl::InvalidArgumentError(
        "Failed to parse PirPreprocessedData from file.");
  }
  return DeserializeVpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                                         ZDbValueType, ZCoeffType,
                                         ZMatCoeffType>(proto, params);
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType,
          typename ZDbValueType = DbValueType, typename ZCoeffType = CoeffType,
          typename ZMatCoeffType = MatCoeffType>
absl::Status SaveVpirPreprocessedDataToDirectory(
    const VpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                               ZDbValueType, ZCoeffType, ZMatCoeffType>&
        preprocessed_data,
    const VpirParams<CoeffType, ZCoeffType>& params,
    absl::string_view dir_path) {
  RETURN_IF_ERROR(file_util::RecursivelyCreateDir(dir_path));

  proto::PirPreprocessedMetadata metadata_proto;
  std::string metadata_path = file_util::JoinPath(dir_path, "metadata.bin");
  RETURN_IF_ERROR(file_util::SetFileContents(
      metadata_path, metadata_proto.SerializeAsString()));

  const int ell = params.Ell();
  if (preprocessed_data.preprocessed_outputs.size() !=
      static_cast<size_t>(ell)) {
    return absl::InvalidArgumentError(
        "SaveVpirPreprocessedDataToDirectory: Preprocessed outputs size "
        "mismatch with Ell.");
  }

  const int log_modulus = params.RlweParameters().LogModulus();
  for (int i = 0; i < ell; ++i) {
    proto::PirPreprocessedShard shard_proto;

    if (i == 0) {
      ASSIGN_OR_RETURN(auto matrix_proto,
                       preprocessed_data.db_matrix.ToProto());
      *shard_proto.mutable_db_matrix() = std::move(matrix_proto);
    }

    const auto& output = preprocessed_data.preprocessed_outputs[i];
    auto* output_proto = shard_proto.add_outputs();
    ASSIGN_OR_RETURN(auto a_tilde_agg_proto,
                     output.a_tilde_agg.ToProto(log_modulus));
    *output_proto->mutable_a_tilde_agg() = std::move(a_tilde_agg_proto);

    ASSIGN_OR_RETURN(auto mat_proto, output.matrix.ToProto());
    *output_proto->mutable_matrix() = std::move(mat_proto);

    for (const auto& t_h : output.t_vec_h) {
      ASSIGN_OR_RETURN(auto t_h_proto, t_h.ToProto(log_modulus));
      *output_proto->add_t_vec_h() = std::move(t_h_proto);
    }

    std::string shard_path =
        file_util::JoinPath(dir_path, absl::StrCat("shard_", i, ".bin"));
    RETURN_IF_ERROR(file_util::SetFileContents(
        shard_path, shard_proto.SerializeAsString()));
  }

  return absl::OkStatus();
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType,
          typename ZDbValueType = DbValueType, typename ZCoeffType = CoeffType,
          typename ZMatCoeffType = MatCoeffType>
absl::StatusOr<VpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                                    ZDbValueType, ZCoeffType, ZMatCoeffType>>
LoadVpirPreprocessedDataFromDirectory(
    const VpirParams<CoeffType, ZCoeffType>& params,
    absl::string_view dir_path) {
  const int log_modulus = params.RlweParameters().LogModulus();
  const int degree = params.RlweParameters().Degree();
  const int ell = params.Ell();

  std::string metadata_content;
  std::string metadata_path = file_util::JoinPath(dir_path, "metadata.bin");
  RETURN_IF_ERROR(
      file_util::GetFileContents(metadata_path, &metadata_content));

  proto::PirPreprocessedMetadata metadata_proto;
  if (!metadata_proto.ParseFromString(metadata_content)) {
    return absl::InvalidArgumentError(
        "Failed to parse PirPreprocessedMetadata.");
  }

  Matrix<DbValueType> db_matrix;
  std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
      preprocessed_outputs;
  preprocessed_outputs.reserve(ell);

  for (int i = 0; i < ell; ++i) {
    std::string shard_content;
    std::string shard_path =
        file_util::JoinPath(dir_path, absl::StrCat("shard_", i, ".bin"));
    RETURN_IF_ERROR(
        file_util::GetFileContents(shard_path, &shard_content));

    proto::PirPreprocessedShard shard_proto;
    if (!shard_proto.ParseFromString(shard_content)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to parse PirPreprocessedShard ", i));
    }

    if (i == 0) {
      ASSIGN_OR_RETURN(db_matrix,
                       Matrix<DbValueType>::FromProto(shard_proto.db_matrix()));
    }

    if (shard_proto.outputs_size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected exactly 1 output in shard ", i));
    }
    const auto& output_proto = shard_proto.outputs(0);
    ASSIGN_OR_RETURN(auto a_tilde_agg,
                     Polynomial<CoeffType>::CreateFromProto(
                         output_proto.a_tilde_agg(), degree, log_modulus));
    ASSIGN_OR_RETURN(auto matrix_pack,
                     Matrix<MatCoeffType>::FromProto(output_proto.matrix()));
    std::vector<Polynomial<CoeffType>> t_vec_h;
    t_vec_h.reserve(output_proto.t_vec_h_size());
    for (const auto& t_h_proto : output_proto.t_vec_h()) {
      ASSIGN_OR_RETURN(auto t_h, Polynomial<CoeffType>::CreateFromProto(
                                     t_h_proto, degree, log_modulus));
      t_vec_h.push_back(std::move(t_h));
    }
    preprocessed_outputs.push_back(
        PreprocessMatrixPackOutput<CoeffType, MatCoeffType>{
            .a_tilde_agg = std::move(a_tilde_agg),
            .matrix = std::move(matrix_pack),
            .t_vec_h = std::move(t_vec_h),
        });
  }

  return VpirPreprocessedData<DbValueType, CoeffType, MatCoeffType,
                              ZDbValueType, ZCoeffType, ZMatCoeffType>{
      .db_matrix = std::move(db_matrix),
      .preprocessed_outputs = std::move(preprocessed_outputs),
  };
}

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_VPIR_SERVER_VPIR_PREPROCESSED_DATA_H_
