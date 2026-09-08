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

#ifndef CRYPTO_PIR_SERVER_PIR_PREPROCESSED_DATA_H_
#define CRYPTO_PIR_SERVER_PIR_PREPROCESSED_DATA_H_

#include <string>
#include <vector>

#include "crypto/file_util.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/pir_params.h"
#include "crypto/polynomial.h"
#include "crypto/proto/pir_server.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "crypto/status_macros.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// Holds preprocessing artifacts derived from the database.
template <typename DbValueType, typename CoeffType, typename MatCoeffType>
struct PirPreprocessedData {
  // Matrices resulting from interpolating and transposing the database shards.
  std::vector<Matrix<DbValueType>> db_matrices;

  // Preprocessed outputs used for efficient packing, for each shard.
  std::vector<std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
      preprocessed_outputs;

  // Public components of Rgsw encryption of polynomial variable, used for
  // efficient polynomial evaluation.
  std::vector<Polynomial<CoeffType>> second_dim_a;
};

template <typename DbValueType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<proto::PirPreprocessedData> SerializePirPreprocessedData(
    const PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>&
        preprocessed_data,
    const PirParams<CoeffType>& params) {
  proto::PirPreprocessedData proto;
  const int log_modulus = params.RlweParameters().LogModulus();

  for (const auto& matrix : preprocessed_data.db_matrices) {
    ASSIGN_OR_RETURN(auto matrix_proto, matrix.ToProto());
    *proto.add_db_matrices() = std::move(matrix_proto);
  }

  for (const auto& shard_outputs : preprocessed_data.preprocessed_outputs) {
    auto* shard_proto = proto.add_preprocessed_outputs();
    for (const auto& output : shard_outputs) {
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
  }

  for (const auto& poly : preprocessed_data.second_dim_a) {
    ASSIGN_OR_RETURN(auto poly_proto, poly.ToProto(log_modulus));
    *proto.add_second_dim_a() = std::move(poly_proto);
  }

  return proto;
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>>
DeserializePirPreprocessedData(const proto::PirPreprocessedData& proto,
                               const PirParams<CoeffType>& params) {
  const int log_modulus = params.RlweParameters().LogModulus();
  const int degree = params.RlweParameters().Degree();

  std::vector<Matrix<DbValueType>> db_matrices;
  db_matrices.reserve(proto.db_matrices_size());
  for (const auto& matrix_proto : proto.db_matrices()) {
    ASSIGN_OR_RETURN(auto matrix, Matrix<DbValueType>::FromProto(matrix_proto));
    db_matrices.push_back(std::move(matrix));
  }

  std::vector<std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
      preprocessed_outputs;
  preprocessed_outputs.reserve(proto.preprocessed_outputs_size());
  for (const auto& shard_proto : proto.preprocessed_outputs()) {
    std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
        shard_outputs;
    shard_outputs.reserve(shard_proto.outputs_size());
    for (const auto& output_proto : shard_proto.outputs()) {
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
      shard_outputs.push_back(
          PreprocessMatrixPackOutput<CoeffType, MatCoeffType>{
              .a_tilde_agg = std::move(a_tilde_agg),
              .matrix = std::move(matrix),
              .t_vec_h = std::move(t_vec_h),
          });
    }
    preprocessed_outputs.push_back(std::move(shard_outputs));
  }

  std::vector<Polynomial<CoeffType>> second_dim_a;
  second_dim_a.reserve(proto.second_dim_a_size());
  for (const auto& poly_proto : proto.second_dim_a()) {
    ASSIGN_OR_RETURN(auto poly, Polynomial<CoeffType>::CreateFromProto(
                                    poly_proto, degree, log_modulus));
    second_dim_a.push_back(std::move(poly));
  }

  return PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>{
      .db_matrices = std::move(db_matrices),
      .preprocessed_outputs = std::move(preprocessed_outputs),
      .second_dim_a = std::move(second_dim_a),
  };
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType>
absl::Status SavePirPreprocessedDataToFile(
    const PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>&
        preprocessed_data,
    const PirParams<CoeffType>& params, absl::string_view file_path) {
  ASSIGN_OR_RETURN(auto proto,
                   SerializePirPreprocessedData(preprocessed_data, params));
  return file_util::SetFileContents(file_path, proto.SerializeAsString());
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>>
LoadPirPreprocessedDataFromFile(const PirParams<CoeffType>& params,
                                absl::string_view file_path) {
  std::string contents;
  RETURN_IF_ERROR(file_util::GetFileContents(file_path, &contents));
  proto::PirPreprocessedData proto;
  if (!proto.ParseFromString(contents)) {
    return absl::InvalidArgumentError(
        "Failed to parse PirPreprocessedData from file.");
  }
  return DeserializePirPreprocessedData<DbValueType, CoeffType, MatCoeffType>(
      proto, params);
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType>
absl::Status SavePirPreprocessedDataToDirectory(
    const PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>&
        preprocessed_data,
    const PirParams<CoeffType>& params, absl::string_view dir_path) {
  RETURN_IF_ERROR(file_util::RecursivelyCreateDir(dir_path));

  // 1. Serialize and write metadata.
  proto::PirPreprocessedMetadata metadata_proto;
  const int log_modulus = params.RlweParameters().LogModulus();
  for (const auto& poly : preprocessed_data.second_dim_a) {
    ASSIGN_OR_RETURN(auto poly_proto, poly.ToProto(log_modulus));
    *metadata_proto.add_second_dim_a() = std::move(poly_proto);
  }
  std::string metadata_path = file_util::JoinPath(dir_path, "metadata.bin");
  RETURN_IF_ERROR(file_util::SetFileContents(
      metadata_path, metadata_proto.SerializeAsString()));

  // 2. Serialize and write each shard.
  const int entry_size_multiple = params.EntrySizeMultiple();
  if (preprocessed_data.db_matrices.size() != entry_size_multiple ||
      preprocessed_data.preprocessed_outputs.size() != entry_size_multiple) {
    return absl::InvalidArgumentError(
        "SavePirPreprocessedDataToDirectory: Shard size mismatch with "
        "EntrySizeMultiple.");
  }

  for (int i = 0; i < entry_size_multiple; ++i) {
    proto::PirPreprocessedShard shard_proto;

    // Serialize database matrix for shard i
    ASSIGN_OR_RETURN(auto matrix_proto,
                     preprocessed_data.db_matrices[i].ToProto());
    *shard_proto.mutable_db_matrix() = std::move(matrix_proto);

    // Serialize preprocess outputs for shard i
    const auto& shard_outputs = preprocessed_data.preprocessed_outputs[i];
    for (const auto& output : shard_outputs) {
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
    }

    std::string shard_path =
        file_util::JoinPath(dir_path, absl::StrCat("shard_", i, ".bin"));
    RETURN_IF_ERROR(file_util::SetFileContents(
        shard_path, shard_proto.SerializeAsString()));
  }

  return absl::OkStatus();
}

template <typename DbValueType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>>
LoadPirPreprocessedDataFromDirectory(const PirParams<CoeffType>& params,
                                     absl::string_view dir_path) {
  const int log_modulus = params.RlweParameters().LogModulus();
  const int degree = params.RlweParameters().Degree();
  const int entry_size_multiple = params.EntrySizeMultiple();

  // 1. Read metadata
  std::string metadata_content;
  std::string metadata_path = file_util::JoinPath(dir_path, "metadata.bin");
  RETURN_IF_ERROR(
      file_util::GetFileContents(metadata_path, &metadata_content));

  proto::PirPreprocessedMetadata metadata_proto;
  if (!metadata_proto.ParseFromString(metadata_content)) {
    return absl::InvalidArgumentError(
        "Failed to parse PirPreprocessedMetadata.");
  }

  std::vector<Polynomial<CoeffType>> second_dim_a;
  second_dim_a.reserve(metadata_proto.second_dim_a_size());
  for (const auto& poly_proto : metadata_proto.second_dim_a()) {
    ASSIGN_OR_RETURN(auto poly, Polynomial<CoeffType>::CreateFromProto(
                                    poly_proto, degree, log_modulus));
    second_dim_a.push_back(std::move(poly));
  }

  // 2. Read each shard
  std::vector<Matrix<DbValueType>> db_matrices;
  db_matrices.reserve(entry_size_multiple);

  std::vector<std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
      preprocessed_outputs;
  preprocessed_outputs.reserve(entry_size_multiple);

  for (int i = 0; i < entry_size_multiple; ++i) {
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

    ASSIGN_OR_RETURN(auto matrix,
                     Matrix<DbValueType>::FromProto(shard_proto.db_matrix()));
    db_matrices.push_back(std::move(matrix));

    std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>
        shard_outputs;
    shard_outputs.reserve(shard_proto.outputs_size());
    for (const auto& output_proto : shard_proto.outputs()) {
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
      shard_outputs.push_back(
          PreprocessMatrixPackOutput<CoeffType, MatCoeffType>{
              .a_tilde_agg = std::move(a_tilde_agg),
              .matrix = std::move(matrix),
              .t_vec_h = std::move(t_vec_h),
          });
    }
    preprocessed_outputs.push_back(std::move(shard_outputs));
  }

  return PirPreprocessedData<DbValueType, CoeffType, MatCoeffType>{
      .db_matrices = std::move(db_matrices),
      .preprocessed_outputs = std::move(preprocessed_outputs),
      .second_dim_a = std::move(second_dim_a),
  };
}

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_PIR_SERVER_PIR_PREPROCESSED_DATA_H_
