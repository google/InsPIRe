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

#include "crypto/pir/server/pir_server.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/external_product.h"
#include "crypto/file_util.h"
#include "crypto/lwes_to_rlwe.h"
#include "crypto/matrix.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_preprocessed_data.h"
#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "crypto/status_macros.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
    const PirParams<CoeffType>& params,
    PirPreprocessedData<DbDataType, CoeffType, MatCoeffType>
        preprocessed_data) {
  const int t = params.InterpolationDegree();
  const int d = params.RlweParameters().Degree();
  const int r = params.InterpolatedRows();
  const int entry_size_multiple = params.EntrySizeMultiple();
  if (preprocessed_data.db_matrices.size() != entry_size_multiple) {
    return absl::InvalidArgumentError("db_matrices size mismatch.");
  }
  const int db_cols = preprocessed_data.db_matrices[0].Cols();
  if (db_cols < r) {
    return absl::InvalidArgumentError(
        "DB matrix must have at least N/t columns.");
  }
  for (int k = 0; k < entry_size_multiple; ++k) {
    if (preprocessed_data.db_matrices[k].Rows() != t * d) {
      return absl::InvalidArgumentError("DB matrix must have td rows.");
    }
    if (preprocessed_data.db_matrices[k].Cols() != db_cols) {
      return absl::InvalidArgumentError(
          "All DB matrices must have the same number of columns.");
    }
  }

  if (preprocessed_data.preprocessed_outputs.size() != entry_size_multiple) {
    return absl::InvalidArgumentError("preprocessed_outputs size mismatch.");
  }
  for (int k = 0; k < entry_size_multiple; ++k) {
    if (preprocessed_data.preprocessed_outputs[k].size() != t) {
      return absl::InvalidArgumentError(
          "Preprocessed outputs must be of length t.");
    }
  }

  if (t > 1 && preprocessed_data.second_dim_a.size() !=
                   2 * params.PolyEvalGadgetParams().num_digits) {
    return absl::InvalidArgumentError(
        "second_dim_query_a must have length 2 * "
        "PolyEvalGadgetParams().num_digits.");
  }
  ASSIGN_OR_RETURN(
      Context ctx,
      Context::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));

  // Combine db_matrices
  ASSIGN_OR_RETURN(
      auto combined_db_matrix,
      Matrix<DbDataType>::VerticalConcatenate(preprocessed_data.db_matrices));

  // Combine pack matrices
  std::vector<Matrix<MatCoeffType>> pack_matrices;
  pack_matrices.reserve(entry_size_multiple * t);
  for (int k = 0; k < entry_size_multiple; ++k) {
    for (int i = 0; i < t; ++i) {
      pack_matrices.push_back(
          std::move(preprocessed_data.preprocessed_outputs[k][i].matrix));
      // Replace with empty matrix to avoid duplication
      preprocessed_data.preprocessed_outputs[k][i].matrix =
          Matrix<MatCoeffType>();
    }
  }
  ASSIGN_OR_RETURN(auto combined_pack_matrix,
                   Matrix<MatCoeffType>::VerticalConcatenate(pack_matrices));

  return absl::WrapUnique(new PirServer<DbDataType, CoeffType, MatCoeffType>(
      params, std::move(combined_db_matrix), std::move(combined_pack_matrix),
      std::move(preprocessed_data.preprocessed_outputs),
      std::move(preprocessed_data.second_dim_a), std::move(ctx)));
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<RlweCiphertext<CoeffType>>
PirServer<DbDataType, CoeffType, MatCoeffType>::EvalPoly(
    const std::vector<RlweCiphertext<CoeffType>>& coeffs,
    const RgswCiphertext<CoeffType>& eval_point, FftContext& ctx) const {
  if (coeffs.empty()) {
    return absl::InvalidArgumentError("coeffs vector must not be empty");
  }

  // Implementing Horner's Method:
  // p(x) = c_0 + x * (c_1 + x * (... + x * c_{t-1} ))
  RlweCiphertext<CoeffType> result = coeffs.back();

  const auto& params = params_.RlweParameters();
  for (int i = coeffs.size() - 2; i >= 0; --i) {
    ASSIGN_OR_RETURN(RlweCiphertext<CoeffType> product,
                     ExternalProduct(params, params_.PolyEvalGadgetParams(),
                                     result, eval_point, ctx));
    ASSIGN_OR_RETURN(Polynomial<CoeffType> next_a, product.a.Add(coeffs[i].a));
    ASSIGN_OR_RETURN(Polynomial<CoeffType> next_b, product.b.Add(coeffs[i].b));
    result = {std::move(next_a), std::move(next_b)};
  }

  return result;
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<PirResponse<CoeffType>>
PirServer<DbDataType, CoeffType, MatCoeffType>::ProcessResponse(
    const PirRequest<CoeffType>& request) const {
  const int d = params_.RlweParameters().Degree();
  const int t = params_.InterpolationDegree();
  const int entry_size_multiple = params_.EntrySizeMultiple();

  // Extract y_vec_g and y_vec_h from the request packing key.
  const int num_digits = params_.PackGadgetParams().num_digits;
  if (request.packing_key.size() != 2 * num_digits) {
    return absl::InvalidArgumentError(
        "Packing key must contain 2 * num_digits polynomials.");
  }
  auto pack_begin = request.packing_key.begin();
  std::vector<Polynomial<CoeffType>> y_vec_g(pack_begin,
                                             pack_begin + num_digits);
  std::vector<Polynomial<CoeffType>> y_vec_h(pack_begin + num_digits,
                                             pack_begin + 2 * num_digits);

  // Stage 1: First dimension query processing via matrix-vector multiplication.
  const int db_cols = combined_db_matrix_.Cols();
  if (request.first_dimension_query.size() <
          static_cast<size_t>(params_.InterpolatedRows()) ||
      request.first_dimension_query.size() > static_cast<size_t>(db_cols)) {
    return absl::InvalidArgumentError(
        absl::StrCat("First dimension query size (",
                     request.first_dimension_query.size(),
                     ") is invalid for DB columns (", db_cols, ")."));
  }

  // Stage 3 (setup): Second dimension query processing via polynomial
  // evaluation.

  // Construct the full RGSW ciphertext (the evaluation point) from the given a
  // component and the second dimension query sent by the client, if applicable.
  RgswCiphertext<CoeffType> rgsw_ct;
  if (t > 1) {
    const int eval_num_digits = params_.PolyEvalGadgetParams().num_digits;
    if (request.second_dimension_query.size() != 2 * eval_num_digits) {
      return absl::InvalidArgumentError(
          "second_dimension_query size mismatch. Must be 2 * num_digits.");
    }
    rgsw_ct.u.reserve(eval_num_digits);
    rgsw_ct.v.reserve(eval_num_digits);
    for (int i = 0; i < eval_num_digits; ++i) {
      rgsw_ct.u.push_back(
          {second_dim_query_a_[i], request.second_dimension_query[i]});
    }
    for (int i = eval_num_digits; i < 2 * eval_num_digits; ++i) {
      rgsw_ct.v.push_back(
          {second_dim_query_a_[i], request.second_dimension_query[i]});
    }
  }

  ASSIGN_OR_RETURN(
      auto fft_ctx,
      FftContext::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));

  std::vector<RlweCiphertext<CoeffType>> shard_responses;
  shard_responses.reserve(entry_size_multiple);

  // ONE large DB multiplication
  auto start_db = absl::Now();
  std::vector<CoeffType> b_prime_all;
  if (request.first_dimension_query.size() == static_cast<size_t>(db_cols)) {
    ASSIGN_OR_RETURN(b_prime_all,
                     combined_db_matrix_.template Multiply<CoeffType>(
                         request.first_dimension_query));
  } else {
    std::vector<CoeffType> padded_first_dim(db_cols, 0);
    std::copy(request.first_dimension_query.begin(),
              request.first_dimension_query.end(), padded_first_dim.begin());
    ASSIGN_OR_RETURN(b_prime_all,
                     combined_db_matrix_.template Multiply<CoeffType>(
                         padded_first_dim));
  }
  db_mult_time_ += (absl::Now() - start_db);

  if (b_prime_all.size() != static_cast<size_t>(entry_size_multiple * t * d)) {
    return absl::InternalError(
        "DB multiplication resulted in incorrect vector size.");
  }

  // ONE large packing multiplication
  auto start_pack = absl::Now();
  const int k_gadget = y_vec_g.size();
  std::vector<CoeffType> y_vec_g_vec(k_gadget * d);
  for (int l = 0; l < k_gadget; ++l) {
    for (int m = 0; m < d; ++m) {
      y_vec_g_vec[l * d + m] = y_vec_g[l].Coeffs()[m];
    }
  }

  ASSIGN_OR_RETURN(
      std::vector<CoeffType> b_agg_partial_all,
      combined_pack_matrix_.template Multiply<CoeffType>(y_vec_g_vec));

  if (b_agg_partial_all.size() !=
      static_cast<size_t>(entry_size_multiple * t * d)) {
    return absl::InternalError(
        "Packing multiplication resulted in incorrect vector size.");
  }

  for (int k = 0; k < entry_size_multiple; ++k) {
    // Stage 2: Packing
    std::vector<RlweCiphertext<CoeffType>> packed_ciphertexts;
    packed_ciphertexts.reserve(t);
    for (int i = 0; i < t; ++i) {
      const size_t offset = k * t * d + i * d;
      // Extract d consecutive elements from b'
      std::vector<CoeffType> chunk(b_prime_all.begin() + offset,
                                   b_prime_all.begin() + offset + d);

      // Extract d consecutive elements from b_agg_partial_all
      std::vector<CoeffType> b_agg_partial(
          b_agg_partial_all.begin() + offset,
          b_agg_partial_all.begin() + offset + d);

      // Call FinalizeMatrixPack for this chunk.
      ASSIGN_OR_RETURN(
          RlweCiphertext<CoeffType> pack_result,
          FinalizeMatrixPack(params_.RlweParameters(), chunk, b_agg_partial,
                             y_vec_h, preprocessed_outputs_[k][i].t_vec_h,
                             preprocessed_outputs_[k][i].a_tilde_agg,
                             params_.PackGadgetParams(), *fft_ctx));
      packed_ciphertexts.push_back(std::move(pack_result));
    }
    packing_time_ += (absl::Now() - start_pack);

    auto start_eval = absl::Now();
    ASSIGN_OR_RETURN(RlweCiphertext<CoeffType> final_ciphertext,
                     EvalPoly(packed_ciphertexts, rgsw_ct, *fft_ctx));
    poly_eval_time_ += (absl::Now() - start_eval);

    // mod switch the first component of the ciphertext to q1
    ASSIGN_OR_RETURN(auto a_mod_switched,
                     final_ciphertext.a.Rescale(
                         params_.RlweParameters().LogModulus(),
                         params_.RlweParameters().LogModulus1AfterSwitch()));

    // mod switch the second component of the ciphertext to q2
    ASSIGN_OR_RETURN(auto b_mod_switched,
                     final_ciphertext.b.Rescale(
                         params_.RlweParameters().LogModulus(),
                         params_.RlweParameters().LogModulus2AfterSwitch()));

    RlweCiphertext<CoeffType> final_ciphertext_mod_switched = {
        std::move(a_mod_switched), std::move(b_mod_switched)};

    shard_responses.push_back(std::move(final_ciphertext_mod_switched));
  }

  return PirResponse<CoeffType>{std::move(shard_responses)};
}

namespace {

template <typename SignedType>
std::vector<uint8_t> BitPackSigned(const std::vector<SignedType>& values,
                                   int num_bits) {
  if (values.empty() || num_bits <= 0) return {};
  size_t total_bits = values.size() * num_bits;
  size_t total_bytes = (total_bits + 7) / 8;
  std::vector<uint8_t> out(total_bytes, 0);

  uint64_t mask = (num_bits == 64) ? ~0ULL : ((1ULL << num_bits) - 1);
  uint64_t bit_pos = 0;
  for (SignedType val : values) {
    uint64_t v = static_cast<uint64_t>(
                     static_cast<std::make_unsigned_t<SignedType>>(val)) &
                 mask;
    int bits_left = num_bits;
    while (bits_left > 0) {
      size_t byte_idx = bit_pos / 8;
      int bit_offset = bit_pos % 8;
      int take = std::min(bits_left, 8 - bit_offset);
      uint8_t m = (1 << take) - 1;
      out[byte_idx] |= static_cast<uint8_t>((v & m) << bit_offset);
      v >>= take;
      bit_pos += take;
      bits_left -= take;
    }
  }
  return out;
}

template <typename SignedType>
std::vector<SignedType> BitUnpackSigned(const uint8_t* data, size_t num_values,
                                        int num_bits) {
  std::vector<SignedType> out(num_values, 0);
  if (num_values == 0 || num_bits <= 0) return out;

  uint64_t bit_pos = 0;
  for (size_t i = 0; i < num_values; ++i) {
    uint64_t val = 0;
    int bits_left = num_bits;
    int shift = 0;
    while (bits_left > 0) {
      size_t byte_idx = bit_pos / 8;
      int bit_offset = bit_pos % 8;
      int take = std::min(bits_left, 8 - bit_offset);
      uint8_t chunk = (data[byte_idx] >> bit_offset) & ((1 << take) - 1);
      val |= (static_cast<uint64_t>(chunk) << shift);
      shift += take;
      bit_pos += take;
      bits_left -= take;
    }
    // Sign-extend if negative
    if (num_bits < 64 && (val & (1ULL << (num_bits - 1)))) {
      val |= ~((1ULL << num_bits) - 1);
    }
    out[i] = static_cast<SignedType>(val);
  }
  return out;
}

template <typename CoeffType>
std::vector<uint8_t> BitPackCoefficients(const std::vector<CoeffType>& values,
                                         int bits_per_val) {
  if (values.empty() || bits_per_val <= 0) return {};
  if (bits_per_val >= static_cast<int>(sizeof(CoeffType) * 8)) {
    std::vector<uint8_t> out(values.size() * sizeof(CoeffType));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
  }
  size_t total_bits = values.size() * bits_per_val;
  size_t total_bytes = (total_bits + 7) / 8;
  std::vector<uint8_t> out(total_bytes, 0);

  uint64_t bit_pos = 0;
  for (CoeffType val : values) {
    uint64_t v = static_cast<uint64_t>(val) &
                 ((bits_per_val == 64) ? ~0ULL : ((1ULL << bits_per_val) - 1));
    int bits_left = bits_per_val;
    while (bits_left > 0) {
      size_t byte_idx = bit_pos / 8;
      int bit_offset = bit_pos % 8;
      int take = std::min(bits_left, 8 - bit_offset);
      uint8_t mask = (1 << take) - 1;
      out[byte_idx] |= static_cast<uint8_t>((v & mask) << bit_offset);
      v >>= take;
      bit_pos += take;
      bits_left -= take;
    }
  }
  return out;
}

template <typename CoeffType>
std::vector<CoeffType> BitUnpackCoefficients(const uint8_t* data,
                                             size_t num_values,
                                             int bits_per_val) {
  std::vector<CoeffType> out(num_values, 0);
  if (num_values == 0 || bits_per_val <= 0) return out;
  if (bits_per_val >= static_cast<int>(sizeof(CoeffType) * 8)) {
    std::memcpy(out.data(), data, num_values * sizeof(CoeffType));
    return out;
  }
  uint64_t bit_pos = 0;
  for (size_t i = 0; i < num_values; ++i) {
    uint64_t val = 0;
    int bits_left = bits_per_val;
    int shift = 0;
    while (bits_left > 0) {
      size_t byte_idx = bit_pos / 8;
      int bit_offset = bit_pos % 8;
      int take = std::min(bits_left, 8 - bit_offset);
      uint8_t chunk = (data[byte_idx] >> bit_offset) & ((1 << take) - 1);
      val |= (static_cast<uint64_t>(chunk) << shift);
      shift += take;
      bit_pos += take;
      bits_left -= take;
    }
    out[i] = static_cast<CoeffType>(val);
  }
  return out;
}

}  // namespace

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::Status PirServer<DbDataType, CoeffType, MatCoeffType>::SaveToFile(
    absl::string_view file_path) const {
  std::ofstream out(std::string(file_path), std::ios::binary);
  if (!out.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to open file for writing: ", file_path));
  }

  // 1. Write combined_db_matrix_
  int32_t db_rows = combined_db_matrix_.Rows();
  int32_t db_cols = combined_db_matrix_.PhysicalCols();
  uint8_t db_condensed = combined_db_matrix_.IsCondensed() ? 1 : 0;
  int32_t db_orig_cols = combined_db_matrix_.OriginalCols();
  out.write(reinterpret_cast<const char*>(&db_rows), sizeof(db_rows));
  out.write(reinterpret_cast<const char*>(&db_cols), sizeof(db_cols));
  out.write(reinterpret_cast<const char*>(&db_condensed), sizeof(db_condensed));
  out.write(reinterpret_cast<const char*>(&db_orig_cols), sizeof(db_orig_cols));

  if (combined_db_matrix_.IsCondensed()) {
    const auto& cdata = combined_db_matrix_.CondensedData();
    uint64_t num_elems = cdata.size();
    out.write(reinterpret_cast<const char*>(&num_elems), sizeof(num_elems));
    if (num_elems > 0) {
      out.write(reinterpret_cast<const char*>(cdata.data()),
                num_elems * sizeof(uint64_t));
    }
  } else {
    const auto& data = combined_db_matrix_.Data();
    uint64_t num_elems = data.size();
    out.write(reinterpret_cast<const char*>(&num_elems), sizeof(num_elems));
    if (num_elems > 0) {
      out.write(reinterpret_cast<const char*>(data.data()),
                num_elems * sizeof(DbDataType));
    }
  }

  // 2. Write combined_pack_matrix_ in bit-packed format (pack_log_digit +
  // log2(d) bits per value).
  int32_t pack_rows = combined_pack_matrix_.Rows();
  int32_t pack_cols = combined_pack_matrix_.Cols();
  int32_t pack_phys_cols = combined_pack_matrix_.PhysicalCols();
  uint8_t pack_condensed = combined_pack_matrix_.IsCondensed() ? 1 : 0;
  int32_t pack_orig_cols = combined_pack_matrix_.OriginalCols();
  int32_t log2_d = absl::bit_width(
      static_cast<uint32_t>(params_.RlweParameters().Degree() - 1));
  int32_t pack_bits_per_entry = params_.PackGadgetParams().log_digit + log2_d;

  out.write(reinterpret_cast<const char*>(&pack_rows), sizeof(pack_rows));
  out.write(reinterpret_cast<const char*>(&pack_cols), sizeof(pack_cols));
  out.write(reinterpret_cast<const char*>(&pack_phys_cols),
            sizeof(pack_phys_cols));
  out.write(reinterpret_cast<const char*>(&pack_condensed),
            sizeof(pack_condensed));
  out.write(reinterpret_cast<const char*>(&pack_orig_cols),
            sizeof(pack_orig_cols));
  out.write(reinterpret_cast<const char*>(&pack_bits_per_entry),
            sizeof(pack_bits_per_entry));

  std::vector<MatCoeffType> pack_entries;
  pack_entries.reserve(static_cast<size_t>(pack_rows) * pack_cols);
  if (combined_pack_matrix_.IsCondensed()) {
    const auto& cdata = combined_pack_matrix_.CondensedData();
    for (int r = 0; r < pack_rows; ++r) {
      for (int c = 0; c < pack_cols; ++c) {
        if constexpr (std::is_same_v<MatCoeffType, int32_t>) {
          uint64_t w = cdata[static_cast<size_t>(r) * pack_phys_cols + (c / 2)];
          int32_t val = (c % 2 == 0)
                            ? static_cast<int32_t>(w & 0xFFFFFFFFULL)
                            : static_cast<int32_t>((w >> 32) & 0xFFFFFFFFULL);
          pack_entries.push_back(val);
        } else if constexpr (std::is_same_v<MatCoeffType, int16_t>) {
          uint64_t w = cdata[static_cast<size_t>(r) * pack_phys_cols + (c / 4)];
          int16_t val = static_cast<int16_t>((w >> ((c % 4) * 16)) & 0xFFFFULL);
          pack_entries.push_back(val);
        } else {
          pack_entries.push_back(0);
        }
      }
    }
  } else {
    pack_entries = combined_pack_matrix_.Data();
  }

  auto packed_matrix_bytes = BitPackSigned(pack_entries, pack_bits_per_entry);
  uint64_t packed_matrix_len = packed_matrix_bytes.size();
  out.write(reinterpret_cast<const char*>(&packed_matrix_len),
            sizeof(packed_matrix_len));
  if (packed_matrix_len > 0) {
    out.write(reinterpret_cast<const char*>(packed_matrix_bytes.data()),
              packed_matrix_len);
  }

  // 3. Write preprocessed_outputs_ in bit-packed format.
  int32_t log_modulus = params_.RlweParameters().LogModulus();
  out.write(reinterpret_cast<const char*>(&log_modulus), sizeof(log_modulus));

  int32_t num_shards = static_cast<int32_t>(preprocessed_outputs_.size());
  out.write(reinterpret_cast<const char*>(&num_shards), sizeof(num_shards));
  for (int32_t k = 0; k < num_shards; ++k) {
    int32_t num_slices = static_cast<int32_t>(preprocessed_outputs_[k].size());
    out.write(reinterpret_cast<const char*>(&num_slices), sizeof(num_slices));
    for (int32_t i = 0; i < num_slices; ++i) {
      // a_tilde_agg (bit-packed)
      const auto& a_coeffs = preprocessed_outputs_[k][i].a_tilde_agg.Coeffs();
      uint64_t a_len = a_coeffs.size();
      out.write(reinterpret_cast<const char*>(&a_len), sizeof(a_len));
      auto packed_a = BitPackCoefficients(a_coeffs, log_modulus);
      uint64_t packed_a_len = packed_a.size();
      out.write(reinterpret_cast<const char*>(&packed_a_len),
                sizeof(packed_a_len));
      if (packed_a_len > 0) {
        out.write(reinterpret_cast<const char*>(packed_a.data()), packed_a_len);
      }

      // t_vec_h (bit-packed)
      const auto& t_vec = preprocessed_outputs_[k][i].t_vec_h;
      uint64_t t_vec_len = t_vec.size();
      out.write(reinterpret_cast<const char*>(&t_vec_len), sizeof(t_vec_len));
      for (const auto& poly : t_vec) {
        const auto& p_coeffs = poly.Coeffs();
        uint64_t p_len = p_coeffs.size();
        out.write(reinterpret_cast<const char*>(&p_len), sizeof(p_len));
        auto packed_p = BitPackCoefficients(p_coeffs, log_modulus);
        uint64_t packed_p_len = packed_p.size();
        out.write(reinterpret_cast<const char*>(&packed_p_len),
                  sizeof(packed_p_len));
        if (packed_p_len > 0) {
          out.write(reinterpret_cast<const char*>(packed_p.data()),
                    packed_p_len);
        }
      }
    }
  }

  // 4. Write second_dim_query_a_ (bit-packed)
  uint64_t num_second_dim_a = second_dim_query_a_.size();
  out.write(reinterpret_cast<const char*>(&num_second_dim_a),
            sizeof(num_second_dim_a));
  for (const auto& poly : second_dim_query_a_) {
    const auto& p_coeffs = poly.Coeffs();
    uint64_t p_len = p_coeffs.size();
    out.write(reinterpret_cast<const char*>(&p_len), sizeof(p_len));
    auto packed_p = BitPackCoefficients(p_coeffs, log_modulus);
    uint64_t packed_p_len = packed_p.size();
    out.write(reinterpret_cast<const char*>(&packed_p_len),
              sizeof(packed_p_len));
    if (packed_p_len > 0) {
      out.write(reinterpret_cast<const char*>(packed_p.data()), packed_p_len);
    }
  }

  out.flush();
  if (!out.good()) {
    return absl::InternalError(
        absl::StrCat("Error writing to file: ", file_path));
  }
  return absl::OkStatus();
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
PirServer<DbDataType, CoeffType, MatCoeffType>::LoadFromBuffer(
    const PirParams<CoeffType>& params, const uint8_t* buffer, size_t size) {
  class BufferReader {
   public:
    BufferReader(const uint8_t* buffer, size_t size)
        : buffer_(buffer), size_(size), offset_(0) {}
    void read(char* dest, size_t n) {
      if (offset_ + n > size_) {
        ok_ = false;
        return;
      }
      memcpy(dest, buffer_ + offset_, n);
      offset_ += n;
    }
    bool is_open() const { return ok_; }
    bool good() const { return ok_; }
    bool eof() const { return offset_ >= size_; }

   private:
    const uint8_t* buffer_;
    size_t size_;
    size_t offset_;
    bool ok_ = true;
  };

  BufferReader in(buffer, size);
  if (!in.is_open()) {
    return absl::InternalError("Failed to read buffer");
  }

  // 1. Read combined_db_matrix_
  int32_t db_rows = 0, db_cols = 0, db_orig_cols = 0;
  uint8_t db_condensed = 0;
  in.read(reinterpret_cast<char*>(&db_rows), sizeof(db_rows));
  in.read(reinterpret_cast<char*>(&db_cols), sizeof(db_cols));
  in.read(reinterpret_cast<char*>(&db_condensed), sizeof(db_condensed));
  in.read(reinterpret_cast<char*>(&db_orig_cols), sizeof(db_orig_cols));

  uint64_t db_num_elems = 0;
  in.read(reinterpret_cast<char*>(&db_num_elems), sizeof(db_num_elems));

  Matrix<DbDataType> combined_db_matrix;
  if (db_condensed != 0) {
    std::vector<uint64_t> cdata(db_num_elems);
    if (db_num_elems > 0) {
      in.read(reinterpret_cast<char*>(cdata.data()),
              db_num_elems * sizeof(uint64_t));
    }
    combined_db_matrix = Matrix<DbDataType>(
        std::vector<DbDataType>(), std::move(cdata), db_rows, db_cols,
        /*is_condensed=*/true, db_orig_cols);
  } else {
    std::vector<DbDataType> data(db_num_elems);
    if (db_num_elems > 0) {
      in.read(reinterpret_cast<char*>(data.data()),
              db_num_elems * sizeof(DbDataType));
    }
    combined_db_matrix = Matrix<DbDataType>(
        std::move(data), std::vector<uint64_t>(), db_rows, db_cols,
        /*is_condensed=*/false, db_orig_cols);
  }

  // 2. Read combined_pack_matrix_ from bit-packed format.
  int32_t pack_rows = 0, pack_cols = 0, pack_phys_cols = 0, pack_orig_cols = 0;
  uint8_t pack_condensed = 0;
  int32_t pack_bits_per_entry = 0;
  in.read(reinterpret_cast<char*>(&pack_rows), sizeof(pack_rows));
  in.read(reinterpret_cast<char*>(&pack_cols), sizeof(pack_cols));
  in.read(reinterpret_cast<char*>(&pack_phys_cols), sizeof(pack_phys_cols));
  in.read(reinterpret_cast<char*>(&pack_condensed), sizeof(pack_condensed));
  in.read(reinterpret_cast<char*>(&pack_orig_cols), sizeof(pack_orig_cols));
  in.read(reinterpret_cast<char*>(&pack_bits_per_entry),
          sizeof(pack_bits_per_entry));

  uint64_t packed_matrix_len = 0;
  in.read(reinterpret_cast<char*>(&packed_matrix_len),
          sizeof(packed_matrix_len));
  std::vector<uint8_t> packed_matrix_bytes(packed_matrix_len);
  if (packed_matrix_len > 0) {
    in.read(reinterpret_cast<char*>(packed_matrix_bytes.data()),
            packed_matrix_len);
  }

  size_t total_pack_entries = static_cast<size_t>(pack_rows) * pack_cols;
  auto pack_entries = BitUnpackSigned<MatCoeffType>(
      packed_matrix_bytes.data(), total_pack_entries, pack_bits_per_entry);

  Matrix<MatCoeffType> combined_pack_matrix;
  if (pack_condensed != 0) {
    std::vector<uint64_t> cdata(static_cast<size_t>(pack_rows) * pack_phys_cols,
                                0);
    for (int r = 0; r < pack_rows; ++r) {
      for (int c = 0; c < pack_cols; ++c) {
        MatCoeffType val = pack_entries[static_cast<size_t>(r) * pack_cols + c];
        if constexpr (std::is_same_v<MatCoeffType, int32_t>) {
          if (c % 2 == 0) {
            cdata[static_cast<size_t>(r) * pack_phys_cols + (c / 2)] |=
                static_cast<uint64_t>(static_cast<uint32_t>(val));
          } else {
            cdata[static_cast<size_t>(r) * pack_phys_cols + (c / 2)] |=
                (static_cast<uint64_t>(static_cast<uint32_t>(val)) << 32);
          }
        } else if constexpr (std::is_same_v<MatCoeffType, int16_t>) {
          cdata[static_cast<size_t>(r) * pack_phys_cols + (c / 4)] |=
              (static_cast<uint64_t>(static_cast<uint16_t>(val))
               << ((c % 4) * 16));
        }
      }
    }
    combined_pack_matrix = Matrix<MatCoeffType>(
        std::vector<MatCoeffType>(), std::move(cdata), pack_rows, pack_cols,
        /*is_condensed=*/true, pack_orig_cols);
  } else {
    combined_pack_matrix = Matrix<MatCoeffType>(
        std::move(pack_entries), std::vector<uint64_t>(), pack_rows, pack_cols,
        /*is_condensed=*/false, pack_orig_cols);
  }
  // 3. Read preprocessed_outputs_ in bit-packed format.
  int32_t log_modulus = 0;
  in.read(reinterpret_cast<char*>(&log_modulus), sizeof(log_modulus));

  int32_t num_shards = 0;
  in.read(reinterpret_cast<char*>(&num_shards), sizeof(num_shards));
  std::vector<std::vector<PreprocessMatrixPackOutput<CoeffType, MatCoeffType>>>
      preprocessed_outputs(num_shards);
  for (int32_t k = 0; k < num_shards; ++k) {
    int32_t num_slices = 0;
    in.read(reinterpret_cast<char*>(&num_slices), sizeof(num_slices));
    preprocessed_outputs[k].resize(num_slices);
    for (int32_t i = 0; i < num_slices; ++i) {
      // a_tilde_agg (bit-unpacked)
      uint64_t a_len = 0;
      in.read(reinterpret_cast<char*>(&a_len), sizeof(a_len));
      uint64_t packed_a_len = 0;
      in.read(reinterpret_cast<char*>(&packed_a_len), sizeof(packed_a_len));
      std::vector<uint8_t> packed_a(packed_a_len);
      if (packed_a_len > 0) {
        in.read(reinterpret_cast<char*>(packed_a.data()), packed_a_len);
      }
      auto a_coeffs =
          BitUnpackCoefficients<CoeffType>(packed_a.data(), a_len, log_modulus);
      ASSIGN_OR_RETURN(preprocessed_outputs[k][i].a_tilde_agg,
                       Polynomial<CoeffType>::Create(std::move(a_coeffs)));

      // t_vec_h (bit-unpacked)
      uint64_t t_vec_len = 0;
      in.read(reinterpret_cast<char*>(&t_vec_len), sizeof(t_vec_len));
      preprocessed_outputs[k][i].t_vec_h.resize(t_vec_len);
      for (uint64_t j = 0; j < t_vec_len; ++j) {
        uint64_t p_len = 0;
        in.read(reinterpret_cast<char*>(&p_len), sizeof(p_len));
        uint64_t packed_p_len = 0;
        in.read(reinterpret_cast<char*>(&packed_p_len), sizeof(packed_p_len));
        std::vector<uint8_t> packed_p(packed_p_len);
        if (packed_p_len > 0) {
          in.read(reinterpret_cast<char*>(packed_p.data()), packed_p_len);
        }
        auto p_coeffs = BitUnpackCoefficients<CoeffType>(packed_p.data(), p_len,
                                                         log_modulus);
        ASSIGN_OR_RETURN(preprocessed_outputs[k][i].t_vec_h[j],
                         Polynomial<CoeffType>::Create(std::move(p_coeffs)));
      }
    }
  }

  // 4. Read second_dim_query_a_ (bit-unpacked)
  uint64_t num_second_dim_a = 0;
  in.read(reinterpret_cast<char*>(&num_second_dim_a), sizeof(num_second_dim_a));
  std::vector<Polynomial<CoeffType>> second_dim_query_a(num_second_dim_a);
  for (uint64_t j = 0; j < num_second_dim_a; ++j) {
    uint64_t p_len = 0;
    in.read(reinterpret_cast<char*>(&p_len), sizeof(p_len));
    uint64_t packed_p_len = 0;
    in.read(reinterpret_cast<char*>(&packed_p_len), sizeof(packed_p_len));
    std::vector<uint8_t> packed_p(packed_p_len);
    if (packed_p_len > 0) {
      in.read(reinterpret_cast<char*>(packed_p.data()), packed_p_len);
    }
    auto p_coeffs =
        BitUnpackCoefficients<CoeffType>(packed_p.data(), p_len, log_modulus);
    ASSIGN_OR_RETURN(second_dim_query_a[j],
                     Polynomial<CoeffType>::Create(std::move(p_coeffs)));
  }

  if (!in.good() && !in.eof()) {
    return absl::InternalError("Error reading from buffer");
  }

  const int d = params.RlweParameters().Degree();
  ASSIGN_OR_RETURN(
      Context ctx,
      Context::Create(absl::bit_width(static_cast<uint32_t>(d)) - 1));

  return absl::WrapUnique(new PirServer<DbDataType, CoeffType, MatCoeffType>(
      params, std::move(combined_db_matrix), std::move(combined_pack_matrix),
      std::move(preprocessed_outputs), std::move(second_dim_query_a),
      std::move(ctx)));
}

template <typename DbDataType, typename CoeffType, typename MatCoeffType>
absl::StatusOr<std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>>>
PirServer<DbDataType, CoeffType, MatCoeffType>::LoadFromFile(
    const PirParams<CoeffType>& params, absl::string_view file_path) {
  std::string buffer;
  if (!file_util::GetFileContents(file_path, &buffer).ok()) {
    return absl::InternalError("Failed to read file.");
  }
  return LoadFromBuffer(
      params, reinterpret_cast<const uint8_t*>(buffer.data()), buffer.size());
}

template class PirServer<uint16_t, uint32_t, uint32_t>;
template class PirServer<uint16_t, uint64_t, uint64_t>;
template class PirServer<uint16_t, uint32_t, int16_t>;
template class PirServer<uint16_t, uint64_t, int16_t>;
template class PirServer<uint16_t, uint32_t, int32_t>;
template class PirServer<uint16_t, uint64_t, int32_t>;

template class PirServer<uint8_t, uint32_t, uint32_t>;
template class PirServer<uint8_t, uint64_t, uint64_t>;
template class PirServer<uint8_t, uint32_t, int16_t>;
template class PirServer<uint8_t, uint64_t, int16_t>;
template class PirServer<uint8_t, uint32_t, int32_t>;
template class PirServer<uint8_t, uint64_t, int32_t>;

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
