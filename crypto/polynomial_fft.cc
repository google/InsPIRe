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

#include "crypto/polynomial_fft.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "crypto/polynomial.h"
#include "crypto/fft.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "crypto/status_macros.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

namespace fft = ::security::fft;

absl::StatusOr<std::unique_ptr<FftContext>> FftContext::Create(int log_d) {
  if (log_d < 0 || log_d > 30) {
    return absl::InvalidArgumentError("Invalid log_d.");
  }
  int d = 1 << log_d;
  int len2 = 2 * d;
  std::unique_ptr<FftContext> ctx(new FftContext());
  ctx->d = d;

  ctx->forward_in.resize(len2);
  ctx->forward_out.resize(len2);
  ctx->backward_in.resize(len2);
  ctx->backward_out.resize(len2);

  ctx->plan = std::make_unique<fft::FftPlan<double, 1>>(
      std::array<size_t, 1>{static_cast<size_t>(len2)},
      fft::Normalization::kNone);

  return ctx;
}

template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> Polynomial<CoeffType>::MultFft(
    const Polynomial& that, FftContext& ctx, int this_bits, int that_bits,
    int chunk_bits) const {
  return InnerProductFft({*this}, {that}, ctx, this_bits, that_bits,
                         chunk_bits);
}

template <typename CoeffType>
absl::StatusOr<Polynomial<CoeffType>> Polynomial<CoeffType>::InnerProductFft(
    const std::vector<Polynomial>& u, const std::vector<Polynomial>& v,
    FftContext& ctx, int u_bits, int v_bits, int chunk_bits, bool u_is_signed) {
  if (u.size() != v.size()) {
    return absl::InvalidArgumentError(
        "Polynomial vectors must have the same size.");
  }
  if (u.empty()) {
    return absl::InvalidArgumentError("Polynomial vectors cannot be empty.");
  }
  int d = u[0].Len();
  int len2 = 2 * d;
  for (size_t i = 0; i < u.size(); ++i) {
    if (u[i].Len() != d || v[i].Len() != d || d != ctx.d) {
      return absl::InvalidArgumentError(
          "Polynomial lengths or context mismatch.");
    }
  }
  if (ctx.plan == nullptr) {
    return absl::InvalidArgumentError("FftContext plan is not initialized.");
  }
  if (u_bits <= 0 || v_bits <= 0 || chunk_bits <= 0 || chunk_bits > 63) {
    return absl::InvalidArgumentError("Invalid bits or chunk_bits.");
  }
  if (std::log2(u.size()) + std::log2(d) + 2.0 * chunk_bits > 53.0) {
    return absl::InvalidArgumentError(
        "Potential precision overflow in accumulation.");
  }

  uint64_t chunk_mask = (1ULL << chunk_bits) - 1;
  int max_chunks = (8 * sizeof(CoeffType) + chunk_bits - 1) / chunk_bits;
  int u_num_chunks = (u_bits + chunk_bits - 1) / chunk_bits;
  int v_num_chunks = (v_bits + chunk_bits - 1) / chunk_bits;
  if (u_num_chunks > max_chunks) {
    u_num_chunks = max_chunks;
  }
  if (v_num_chunks > max_chunks) {
    v_num_chunks = max_chunks;
  }

  std::vector<std::vector<std::complex<double>>> accum(
      u_num_chunks * v_num_chunks,
      std::vector<std::complex<double>>(len2, {0.0, 0.0}));

  std::vector<std::vector<std::complex<double>>> u_ffts(
      u_num_chunks, std::vector<std::complex<double>>(len2));
  std::vector<std::vector<std::complex<double>>> v_ffts(
      v_num_chunks, std::vector<std::complex<double>>(len2));

  for (size_t i = 0; i < u.size(); ++i) {
    for (int c = 0; c < u_num_chunks; ++c) {
      for (int j = 0; j < len2; ++j) {
        double val = 0.0;
        if (j < d) {
          uint64_t chunk = (u[i].coeffs_[j] >> (chunk_bits * c)) & chunk_mask;
          if (u_is_signed && c == u_num_chunks - 1) {
            int bits_in_chunk = u_bits - chunk_bits * c;
            int shift = 64 - bits_in_chunk;
            int64_t signed_chunk =
                static_cast<int64_t>(chunk << shift) >> shift;
            val = static_cast<double>(signed_chunk);
          } else {
            val = static_cast<double>(chunk);
          }
        }
        ctx.forward_in[j] = std::complex<double>(val, 0.0);
      }
      RETURN_IF_ERROR(fft::Fft(absl::MakeConstSpan(ctx.forward_in),
                               absl::MakeSpan(ctx.forward_out), *ctx.plan));
      u_ffts[c] = ctx.forward_out;
    }

    for (int c = 0; c < v_num_chunks; ++c) {
      for (int j = 0; j < len2; ++j) {
        ctx.forward_in[j] = std::complex<double>(
            (j < d) ? static_cast<double>(
                          (v[i].coeffs_[j] >> (chunk_bits * c)) & chunk_mask)
                    : 0.0,
            0.0);
      }
      RETURN_IF_ERROR(fft::Fft(absl::MakeConstSpan(ctx.forward_in),
                               absl::MakeSpan(ctx.forward_out), *ctx.plan));
      v_ffts[c] = ctx.forward_out;
    }

    for (int a = 0; a < u_num_chunks; ++a) {
      for (int b = 0; b < v_num_chunks; ++b) {
        if (chunk_bits * (a + b) >= 8 * sizeof(CoeffType)) {
          continue;
        }
        int accum_idx = a * v_num_chunks + b;
        for (int j = 0; j < len2; ++j) {
          accum[accum_idx][j] += u_ffts[a][j] * v_ffts[b][j];
        }
      }
    }
  }

  std::vector<CoeffType> result(d, 0);

  for (int a = 0; a < u_num_chunks; ++a) {
    for (int b = 0; b < v_num_chunks; ++b) {
      if (chunk_bits * (a + b) >= 8 * sizeof(CoeffType)) {
        continue;
      }

      int accum_idx = a * v_num_chunks + b;
      for (int j = 0; j < len2; ++j) {
        ctx.backward_in[j] = accum[accum_idx][j];
      }

      RETURN_IF_ERROR(fft::Ifft(absl::MakeConstSpan(ctx.backward_in),
                                absl::MakeSpan(ctx.backward_out), *ctx.plan));

      for (int j = 0; j < len2; ++j) {
        double val_double = ctx.backward_out[j].real() / len2;
        int64_t val64 = static_cast<int64_t>(std::round(val_double));
        CoeffType val = static_cast<CoeffType>(static_cast<uint64_t>(val64));
        if (j < d) {
          result[j] += (val << (chunk_bits * (a + b)));
        } else {
          result[j - d] -= (val << (chunk_bits * (a + b)));
        }
      }
    }
  }

  return Polynomial<CoeffType>::Create(std::move(result));
}

template class Polynomial<uint32_t>;
template class Polynomial<uint64_t>;

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
