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

#ifndef CRYPTO_FFT_H_
#define CRYPTO_FFT_H_

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"

namespace security::fft {

enum class Normalization {
  kNone,
};

template <typename T, size_t N>
class FftPlan;

template <>
class FftPlan<double, 1> {
 public:
  explicit FftPlan(std::array<size_t, 1> dims,
                   Normalization norm = Normalization::kNone)
      : size_(dims[0]) {
    bit_rev_.resize(size_);
    size_t log_n = 0;
    while ((size_t(1) << log_n) < size_) {
      ++log_n;
    }
    for (size_t i = 0; i < size_; ++i) {
      size_t rev = 0;
      for (size_t b = 0; b < log_n; ++b) {
        if (i & (1 << b)) {
          rev |= (1 << (log_n - 1 - b));
        }
      }
      bit_rev_[i] = rev;
    }

    twiddles_forward_.resize(size_ / 2);
    twiddles_backward_.resize(size_ / 2);
    const double pi = std::acos(-1.0);
    for (size_t i = 0; i < size_ / 2; ++i) {
      double angle = -2.0 * pi * i / size_;
      twiddles_forward_[i] =
          std::complex<double>(std::cos(angle), std::sin(angle));
      twiddles_backward_[i] =
          std::complex<double>(std::cos(-angle), std::sin(-angle));
    }
  }

  ~FftPlan() = default;

  FftPlan(const FftPlan&) = delete;
  FftPlan& operator=(const FftPlan&) = delete;
  FftPlan(FftPlan&&) = default;
  FftPlan& operator=(FftPlan&&) = default;

  size_t size() const { return size_; }
  const std::vector<size_t>& bit_rev() const { return bit_rev_; }
  const std::vector<std::complex<double>>& twiddles_forward() const {
    return twiddles_forward_;
  }
  const std::vector<std::complex<double>>& twiddles_backward() const {
    return twiddles_backward_;
  }

 private:
  size_t size_;
  std::vector<size_t> bit_rev_;
  std::vector<std::complex<double>> twiddles_forward_;
  std::vector<std::complex<double>> twiddles_backward_;
};

inline void ComputeFftInternal(
    absl::Span<const std::complex<double>> in,
    absl::Span<std::complex<double>> out,
    const FftPlan<double, 1>& plan,
    const std::vector<std::complex<double>>& twiddles) {
  const size_t n = plan.size();
  const auto& bit_rev = plan.bit_rev();
  for (size_t i = 0; i < n; ++i) {
    out[bit_rev[i]] = in[i];
  }

  for (size_t len = 2; len <= n; len <<= 1) {
    size_t half_len = len >> 1;
    size_t step = n / len;
    for (size_t i = 0; i < n; i += len) {
      for (size_t j = 0; j < half_len; ++j) {
        std::complex<double> u = out[i + j];
        std::complex<double> v = out[i + j + half_len] * twiddles[j * step];
        out[i + j] = u + v;
        out[i + j + half_len] = u - v;
      }
    }
  }
}

inline absl::Status Fft(absl::Span<const std::complex<double>> in,
                        absl::Span<std::complex<double>> out,
                        const FftPlan<double, 1>& plan) {
  if (in.size() != plan.size() || out.size() != plan.size()) {
    return absl::InvalidArgumentError(
        "Input/output size mismatch with FFT plan.");
  }
  ComputeFftInternal(in, out, plan, plan.twiddles_forward());
  return absl::OkStatus();
}

inline absl::Status Ifft(absl::Span<const std::complex<double>> in,
                         absl::Span<std::complex<double>> out,
                         const FftPlan<double, 1>& plan) {
  if (in.size() != plan.size() || out.size() != plan.size()) {
    return absl::InvalidArgumentError(
        "Input/output size mismatch with FFT plan.");
  }
  ComputeFftInternal(in, out, plan, plan.twiddles_backward());
  return absl::OkStatus();
}

}  // namespace security::fft

#endif  // CRYPTO_FFT_H_
