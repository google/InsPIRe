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

#ifndef CRYPTO_POLYNOMIAL_FFT_H_
#define CRYPTO_POLYNOMIAL_FFT_H_

#include <complex>
#include <memory>
#include <vector>

#include "crypto/polynomial.h"
#include "crypto/fft.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// FftContext holds pre-allocated buffers and precomputed FFT plans needed
// for optimized FFT-based polynomial multiplication.
struct FftContext {
  static absl::StatusOr<std::unique_ptr<FftContext>> Create(int log_d);

  ~FftContext() = default;

  // Non-copyable and non-movable.
  FftContext(const FftContext&) = delete;
  FftContext& operator=(const FftContext&) = delete;

  int d;
  std::vector<std::complex<double>> forward_in;
  std::vector<std::complex<double>> forward_out;
  std::vector<std::complex<double>> backward_in;
  std::vector<std::complex<double>> backward_out;

  std::unique_ptr<::security::fft::FftPlan<double, 1>> plan;

 private:
  FftContext() = default;
};

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_POLYNOMIAL_FFT_H_
