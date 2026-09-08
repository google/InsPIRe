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

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "crypto/polynomial.h"
#include "crypto/polynomial_fft.h"
#include <benchmark/benchmark.h>
#include <gmock/gmock.h>
#include "crypto/status_macros.h"
#include "absl/flags/parse.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

void BM_PolynomialMultU32(benchmark::State& state) {
  int degree = 1024;
  std::vector<uint32_t> coeffs1(degree);
  std::vector<uint32_t> coeffs2(degree);
  // Generate coefficients using a pseudo random number generator with a fixed
  // seed.
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint32_t> distrib;
  for (int i = 0; i < degree; ++i) {
    coeffs1[i] = distrib(gen);
    coeffs2[i] = distrib(gen);
  }
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create(coeffs1));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create(coeffs2));

  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(10));

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto p3, p1.Mult(p2, ctx));
    benchmark::DoNotOptimize(p3);
  }
}

void BM_PolynomialMultU64(benchmark::State& state) {
  int degree = 1024;
  std::vector<uint64_t> coeffs1(degree);
  std::vector<uint64_t> coeffs2(degree);
  // Generate coefficients using a pseudo random number generator with a fixed
  // seed.
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint64_t> distrib;
  for (int i = 0; i < degree; ++i) {
    coeffs1[i] = distrib(gen);
    coeffs2[i] = distrib(gen);
  }
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint64_t>::Create(coeffs1));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint64_t>::Create(coeffs2));

  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(10));

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto p3, p1.Mult(p2, ctx));
    benchmark::DoNotOptimize(p3);
  }
}

BENCHMARK(BM_PolynomialMultU32);
BENCHMARK(BM_PolynomialMultU64);

void BM_PolynomialMultFftU32(benchmark::State& state) {
  int degree = 1024;
  std::vector<uint32_t> coeffs1(degree);
  std::vector<uint32_t> coeffs2(degree);
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint32_t> distrib;
  for (int i = 0; i < degree; ++i) {
    coeffs1[i] = distrib(gen);
    coeffs2[i] = distrib(gen);
  }
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint32_t>::Create(coeffs1));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint32_t>::Create(coeffs2));

  ASSERT_OK_AND_ASSIGN(auto ctx, FftContext::Create(10));

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto p3, p1.MultFft(p2, *ctx));
    benchmark::DoNotOptimize(p3);
  }
}

void BM_PolynomialMultFftU64(benchmark::State& state) {
  int degree = 1024;
  std::vector<uint64_t> coeffs1(degree);
  std::vector<uint64_t> coeffs2(degree);
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint64_t> distrib;
  for (int i = 0; i < degree; ++i) {
    coeffs1[i] = distrib(gen);
    coeffs2[i] = distrib(gen);
  }
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint64_t>::Create(coeffs1));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint64_t>::Create(coeffs2));

  ASSERT_OK_AND_ASSIGN(auto ctx, FftContext::Create(10));

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto p3, p1.MultFft(p2, *ctx));
    benchmark::DoNotOptimize(p3);
  }
}

BENCHMARK(BM_PolynomialMultFftU32);
BENCHMARK(BM_PolynomialMultFftU64);

void BM_PolynomialMultipleMult(benchmark::State& state) {
  int degree = 1024;
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint64_t> distrib;

  std::vector<Polynomial<uint64_t>> polys1;
  std::vector<Polynomial<uint64_t>> polys2;
  for (int k = 0; k < 256; ++k) {
    std::vector<uint64_t> c1(degree), c2(degree);
    for (int i = 0; i < degree; ++i) {
      c1[i] = distrib(gen);
      c2[i] = distrib(gen);
    }
    ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint64_t>::Create(c1));
    ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint64_t>::Create(c2));
    polys1.push_back(std::move(p1));
    polys2.push_back(std::move(p2));
  }

  ASSERT_OK_AND_ASSIGN(auto ctx, Context::Create(10));

  for (auto s : state) {
    for (int i = 0; i < 256; ++i) {
      ASSERT_OK_AND_ASSIGN(auto p3, polys1[i].Mult(polys2[i], ctx));
      benchmark::DoNotOptimize(p3);
    }
  }
}

void BM_PolynomialMultipleMultFft(benchmark::State& state) {
  int degree = 1024;
  std::mt19937 gen(42);
  std::uniform_int_distribution<uint64_t> distrib;

  std::vector<Polynomial<uint64_t>> polys1;
  std::vector<Polynomial<uint64_t>> polys2;
  for (int k = 0; k < 256; ++k) {
    std::vector<uint64_t> c1(degree), c2(degree);
    for (int i = 0; i < degree; ++i) {
      c1[i] = distrib(gen);
      c2[i] = distrib(gen);
    }
    ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint64_t>::Create(c1));
    ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint64_t>::Create(c2));
    polys1.push_back(std::move(p1));
    polys2.push_back(std::move(p2));
  }

  ASSERT_OK_AND_ASSIGN(auto ctx, FftContext::Create(10));

  for (auto s : state) {
    for (int i = 0; i < 256; ++i) {
      ASSERT_OK_AND_ASSIGN(auto p3, polys1[i].MultFft(polys2[i], *ctx));
      benchmark::DoNotOptimize(p3);
    }
  }
}

BENCHMARK(BM_PolynomialMultipleMult);
BENCHMARK(BM_PolynomialMultipleMultFft);

void BM_PolynomialMultFftSpecificBits(benchmark::State& state) {
  int degree = 1024;
  std::vector<uint64_t> coeffs1(degree);
  std::vector<uint64_t> coeffs2(degree);
  std::mt19937 gen(47);
  std::uniform_int_distribution<uint64_t> distrib;
  for (int i = 0; i < degree; ++i) {
    coeffs1[i] = distrib(gen);
    coeffs2[i] = distrib(gen) & ((1ULL << 19) - 1);
  }
  ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint64_t>::Create(coeffs1));
  ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint64_t>::Create(coeffs2));

  ASSERT_OK_AND_ASSIGN(auto ctx, FftContext::Create(10));

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto p3, p1.MultFft(p2, *ctx, 64, 19));
    benchmark::DoNotOptimize(p3);
  }
}

void BM_PolynomialMultipleMultFftSpecificBits(benchmark::State& state) {
  int degree = 1024;
  std::mt19937 gen(47);
  std::uniform_int_distribution<uint64_t> distrib;

  std::vector<Polynomial<uint64_t>> polys1;
  std::vector<Polynomial<uint64_t>> polys2;
  for (int k = 0; k < 256; ++k) {
    std::vector<uint64_t> c1(degree), c2(degree);
    for (int i = 0; i < degree; ++i) {
      c1[i] = distrib(gen) & ((1ULL << 56) - 1);
      c2[i] = distrib(gen) & ((1ULL << 19) - 1);
    }
    ASSERT_OK_AND_ASSIGN(auto p1, Polynomial<uint64_t>::Create(c1));
    ASSERT_OK_AND_ASSIGN(auto p2, Polynomial<uint64_t>::Create(c2));
    polys1.push_back(std::move(p1));
    polys2.push_back(std::move(p2));
  }

  ASSERT_OK_AND_ASSIGN(auto ctx, FftContext::Create(10));

  for (auto s : state) {
    for (int i = 0; i < 256; ++i) {
      ASSERT_OK_AND_ASSIGN(auto p3,
                           polys1[i].MultFft(polys2[i], *ctx, 56, 19, 20));
      benchmark::DoNotOptimize(p3);
    }
  }
}

BENCHMARK(BM_PolynomialMultFftSpecificBits);
BENCHMARK(BM_PolynomialMultipleMultFftSpecificBits);

// blaze run -c opt --dynamic_mode=off
// //privacy/private_membership/rlwe/v2/crypto:polynomial_benchmark --
// --benchmark_filter=all

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
