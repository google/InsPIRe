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
#include <vector>

#include "crypto/matrix.h"
#include <benchmark/benchmark.h>
#include <gmock/gmock.h>
#include "crypto/status_macros.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

ABSL_FLAG(int, rows, 300000, "Number of rows of the matrix.");
ABSL_FLAG(int, cols, 3000, "Number of columns of the matrix.");

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

void BM_MatrixSum_U16_U64(benchmark::State& state) {
  const int rows = absl::GetFlag(FLAGS_rows);
  const int cols = absl::GetFlag(FLAGS_cols);

  std::vector<std::vector<uint16_t>> matrix_data(rows,
                                                 std::vector<uint16_t>(cols));
  std::mt19937 gen(1);
  std::uniform_int_distribution<uint16_t> distrib;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] = distrib(gen);
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<uint16_t>::Create(matrix_data));

  for (auto s : state) {
    uint64_t sum = 0;
    for (uint64_t element : matrix.Data()) {
      sum += element;
    }
    benchmark::DoNotOptimize(sum);
  }
}

void BM_MatrixMultiply_U16_U64(benchmark::State& state) {
  const int rows = absl::GetFlag(FLAGS_rows);
  const int cols = absl::GetFlag(FLAGS_cols);

  std::vector<std::vector<uint16_t>> matrix_data(rows,
                                                 std::vector<uint16_t>(cols));
  std::mt19937 gen(1);
  std::uniform_int_distribution<uint16_t> distrib;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] = distrib(gen);
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<uint16_t>::Create(matrix_data));

  std::vector<uint64_t> vec(cols);
  std::uniform_int_distribution<uint64_t> vec_distrib;
  for (int j = 0; j < cols; ++j) {
    vec[j] = vec_distrib(gen);
  }

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto result,
                         (matrix.template Multiply<uint64_t, uint64_t>(vec)));
    benchmark::DoNotOptimize(result);
  }
}

void BM_MatrixSum_I32_U64(benchmark::State& state) {
  const int rows = absl::GetFlag(FLAGS_rows);
  const int cols = absl::GetFlag(FLAGS_cols);

  std::vector<std::vector<int32_t>> matrix_data(rows,
                                                std::vector<int32_t>(cols));
  std::mt19937 gen(1);
  std::uniform_int_distribution<int32_t> distrib;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] = distrib(gen);
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<int32_t>::Create(matrix_data));

  for (auto s : state) {
    uint64_t sum = 0;
    for (uint64_t element : matrix.Data()) {
      sum += element;
    }
    benchmark::DoNotOptimize(sum);
  }
}

void BM_MatrixMultiply_I32_U64(benchmark::State& state) {
  const int rows = absl::GetFlag(FLAGS_rows);
  const int cols = absl::GetFlag(FLAGS_cols);

  std::vector<std::vector<int32_t>> matrix_data(rows,
                                                std::vector<int32_t>(cols));
  std::mt19937 gen(1);
  std::uniform_int_distribution<int32_t> distrib;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] = distrib(gen);
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<int32_t>::Create(matrix_data));

  std::vector<uint64_t> vec(cols);
  std::uniform_int_distribution<uint64_t> vec_distrib;
  for (int j = 0; j < cols; ++j) {
    vec[j] = vec_distrib(gen);
  }

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto result,
                         (matrix.template Multiply<uint64_t, uint64_t>(vec)));
    benchmark::DoNotOptimize(result);
  }
}

void BM_MatrixMultiply_U64_U64(benchmark::State& state) {
  const int rows = absl::GetFlag(FLAGS_rows);
  const int cols = absl::GetFlag(FLAGS_cols);

  std::vector<std::vector<uint64_t>> matrix_data(rows,
                                                 std::vector<uint64_t>(cols));
  std::mt19937 gen(1);
  std::uniform_int_distribution<uint64_t> distrib;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] = distrib(gen);
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<uint64_t>::Create(matrix_data));

  std::vector<uint64_t> vec(cols);
  std::uniform_int_distribution<uint64_t> vec_distrib;
  for (int j = 0; j < cols; ++j) {
    vec[j] = vec_distrib(gen);
  }

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto result,
                         (matrix.template Multiply<uint64_t, uint64_t>(vec)));
    benchmark::DoNotOptimize(result);
  }
}

void BM_MatrixSum_U64_U64(benchmark::State& state) {
  const int rows = absl::GetFlag(FLAGS_rows);
  const int cols = absl::GetFlag(FLAGS_cols);

  std::vector<std::vector<uint64_t>> matrix_data(rows,
                                                 std::vector<uint64_t>(cols));
  std::mt19937 gen(1);
  std::uniform_int_distribution<uint64_t> distrib;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] = distrib(gen);
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix, Matrix<uint64_t>::Create(matrix_data));

  for (auto s : state) {
    uint64_t sum = 0;
    for (uint64_t element : matrix.Data()) {
      sum += element;
    }
    benchmark::DoNotOptimize(sum);
  }
}

BENCHMARK(BM_MatrixSum_U16_U64);
BENCHMARK(BM_MatrixMultiply_U16_U64);

BENCHMARK(BM_MatrixSum_I32_U64);
BENCHMARK(BM_MatrixMultiply_I32_U64);

BENCHMARK(BM_MatrixSum_U64_U64);
BENCHMARK(BM_MatrixMultiply_U64_U64);

// blaze run -c opt --dynamic_mode=off
// //privacy/private_membership/rlwe/v2/crypto:matrix_benchmark --
// --benchmark_filter=all --rows=300000 --cols=3000

void BM_MatrixMultiplyCondensed_U8_U64(benchmark::State& state) {
  const int rows = absl::GetFlag(FLAGS_rows);
  const int cols = absl::GetFlag(FLAGS_cols);

  std::vector<std::vector<uint8_t>> matrix_data(rows,
                                                std::vector<uint8_t>(cols));
  std::mt19937 gen(1);
  std::uniform_int_distribution<uint16_t> distrib;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] = distrib(gen) % 256;
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix,
                       Matrix<uint8_t>::CreateCondensed(matrix_data));

  std::vector<uint64_t> vec(cols);
  std::uniform_int_distribution<uint64_t> vec_distrib;
  for (int j = 0; j < cols; ++j) {
    vec[j] = vec_distrib(gen);
  }

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto result,
                         matrix.template Multiply<uint64_t>(vec));
    benchmark::DoNotOptimize(result);
  }
}

void BM_MatrixMultiplyCondensed_U16_U64(benchmark::State& state) {
  const int rows = absl::GetFlag(FLAGS_rows);
  const int cols = absl::GetFlag(FLAGS_cols);

  std::vector<std::vector<uint16_t>> matrix_data(rows,
                                                 std::vector<uint16_t>(cols));
  std::mt19937 gen(1);
  std::uniform_int_distribution<uint16_t> distrib;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] = distrib(gen);
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix,
                       Matrix<uint16_t>::CreateCondensed(matrix_data));

  std::vector<uint64_t> vec(cols);
  std::uniform_int_distribution<uint64_t> vec_distrib;
  for (int j = 0; j < cols; ++j) {
    vec[j] = vec_distrib(gen);
  }

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto result,
                         matrix.template Multiply<uint64_t>(vec));
    benchmark::DoNotOptimize(result);
  }
}

void BM_MatrixMultiplyCondensed_I32_U64(benchmark::State& state) {
  const int rows = absl::GetFlag(FLAGS_rows);
  const int cols = absl::GetFlag(FLAGS_cols);

  std::vector<std::vector<int32_t>> matrix_data(rows,
                                                std::vector<int32_t>(cols));
  std::mt19937 gen(1);
  std::uniform_int_distribution<int32_t> distrib;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix_data[i][j] = distrib(gen);
    }
  }

  ASSERT_OK_AND_ASSIGN(auto matrix,
                       Matrix<int32_t>::CreateCondensed(matrix_data));

  std::vector<uint64_t> vec(cols);
  std::uniform_int_distribution<uint64_t> vec_distrib;
  for (int j = 0; j < cols; ++j) {
    vec[j] = vec_distrib(gen);
  }

  for (auto s : state) {
    ASSERT_OK_AND_ASSIGN(auto result,
                         matrix.template Multiply<uint64_t>(vec));
    benchmark::DoNotOptimize(result);
  }
}

BENCHMARK(BM_MatrixMultiplyCondensed_U8_U64);
BENCHMARK(BM_MatrixMultiplyCondensed_U16_U64);
BENCHMARK(BM_MatrixMultiplyCondensed_I32_U64);

}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
