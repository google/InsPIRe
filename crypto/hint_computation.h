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

#ifndef CRYPTO_HINT_COMPUTATION_H_
#define CRYPTO_HINT_COMPUTATION_H_

#include <vector>

#include "crypto/matrix.h"
#include "crypto/polynomial.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {

// Computes the matrix multiplication of a left Matrix `mat` and a right
// negacyclic Matrix formed by Polynomial `poly`.
//
// If `poly` = a_0 + a_1*X + ... + a_{d-1}*X^{d-1}, then the right matrix is
// implicitly constructed as:
//   [ a_0     -a_{d-1}  ...  -a_1 ]
//   [ a_1      a_0      ...  -a_2 ]
//   [ ...      ...      ...   ... ]
//   [ a_{d-1}  a_{d-2}  ...   a_0 ]
//
// The resulting Matrix has the same dimensions as `mat`, with elements of type
// `OutputType`. `mat` must have the same number of columns as the length of
// `poly`, which is a power of two.
//
// During computations, both `mat` and `poly` are cast to `OutputType`.
template <typename OutputType, typename S, typename T>
absl::StatusOr<Matrix<OutputType>> MulMatNegacyclic(const Matrix<S>& mat,
                                                    const Polynomial<T>& poly,
                                                    FftContext& ctx);

// Computes the hint matrix given a `database` matrix and the set of
// "a components" of the query RLWE ciphertexts.
//
// Let `database` have dimension m by d * n, and `a_components` have length n.
// Each element of `a_components` is a polynomial of degree d.
//
// Interpret `database` as the following block matrix:
//  database = [ D_1 | D_2 | ... | D_n ]
// where each D_k is a m x d matrix.
//
// The right matrix formed by `a_components` can be implicitly represented as
// the following block matrix:
//        [ A_1 ]
//   A =  [ A_2 ]
//        [ ... ]
//        [ A_n ]
// where each A_k is a d x d negacyclic matrix formed by the coefficients of the
// k-th polynomial in `a_components`.
//
// This function computes the inner product of the block rows of `database` and
// the block column `A`.
//
// Elements of `database` and `a_components` are cast to `OutputType` during
// computations.
//
// Note that this function can also be used in a distributed setting by
// partitioning the `database` matrix into blocks by rows.
template <typename OutputType, typename S, typename T>
absl::StatusOr<Matrix<OutputType>> ComputeHintMatrix(
    const Matrix<S>& database, const std::vector<Polynomial<T>>& a_components,
    FftContext& ctx);

// Expands the set of polynomials into the vertically stacked block matrix A
// where each block is a d x d negacyclic matrix formed by a polynomial.
template <typename OutputType, typename T>
absl::StatusOr<Matrix<OutputType>> ExpandAMatrix(
    const std::vector<Polynomial<T>>& a_components);

}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_HINT_COMPUTATION_H_
