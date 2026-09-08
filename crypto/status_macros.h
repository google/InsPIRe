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

#ifndef CRYPTO_STATUS_MACROS_H_
#define CRYPTO_STATUS_MACROS_H_

#include <utility>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace internal_status_macros {

inline const ::absl::Status& GetStatus(const ::absl::Status& status) {
  return status;
}

template <typename T>
inline const ::absl::Status& GetStatus(const ::absl::StatusOr<T>& statusor) {
  return statusor.status();
}

}  // namespace internal_status_macros
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

// Internal helpers for macro expansion and unparenthesizing.
#define STATUS_MACROS_IMPL_CONCAT_INNER_(x, y) x##y
#define STATUS_MACROS_IMPL_CONCAT_(x, y) STATUS_MACROS_IMPL_CONCAT_INNER_(x, y)

#define STATUS_MACROS_IMPL_EAT(...)
#define STATUS_MACROS_IMPL_REM(...) __VA_ARGS__
#define STATUS_MACROS_IMPL_COMMA(...) ,
#define STATUS_MACROS_IMPL_ARG_3(a, b, c, ...) c

#define STATUS_MACROS_IMPL_HAS_COMMA(...) \
  STATUS_MACROS_IMPL_REM(                 \
      STATUS_MACROS_IMPL_ARG_3(__VA_ARGS__, 1, 10))

#define STATUS_MACROS_IMPL_HAVE_VA_OPT(...) \
  STATUS_MACROS_IMPL_ARG_3(__VA_OPT__(, ), 1, 0, )

#if STATUS_MACROS_IMPL_HAVE_VA_OPT(.)
#define STATUS_MACROS_1_IF_EMPTY_ELSE_10(...) 1##__VA_OPT__(0)
#else
#define STATUS_MACROS_IMPL_CONCAT_5(a, b, c, d, e) a##b##c##d##e
#define STATUS_MACROS_IMPL_IS_EMPTY_CASE_1010101 ,
#define STATUS_MACROS_1_IF_1010101_ELSE_10(a, b, c, d) \
  STATUS_MACROS_IMPL_HAS_COMMA(                        \
      STATUS_MACROS_IMPL_CONCAT_5(                     \
          STATUS_MACROS_IMPL_IS_EMPTY_CASE_, a, b, c, d))
#define STATUS_MACROS_1_IF_EMPTY_ELSE_10(...)      \
  STATUS_MACROS_1_IF_1010101_ELSE_10(              \
      STATUS_MACROS_IMPL_HAS_COMMA(__VA_ARGS__),   \
      STATUS_MACROS_IMPL_HAS_COMMA(                \
          STATUS_MACROS_IMPL_COMMA __VA_ARGS__),   \
      STATUS_MACROS_IMPL_HAS_COMMA(__VA_ARGS__()), \
      STATUS_MACROS_IMPL_HAS_COMMA(                \
          STATUS_MACROS_IMPL_COMMA __VA_ARGS__()))
#endif

#define STATUS_MACROS_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED_1(x) \
  STATUS_MACROS_IMPL_REM x

#define STATUS_MACROS_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED_10(x) \
  STATUS_MACROS_IMPL_REM(x)

#define STATUS_MACROS_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED(...) \
  STATUS_MACROS_IMPL_REM(                                       \
      STATUS_MACROS_IMPL_CONCAT_(                               \
          STATUS_MACROS_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED_,  \
          STATUS_MACROS_1_IF_EMPTY_ELSE_10(                     \
              STATUS_MACROS_IMPL_EAT __VA_ARGS__))(             \
          STATUS_MACROS_IMPL_REM(__VA_ARGS__)))

#ifndef ASSIGN_OR_RETURN
#define ASSIGN_OR_RETURN(lhs, rexpr) \
  STATUS_MACROS_ASSIGN_OR_RETURN_IMPL_( \
      STATUS_MACROS_IMPL_CONCAT_(status_or_value_, __LINE__), lhs, rexpr)

#define STATUS_MACROS_ASSIGN_OR_RETURN_IMPL_(statusor, lhs, rexpr) \
  auto statusor = (rexpr); \
  if (ABSL_PREDICT_FALSE(!statusor.ok())) { \
    return std::move(statusor).status(); \
  } \
  STATUS_MACROS_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED(lhs) = *std::move(statusor)
#endif  // ASSIGN_OR_RETURN

#ifndef RETURN_IF_ERROR
#define RETURN_IF_ERROR(expr) \
  STATUS_MACROS_RETURN_IF_ERROR_IMPL_( \
      STATUS_MACROS_IMPL_CONCAT_(status_value_, __LINE__), expr)

#define STATUS_MACROS_RETURN_IF_ERROR_IMPL_(status, expr) \
  auto status = (expr); \
  if (ABSL_PREDICT_FALSE(!status.ok())) { \
    return status; \
  }
#endif  // RETURN_IF_ERROR

#ifndef ASSERT_OK
#define ASSERT_OK(expr) \
  STATUS_MACROS_ASSERT_OK_IMPL_( \
      STATUS_MACROS_IMPL_CONCAT_(status_value_, __LINE__), expr)

#define STATUS_MACROS_ASSERT_OK_IMPL_(status, expr) \
  auto status = (expr); \
  ASSERT_TRUE(status.ok()) \
      << ::private_membership::rlwe::v2::internal_status_macros::GetStatus( \
             status)
#endif  // ASSERT_OK

#ifndef EXPECT_OK
#define EXPECT_OK(expr) \
  STATUS_MACROS_EXPECT_OK_IMPL_( \
      STATUS_MACROS_IMPL_CONCAT_(status_value_, __LINE__), expr)

#define STATUS_MACROS_EXPECT_OK_IMPL_(status, expr) \
  auto status = (expr); \
  EXPECT_TRUE(status.ok()) \
      << ::private_membership::rlwe::v2::internal_status_macros::GetStatus( \
             status)
#endif  // EXPECT_OK

#ifndef ASSERT_OK_AND_ASSIGN
#define ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  STATUS_MACROS_ASSERT_OK_AND_ASSIGN_IMPL_( \
      STATUS_MACROS_IMPL_CONCAT_(status_or_value_, __LINE__), lhs, rexpr)

#define STATUS_MACROS_ASSERT_OK_AND_ASSIGN_IMPL_(statusor, lhs, rexpr) \
  auto statusor = (rexpr); \
  ASSERT_TRUE(statusor.ok()) << statusor.status(); \
  STATUS_MACROS_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED(lhs) = *std::move(statusor)
#endif  // ASSERT_OK_AND_ASSIGN

#ifndef EXPECT_OK_AND_ASSIGN
#define EXPECT_OK_AND_ASSIGN(lhs, rexpr) \
  STATUS_MACROS_EXPECT_OK_AND_ASSIGN_IMPL_( \
      STATUS_MACROS_IMPL_CONCAT_(status_or_value_, __LINE__), lhs, rexpr)

#define STATUS_MACROS_EXPECT_OK_AND_ASSIGN_IMPL_(statusor, lhs, rexpr) \
  auto statusor = (rexpr); \
  EXPECT_TRUE(statusor.ok()) << statusor.status(); \
  STATUS_MACROS_IMPL_UNPARENTHESIZE_IF_PARENTHESIZED(lhs) = *std::move(statusor)
#endif  // EXPECT_OK_AND_ASSIGN

#endif  // CRYPTO_STATUS_MACROS_H_
