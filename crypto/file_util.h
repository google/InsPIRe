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

#ifndef CRYPTO_FILE_UTIL_H_
#define CRYPTO_FILE_UTIL_H_

#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace file_util {

// Reads the entire content of `file_path` into `contents`.
absl::Status GetFileContents(absl::string_view file_path,
                             std::string* contents);

// Writes `contents` to `file_path` (overwriting existing content).
absl::Status SetFileContents(absl::string_view file_path,
                             absl::string_view contents);

// Creates directory and all parent directories if they do not already exist.
absl::Status RecursivelyCreateDir(absl::string_view dir_path);

// Joins directory path and filename.
std::string JoinPath(absl::string_view dir_path,
                     absl::string_view file_name);

}  // namespace file_util
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership

#endif  // CRYPTO_FILE_UTIL_H_
