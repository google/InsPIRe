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

#include <filesystem>    // NOLINT(build/c++17)
#include <fstream>
#include <string>
#include <system_error>  // NOLINT(build/c++11)

#include "crypto/file_util.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace file_util {

absl::Status GetFileContents(absl::string_view file_path,
                             std::string* contents) {
  std::ifstream in(std::string(file_path), std::ios::binary | std::ios::ate);
  if (!in.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to open file for reading: ", file_path));
  }
  std::streamsize size = in.tellg();
  if (size < 0) {
    return absl::InternalError(
        absl::StrCat("Failed to determine file size: ", file_path));
  }
  in.seekg(0, std::ios::beg);
  contents->resize(static_cast<size_t>(size));
  if (size > 0 && !in.read(contents->data(), size)) {
    return absl::InternalError(
        absl::StrCat("Failed to read file: ", file_path));
  }
  return absl::OkStatus();
}

absl::Status SetFileContents(absl::string_view file_path,
                             absl::string_view contents) {
  std::ofstream out(std::string(file_path),
                    std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    return absl::InternalError(
        absl::StrCat("Failed to open file for writing: ", file_path));
  }
  out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!out.good()) {
    return absl::InternalError(
        absl::StrCat("Failed to write to file: ", file_path));
  }
  return absl::OkStatus();
}

absl::Status RecursivelyCreateDir(absl::string_view dir_path) {
  std::error_code ec;
  std::filesystem::create_directories(std::string(dir_path), ec);
  if (ec) {
    return absl::InternalError(
        absl::StrCat("Failed to create directory ", dir_path, ": ",
                     ec.message()));
  }
  return absl::OkStatus();
}

std::string JoinPath(absl::string_view dir_path,
                     absl::string_view file_name) {
  return (std::filesystem::path(dir_path) / file_name).string();
}

}  // namespace file_util
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
