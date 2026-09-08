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

#include "crypto/file_util.h"

#include <string>

#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "shell_encryption/testing/status_testing.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace file_util {
namespace {

using ::testing::Eq;

TEST(FileUtilTest, JoinPath) {
  EXPECT_THAT(JoinPath("/tmp/foo", "bar.txt"), Eq("/tmp/foo/bar.txt"));
}

TEST(FileUtilTest, ReadWriteFileContents) {
  std::string dir = JoinPath(testing::TempDir(), "file_util_test");
  ASSERT_OK(RecursivelyCreateDir(dir));

  std::string file_path = JoinPath(dir, "test.txt");
  std::string expected_content = "Hello, Private Membership RLWE!";

  ASSERT_OK(SetFileContents(file_path, expected_content));

  std::string read_content;
  ASSERT_OK(GetFileContents(file_path, &read_content));
  EXPECT_THAT(read_content, Eq(expected_content));
}

TEST(FileUtilTest, GetNonExistentFileReturnsError) {
  std::string non_existent = JoinPath(testing::TempDir(), "does_not_exist.bin");
  std::string read_content;
  EXPECT_FALSE(GetFileContents(non_existent, &read_content).ok());
}

}  // namespace
}  // namespace file_util
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
