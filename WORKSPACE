# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

workspace(name = "com_google_rlwe_v2_crypto")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Bazel Skylib
http_archive(
    name = "bazel_skylib",
    sha256 = "bc283cdfcd526a52c3201279cda4bc3815526643c6f99016173703ba4d843098",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
bazel_skylib_workspace()

# Rules CC
http_archive(
    name = "rules_cc",
    sha256 = "2037875b8ffc4d1cc1ea5025004e9367497758bee04699a7730fa3884cb185ea",
    strip_prefix = "rules_cc-0.0.9",
    urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.0.9/rules_cc-0.0.9.tar.gz"],
)

load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies", "rules_cc_toolchains")
rules_cc_dependencies()
rules_cc_toolchains()

# Rules Proto
http_archive(
    name = "rules_proto",
    sha256 = "dc3fb206a2cb3441b485eb1e423165b231235a1ea9b031b4433cf7bc1fa460dd",
    strip_prefix = "rules_proto-5.3.0-21.7",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/5.3.0-21.7.tar.gz",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
rules_proto_dependencies()
rules_proto_toolchains()

# Google Abseil C++
http_archive(
    name = "com_google_absl",
    sha256 = "f50e5ac311a81382da7fa75b97310e4b9006474f95600469e80a561088073005",
    strip_prefix = "abseil-cpp-20240116.2",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20240116.2.tar.gz"],
)

# Google Protobuf
http_archive(
    name = "com_google_protobuf",
    sha256 = "75beab32352ab60ec7852a20b7f1680d5ca0d2f1f0e2279148d44e5421c606de",
    strip_prefix = "protobuf-25.3",
    urls = ["https://github.com/protocolbuffers/protobuf/releases/download/v25.3/protobuf-25.3.tar.gz"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()

# Google Shell Encryption (RLWE library)
http_archive(
    name = "com_github_google_shell_encryption",
    sha256 = "0a61189b875d1c15f88cbf19c770cdf540ee003ed11c853075e96dadc4dbcb32",
    strip_prefix = "shell-encryption-master",
    urls = ["https://github.com/google/shell-encryption/archive/refs/heads/master.tar.gz"],
)

# Google Test
http_archive(
    name = "com_google_googletest",
    sha256 = "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7",
    strip_prefix = "googletest-1.14.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"],
)

# Google Benchmark
http_archive(
    name = "com_google_benchmark",
    sha256 = "6bc180a57d23d4d9515519f92b0c83d61b05b5bab188961f08a9d7047293e6ce",
    strip_prefix = "benchmark-1.8.3",
    urls = ["https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz"],
)

# Google Highway SIMD Library
http_archive(
    name = "highway",
    sha256 = "c9a0b127ffba106fa536098696ab1444855938bf8c83e16b9dd557a53e4b09ff",
    strip_prefix = "highway-1.1.0",
    urls = ["https://github.com/google/highway/archive/refs/tags/1.1.0.tar.gz"],
)

# FFTW3 Library (System dependency mapping)
new_local_repository(
    name = "fftw",
    path = "/usr",
    build_file_content = """
load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "fftw3",
    hdrs = glob(["include/**/fftw3*.h", "include/fftw3*.h"], allow_empty = True),
    includes = ["include"],
    linkopts = ["-lfftw3"],
    visibility = ["//visibility:public"],
)
""",
)
