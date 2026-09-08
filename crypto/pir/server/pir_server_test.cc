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

#include "crypto/pir/server/pir_server.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "crypto/encryption.h"
#include "crypto/file_util.h"
#include "crypto/matrix.h"
#include "crypto/pir/client/pir_client.h"
#include "crypto/pir/pir_params.h"
#include "crypto/pir/server/pir_preprocessed_data.h"
#include "crypto/pir/server/preprocessing_server.h"
#include "crypto/polynomial.h"
#include <gmock/gmock.h>
#include "shell_encryption/testing/status_matchers.h"
#include "crypto/status_macros.h"
#include <gtest/gtest.h>
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "shell_encryption/prng/chacha_prng.h"
#include "shell_encryption/prng/prng.h"

namespace private_membership {
namespace rlwe {
namespace v2 {
namespace {

using ::testing::HasSubstr;
using ::testing::Test;
using ::rlwe::testing::StatusIs;

constexpr int kDegree = 32;
constexpr int kVariance = 4;
constexpr int kLogDigit = 2;
constexpr int kNumEntries = 1024;
struct Uint32Mod0 {
  using DbDataType = uint16_t;
  using CoeffType = uint32_t;
  using MatCoeffType = int16_t;
  static constexpr uint32_t kModulus = 0;
  static constexpr uint32_t kModulus1AfterSwitch = 1U << 31;
  static constexpr uint32_t kModulus2AfterSwitch = 1U << 30;
  static constexpr int kInterpolationDegree = 32;
};

struct Uint32Mod0Interp1 {
  using DbDataType = uint16_t;
  using CoeffType = uint32_t;
  using MatCoeffType = int16_t;
  static constexpr uint32_t kModulus = 0;
  static constexpr uint32_t kModulus1AfterSwitch = 1U << 31;
  static constexpr uint32_t kModulus2AfterSwitch = 1U << 30;
  static constexpr int kInterpolationDegree = 1;
};

struct Uint32Mod30 {
  using DbDataType = uint16_t;
  using CoeffType = uint32_t;
  using MatCoeffType = int16_t;
  static constexpr uint32_t kModulus = 1U << 30;
  static constexpr uint32_t kModulus1AfterSwitch = 1U << 29;
  static constexpr uint32_t kModulus2AfterSwitch = 1U << 28;
  static constexpr int kInterpolationDegree = 32;
};

struct Uint32Mod30Interp1 {
  using DbDataType = uint16_t;
  using CoeffType = uint32_t;
  using MatCoeffType = int16_t;
  static constexpr uint32_t kModulus = 1U << 30;
  static constexpr uint32_t kModulus1AfterSwitch = 1U << 29;
  static constexpr uint32_t kModulus2AfterSwitch = 1U << 28;
  static constexpr int kInterpolationDegree = 1;
};

struct Uint64Mod0 {
  using DbDataType = uint16_t;
  using CoeffType = uint64_t;
  using MatCoeffType = int16_t;
  static constexpr uint64_t kModulus = 0;
  static constexpr uint64_t kModulus1AfterSwitch = 1ULL << 61;
  static constexpr uint64_t kModulus2AfterSwitch = 1ULL << 60;
  static constexpr int kInterpolationDegree = 32;
};

struct Uint64Mod0Interp1 {
  using DbDataType = uint16_t;
  using CoeffType = uint64_t;
  using MatCoeffType = int16_t;
  static constexpr uint64_t kModulus = 0;
  static constexpr uint64_t kModulus1AfterSwitch = 1ULL << 61;
  static constexpr uint64_t kModulus2AfterSwitch = 1ULL << 60;
  static constexpr int kInterpolationDegree = 1;
};

struct Uint64Mod62 {
  using DbDataType = uint16_t;
  using CoeffType = uint64_t;
  using MatCoeffType = int32_t;
  static constexpr uint64_t kModulus = 1ULL << 62;
  static constexpr uint64_t kModulus1AfterSwitch = 1ULL << 61;
  static constexpr uint64_t kModulus2AfterSwitch = 1ULL << 60;
  static constexpr int kInterpolationDegree = 32;
};

struct Uint64Mod62ModSwitch {
  using DbDataType = uint16_t;
  using CoeffType = uint64_t;
  using MatCoeffType = int32_t;
  static constexpr uint64_t kModulus = 1ULL << 62;
  static constexpr uint64_t kModulus1AfterSwitch = 1ULL << 28;
  static constexpr uint64_t kModulus2AfterSwitch = 1ULL << 20;
  static constexpr int kInterpolationDegree = 32;
};

struct Uint64Mod62Interp1 {
  using DbDataType = uint16_t;
  using CoeffType = uint64_t;
  using MatCoeffType = int32_t;
  static constexpr uint64_t kModulus = 1ULL << 62;
  static constexpr uint64_t kModulus1AfterSwitch = 1ULL << 61;
  static constexpr uint64_t kModulus2AfterSwitch = 1ULL << 60;
  static constexpr int kInterpolationDegree = 1;
};

template <typename PirTestType>
class PirServerTest : public Test {
 public:
  using DbDataType = typename PirTestType::DbDataType;
  using CoeffType = typename PirTestType::CoeffType;
  using MatCoeffType = typename PirTestType::MatCoeffType;

 protected:
  PirServerTest() = default;

  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        auto rlwe_params,
        RlweParams<CoeffType>::Create(
            kDegree, PirTestType::kModulus, PirTestType::kModulus1AfterSwitch,
            PirTestType::kModulus2AfterSwitch, kVariance));
    rlwe_params_ =
        std::make_unique<RlweParams<CoeffType>>(std::move(rlwe_params));
    ASSERT_OK_AND_ASSIGN(auto params, this->CreateValidPirParams());
    params_ = std::make_unique<PirParams<CoeffType>>(std::move(params));

    const int d = params_->RlweParameters().Degree();
    const int n = params_->NumEntries();

    ASSERT_OK_AND_ASSIGN(auto db_prng, this->CreatePrng());
    db_raw_.resize(n * d);
    for (int i = 0; i < n * d; ++i) {
      ASSERT_OK_AND_ASSIGN(auto rand_val, db_prng->Rand8());
      db_raw_[i] = rand_val % params_->PlaintextModulus();
    }

    ASSERT_OK_AND_ASSIGN(
        auto preprocessing_server,
        (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
            params_.get())));
    prng_seed_ = preprocessing_server->GetPrngSeed();

    ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                         preprocessing_server->Preprocess(db_raw_));
    ASSERT_OK_AND_ASSIGN(
        auto server, (PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
                         *params_, std::move(preprocessed_data))));
    server_ = std::move(server);
  }

  CoeffType GetPlaintextModulus() const { return 257; }

  GadgetParams GetPackGadgetParams() const {
    int log_q = PirTestType::kModulus == 0
                    ? sizeof(CoeffType) * 8
                    : absl::bit_width(PirTestType::kModulus - 1);
    return GadgetParams{/*log_digit=*/kLogDigit,
                        /*num_digits=*/(log_q + kLogDigit - 1) / kLogDigit - 1};
  }

  GadgetParams GetPolyEvalGadgetParams() const {
    int log_q = PirTestType::kModulus == 0
                    ? sizeof(CoeffType) * 8
                    : absl::bit_width(PirTestType::kModulus - 1);
    return GadgetParams{/*log_digit=*/kLogDigit,
                        /*num_digits=*/(log_q + kLogDigit - 1) / kLogDigit - 1};
  }

  absl::StatusOr<PirParams<CoeffType>> CreateValidPirParams() const {
    return PirParams<CoeffType>::Create(*rlwe_params_, GetPlaintextModulus(),
                                        GetPackGadgetParams(),
                                        GetPolyEvalGadgetParams(), kNumEntries,
                                        PirTestType::kInterpolationDegree);
  }

  absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng() const {
    ASSIGN_OR_RETURN(auto prng_seed, ::rlwe::ChaChaPrng::GenerateSeed());
    return ::rlwe::ChaChaPrng::Create(prng_seed);
  }

  absl::StatusOr<std::unique_ptr<::rlwe::SecurePrng>> CreatePrng(
      absl::string_view prng_seed) const {
    return ::rlwe::ChaChaPrng::Create(prng_seed);
  }

  absl::StatusOr<std::unique_ptr<PirClient<CoeffType>>> CreatePirClient()
      const {
    ASSIGN_OR_RETURN(auto client_a_prng, CreatePrng(prng_seed_));
    ASSIGN_OR_RETURN(auto client_prng, CreatePrng());
    return PirClient<CoeffType>::Create(*params_, std::move(client_a_prng),
                                        std::move(client_prng));
  }

  std::unique_ptr<RlweParams<CoeffType>> rlwe_params_;
  std::unique_ptr<PirParams<CoeffType>> params_;
  std::string prng_seed_;
  std::unique_ptr<PirServer<DbDataType, CoeffType, MatCoeffType>> server_;
  std::vector<DbDataType> db_raw_;
};

using PirTestTypes =
    ::testing::Types<Uint32Mod0, Uint32Mod30, Uint64Mod0, Uint64Mod62,
                     Uint64Mod62ModSwitch, Uint32Mod0Interp1,
                     Uint32Mod30Interp1, Uint64Mod0Interp1, Uint64Mod62Interp1>;
TYPED_TEST_SUITE(PirServerTest, PirTestTypes);

TYPED_TEST(PirServerTest, EndToEndTest) {
  using CoeffType = typename TypeParam::CoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int i = 0; i < num_entries; i += 53) {
    test_indices.push_back(i);
  }
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client, this->CreatePirClient());
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
    if (TypeParam::kInterpolationDegree == 1) {
      EXPECT_TRUE(request.second_dimension_query.empty());
    }

    ASSERT_OK_AND_ASSIGN(auto response,
                         this->server_->ProcessResponse(request));

    ASSERT_OK_AND_ASSIGN(auto recovered_message,
                         client->ProcessResponse(response));

    std::vector<CoeffType> expected_message(
        this->db_raw_.begin() + test_index * d,
        this->db_raw_.begin() + (test_index + 1) * d);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(PirServerTest, EndToEndTestWithLargeEntrySizeMultiple) {
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  int d = this->params_->RlweParameters().Degree();
  int num_entries = this->params_->NumEntries();
  int entry_size_multiple = 3;

  ASSERT_OK_AND_ASSIGN(
      auto params,
      PirParams<CoeffType>::Create(
          this->params_->RlweParameters(), this->GetPlaintextModulus(),
          this->GetPackGadgetParams(), this->GetPolyEvalGadgetParams(),
          num_entries, this->params_->InterpolationDegree(),
          entry_size_multiple));

  ASSERT_OK_AND_ASSIGN(auto db_prng, this->CreatePrng());
  std::vector<typename TypeParam::DbDataType> db_raw(num_entries * d *
                                                     entry_size_multiple);
  for (int i = 0; i < num_entries * d * entry_size_multiple; ++i) {
    ASSERT_OK_AND_ASSIGN(auto rand_val, db_prng->Rand8());
    db_raw[i] = rand_val % params.PlaintextModulus();
  }

  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (PreprocessingServer<typename TypeParam::DbDataType, CoeffType,
                           MatCoeffType>::Create(&params)));

  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       preprocessing_server->Preprocess(db_raw));

  ASSERT_OK_AND_ASSIGN(
      auto server,
      (PirServer<typename TypeParam::DbDataType, CoeffType,
                 MatCoeffType>::Create(params, std::move(preprocessed_data))));

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int i = 0; i < num_entries; i += 53) {
    test_indices.push_back(i);
  }
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                         this->CreatePrng(preprocessing_server->GetPrngSeed()));
    ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
    ASSERT_OK_AND_ASSIGN(auto client, PirClient<CoeffType>::Create(
                                          params, std::move(client_a_prng),
                                          std::move(client_prng)));
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
    if (TypeParam::kInterpolationDegree == 1) {
      EXPECT_TRUE(request.second_dimension_query.empty());
    }

    ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));

    ASSERT_OK_AND_ASSIGN(auto recovered_message,
                         client->ProcessResponse(response));

    std::vector<CoeffType> expected_message(
        db_raw.begin() + test_index * d * entry_size_multiple,
        db_raw.begin() + (test_index + 1) * d * entry_size_multiple);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(PirServerTest, ProcessResponseFailsWhenPackingKeyIsInvalidLength) {
  int target_index = 35;
  ASSERT_OK_AND_ASSIGN(auto client, this->CreatePirClient());
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(target_index));
  request.packing_key.pop_back();

  EXPECT_THAT(this->server_->ProcessResponse(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Packing key must contain")));
}

TYPED_TEST(PirServerTest, EndToEndTestWithNonRingDegreeMultipleNumEntries) {
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  int d = this->params_->RlweParameters().Degree();
  int t = this->params_->InterpolationDegree();
  // num_entries is a multiple of t, but (num_entries / t) is not a multiple of
  // ring degree d.
  int num_entries = 50 * t;

  ASSERT_OK_AND_ASSIGN(
      auto params,
      PirParams<CoeffType>::Create(
          this->params_->RlweParameters(), this->GetPlaintextModulus(),
          this->GetPackGadgetParams(), this->GetPolyEvalGadgetParams(),
          num_entries, t, /*entry_size_multiple=*/1));

  ASSERT_OK_AND_ASSIGN(auto db_prng, this->CreatePrng());
  std::vector<typename TypeParam::DbDataType> db_raw(num_entries * d);
  for (int i = 0; i < num_entries * d; ++i) {
    ASSERT_OK_AND_ASSIGN(auto rand_val, db_prng->Rand8());
    db_raw[i] = rand_val % params.PlaintextModulus();
  }

  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (PreprocessingServer<typename TypeParam::DbDataType, CoeffType,
                           MatCoeffType>::Create(&params)));

  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       preprocessing_server->Preprocess(db_raw));

  ASSERT_OK_AND_ASSIGN(
      auto server,
      (PirServer<typename TypeParam::DbDataType, CoeffType,
                 MatCoeffType>::Create(params, std::move(preprocessed_data))));

  std::vector<int> test_indices = {0, 1, 10, num_entries - 2, num_entries - 1};
  for (int test_index : test_indices) {
    ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                         this->CreatePrng(preprocessing_server->GetPrngSeed()));
    ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
    ASSERT_OK_AND_ASSIGN(auto client, PirClient<CoeffType>::Create(
                                          params, std::move(client_a_prng),
                                          std::move(client_prng)));
    ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
    EXPECT_EQ(request.first_dimension_query.size(), num_entries / t);

    ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));
    ASSERT_OK_AND_ASSIGN(auto recovered_message,
                         client->ProcessResponse(response));

    std::vector<CoeffType> expected_message(
        db_raw.begin() + test_index * d,
        db_raw.begin() + (test_index + 1) * d);

    EXPECT_EQ(recovered_message, expected_message)
        << "Failed at target_index = " << test_index;
  }
}

TYPED_TEST(PirServerTest,
           ProcessResponseFailsWhenFirstDimensionQueryIsInvalidLength) {
  int target_index = 35;
  ASSERT_OK_AND_ASSIGN(auto client, this->CreatePirClient());
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(target_index));
  if (!request.first_dimension_query.empty()) {
    request.first_dimension_query.pop_back();
    EXPECT_THAT(this->server_->ProcessResponse(request),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("First dimension query size")));
  }
}

TYPED_TEST(PirServerTest,
           ProcessResponseFailsWhenFirstDimensionQueryIsTooLarge) {
  int target_index = 35;
  ASSERT_OK_AND_ASSIGN(auto client, this->CreatePirClient());
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(target_index));
  request.first_dimension_query.resize(100000, 0);
  EXPECT_THAT(this->server_->ProcessResponse(request),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("First dimension query size")));
}

TYPED_TEST(PirServerTest,
           ProcessResponseFailsWhenSecondDimensionQueryIsInvalidLength) {
  int target_index = 35;
  ASSERT_OK_AND_ASSIGN(auto client, this->CreatePirClient());
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(target_index));
  if (!request.second_dimension_query.empty()) {
    request.second_dimension_query.pop_back();
    EXPECT_THAT(this->server_->ProcessResponse(request),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("second_dimension_query size mismatch")));
  }
}

TYPED_TEST(PirServerTest, CreateFailsWhenDbMatrixInvalidDimensions) {
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (PreprocessingServer<uint16_t, CoeffType, MatCoeffType>::Create(
          this->params_.get())));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       preprocessing_server->Preprocess(this->db_raw_));

  PirPreprocessedData<uint16_t, CoeffType, MatCoeffType>
      invalid_preprocessed_data1 = {
          .db_matrices = preprocessed_data.db_matrices,
          .preprocessed_outputs = preprocessed_data.preprocessed_outputs,
          .second_dim_a = preprocessed_data.second_dim_a};
  ASSERT_OK_AND_ASSIGN(
      invalid_preprocessed_data1.db_matrices[0],
      Matrix<uint16_t>::Create(1, this->params_->NumEntries() /
                                      this->params_->InterpolationDegree()));
  EXPECT_THAT((PirServer<uint16_t, CoeffType, MatCoeffType>::Create(
                  *this->params_, std::move(invalid_preprocessed_data1))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("DB matrix must have td rows.")));

  PirPreprocessedData<uint16_t, CoeffType, MatCoeffType>
      invalid_preprocessed_data2 = {
          .db_matrices = preprocessed_data.db_matrices,
          .preprocessed_outputs = preprocessed_data.preprocessed_outputs,
          .second_dim_a = preprocessed_data.second_dim_a};
  ASSERT_OK_AND_ASSIGN(
      invalid_preprocessed_data2.db_matrices[0],
      Matrix<uint16_t>::Create(this->params_->InterpolationDegree() *
                                   this->params_->RlweParameters().Degree(),
                               1));
  EXPECT_THAT((PirServer<uint16_t, CoeffType, MatCoeffType>::Create(
                  *this->params_, std::move(invalid_preprocessed_data2))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("DB matrix must have at least N/t columns.")));
}

TYPED_TEST(PirServerTest, CreateFailsWhenPreprocessedOutputsLengthIsInvalid) {
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (PreprocessingServer<uint16_t, CoeffType, MatCoeffType>::Create(
          this->params_.get())));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       preprocessing_server->Preprocess(this->db_raw_));
  preprocessed_data.preprocessed_outputs[0].pop_back();
  EXPECT_THAT((PirServer<uint16_t, CoeffType, MatCoeffType>::Create(
                  *this->params_, std::move(preprocessed_data))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Preprocessed outputs must be of length t")));
}

TYPED_TEST(PirServerTest, CreateFailsWhenSecondDimQueryALengthIsInvalid) {
  if (TypeParam::kInterpolationDegree == 1) {
    GTEST_SKIP() << "second_dim_a size is not validated when "
                    "interpolation degree is 1.";
  }
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (PreprocessingServer<uint16_t, CoeffType, MatCoeffType>::Create(
          this->params_.get())));
  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       preprocessing_server->Preprocess(this->db_raw_));
  preprocessed_data.second_dim_a.pop_back();
  EXPECT_THAT((PirServer<uint16_t, CoeffType, MatCoeffType>::Create(
                  *this->params_, std::move(preprocessed_data))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("second_dim_query_a must have length 2")));
}

TYPED_TEST(PirServerTest, SerializationDeserializationTest) {
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  using DbDataType = typename TypeParam::DbDataType;
  int d = this->params_->RlweParameters().Degree();

  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          this->params_.get())));

  ASSERT_OK_AND_ASSIGN(auto preprocessed_data1,
                       preprocessing_server->Preprocess(this->db_raw_));

  ASSERT_OK_AND_ASSIGN(auto proto, SerializePirPreprocessedData(
                                       preprocessed_data1, *this->params_));

  ASSERT_OK_AND_ASSIGN(
      (PirPreprocessedData<DbDataType, CoeffType, MatCoeffType>
           preprocessed_data2),
      (DeserializePirPreprocessedData<DbDataType, CoeffType, MatCoeffType>(
          proto, *this->params_)));

  ASSERT_OK_AND_ASSIGN(auto server,
                       (PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
                           *this->params_, std::move(preprocessed_data2))));

  int test_index = 5;
  ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                       this->CreatePrng(preprocessing_server->GetPrngSeed()));
  ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
  ASSERT_OK_AND_ASSIGN(
      auto client,
      PirClient<CoeffType>::Create(*this->params_, std::move(client_a_prng),
                                   std::move(client_prng)));
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
  ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));
  ASSERT_OK_AND_ASSIGN(auto recovered_message,
                       client->ProcessResponse(response));

  std::vector<CoeffType> expected_message(
      this->db_raw_.begin() + test_index * d,
      this->db_raw_.begin() + (test_index + 1) * d);

  EXPECT_EQ(recovered_message, expected_message);
}

TYPED_TEST(PirServerTest, FileSaveAndLoadTest) {
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  using DbDataType = typename TypeParam::DbDataType;
  int d = this->params_->RlweParameters().Degree();

  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          this->params_.get())));

  ASSERT_OK_AND_ASSIGN(auto preprocessed_data1,
                       preprocessing_server->Preprocess(this->db_raw_));

  std::string file_path =
      file_util::JoinPath(testing::TempDir(), "preprocessed_data.bin");

  // Save the preprocessed database and material to file
  ASSERT_OK(SavePirPreprocessedDataToFile(preprocessed_data1, *this->params_,
                                          file_path));

  // Load the preprocessed database and material from file
  ASSERT_OK_AND_ASSIGN(
      (PirPreprocessedData<DbDataType, CoeffType, MatCoeffType>
           preprocessed_data2),
      (LoadPirPreprocessedDataFromFile<DbDataType, CoeffType, MatCoeffType>(
          *this->params_, file_path)));

  // Start the server from the loaded database and preprocessed material
  ASSERT_OK_AND_ASSIGN(auto server,
                       (PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
                           *this->params_, std::move(preprocessed_data2))));

  int test_index = 5;
  ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                       this->CreatePrng(preprocessing_server->GetPrngSeed()));
  ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
  ASSERT_OK_AND_ASSIGN(
      auto client,
      PirClient<CoeffType>::Create(*this->params_, std::move(client_a_prng),
                                   std::move(client_prng)));
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
  ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));
  ASSERT_OK_AND_ASSIGN(auto recovered_message,
                       client->ProcessResponse(response));

  std::vector<CoeffType> expected_message(
      this->db_raw_.begin() + test_index * d,
      this->db_raw_.begin() + (test_index + 1) * d);

  EXPECT_EQ(recovered_message, expected_message);
}

TYPED_TEST(PirServerTest, ServerDirectFileSaveAndLoadTest) {
  using CoeffType = typename TypeParam::CoeffType;
  using MatCoeffType = typename TypeParam::MatCoeffType;
  using DbDataType = typename TypeParam::DbDataType;
  int d = this->params_->RlweParameters().Degree();

  ASSERT_OK_AND_ASSIGN(
      auto preprocessing_server,
      (PreprocessingServer<DbDataType, CoeffType, MatCoeffType>::Create(
          this->params_.get())));

  ASSERT_OK_AND_ASSIGN(auto preprocessed_data,
                       preprocessing_server->Preprocess(this->db_raw_));

  ASSERT_OK_AND_ASSIGN(auto initial_server,
                       (PirServer<DbDataType, CoeffType, MatCoeffType>::Create(
                           *this->params_, std::move(preprocessed_data))));

  std::string file_path =
      file_util::JoinPath(testing::TempDir(), "server_direct_data.bin");

  // Save the server directly to file
  ASSERT_OK(initial_server->SaveToFile(file_path));

  // Load the server from file
  ASSERT_OK_AND_ASSIGN(
      auto server,
      (PirServer<DbDataType, CoeffType, MatCoeffType>::LoadFromFile(
          *this->params_, file_path)));

  int test_index = 5;
  ASSERT_OK_AND_ASSIGN(auto client_a_prng,
                       this->CreatePrng(preprocessing_server->GetPrngSeed()));
  ASSERT_OK_AND_ASSIGN(auto client_prng, this->CreatePrng());
  ASSERT_OK_AND_ASSIGN(
      auto client,
      PirClient<CoeffType>::Create(*this->params_, std::move(client_a_prng),
                                   std::move(client_prng)));
  ASSERT_OK_AND_ASSIGN(auto request, client->CreateRequest(test_index));
  ASSERT_OK_AND_ASSIGN(auto response, server->ProcessResponse(request));
  ASSERT_OK_AND_ASSIGN(auto recovered_message,
                       client->ProcessResponse(response));

  std::vector<CoeffType> expected_message(
      this->db_raw_.begin() + test_index * d,
      this->db_raw_.begin() + (test_index + 1) * d);

  EXPECT_EQ(recovered_message, expected_message);
}
}  // namespace
}  // namespace v2
}  // namespace rlwe
}  // namespace private_membership
