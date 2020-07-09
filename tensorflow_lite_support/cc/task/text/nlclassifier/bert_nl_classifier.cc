/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_lite_support/cc/task/text/nlclassifier/bert_nl_classifier.h"

#include "absl/strings/str_format.h"
#include "tensorflow_lite_support/cc/task/core/task_api_factory.h"
#include "tensorflow_lite_support/cc/task/core/task_utils.h"
#include "tensorflow_lite_support/cc/text/tokenizers/tokenizer.h"
#include "tensorflow_lite_support/cc/text/tokenizers/tokenizer_utils.h"
#include "tensorflow_lite_support/cc/utils/common_utils.h"

namespace tflite {
namespace support {
namespace task {
namespace text {
namespace nlclassifier {

using ::tflite::support::task::core::FindTensorByName;
using ::tflite::support::task::core::PopulateTensor;
using ::tflite::support::text::tokenizer::CreateTokenizerFromMetadata;
using ::tflite::support::text::tokenizer::TokenizerResult;
using ::tflite::support::utils::LoadVocabFromBuffer;

absl::Status BertNLClassifier::Preprocess(
    const std::vector<TfLiteTensor*>& input_tensors, const std::string& input) {
  auto* input_tensor_metadatas =
      GetMetadataExtractor()->GetInputTensorMetadata();
  auto* ids_tensor =
      FindTensorByName(input_tensors, input_tensor_metadatas, kIdsTensorName);
  auto* mask_tensor =
      FindTensorByName(input_tensors, input_tensor_metadatas, kMaskTensorName);
  auto* segment_ids_tensor = FindTensorByName(
      input_tensors, input_tensor_metadatas, kSegmentIdsTensorName);

  std::string processed_input = input;
  absl::AsciiStrToLower(&processed_input);

  TokenizerResult input_tokenize_results;
  input_tokenize_results = tokenizer_->Tokenize(processed_input);

  std::vector<std::string> query_tokens = input_tokenize_results.subwords;
  if (query_tokens.size() > kMaxQueryLen) {
    query_tokens.resize(kMaxQueryLen);
  }

  std::vector<std::string> tokens;
  // 2 accounts for [CLS], [SEP]
  tokens.reserve(2 + query_tokens.size());
  std::vector<int> segment_ids;
  segment_ids.reserve(kMaxSeqLen);
  segment_ids.insert(segment_ids.end(), kMaxSeqLen, 0);

  // Start of generating the features.
  tokens.emplace_back("[CLS]");
  // For query input.
  for (const auto& query_token : query_tokens) {
    tokens.emplace_back(query_token);
  }
  // For Separation.
  tokens.emplace_back("[SEP]");

  std::vector<int> input_ids(tokens.size());
  input_ids.reserve(kMaxSeqLen);
  // Convert tokens back into ids
  for (int i = 0; i < tokens.size(); i++) {
    auto& token = tokens[i];
    tokenizer_->LookupId(token, &input_ids[i]);
  }

  std::vector<int> input_mask;
  input_mask.reserve(kMaxSeqLen);
  input_mask.insert(input_mask.end(), tokens.size(), 1);

  int zeros_to_pad = kMaxSeqLen - input_ids.size();
  input_ids.insert(input_ids.end(), zeros_to_pad, 0);
  input_mask.insert(input_mask.end(), zeros_to_pad, 0);

  PopulateTensor(input_ids, ids_tensor);
  PopulateTensor(input_mask, mask_tensor);
  PopulateTensor(segment_ids, segment_ids_tensor);

  return absl::OkStatus();
}

StatusOr<std::vector<core::Category>> BertNLClassifier::Postprocess(
    const std::vector<const TfLiteTensor*>& output_tensors,
    const std::string& /*input*/) {
  if (output_tensors.size() != 1) {
    return CreateStatusWithPayload(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat("BertNLClassifier models are expected to have only 1 "
                        "output, found %d",
                        output_tensors.size()),
        TfLiteSupportStatus::kInvalidNumOutputTensorsError);
  }
  const TfLiteTensor* scores = FindTensorByName(
      output_tensors, GetMetadataExtractor()->GetOutputTensorMetadata(),
      kScoreTensorName);

  // optional labels extracted from metadata
  return BuildResults(scores, nullptr);
}

StatusOr<std::unique_ptr<BertNLClassifier>>
BertNLClassifier::CreateBertNLClassifierWithMetadata(
    const std::string& path_to_model_with_metadata,
    std::unique_ptr<tflite::OpResolver> resolver) {
  std::unique_ptr<BertNLClassifier> bert_nl_classifier;
  ASSIGN_OR_RETURN(bert_nl_classifier,
                   core::TaskAPIFactory::CreateFromFile<BertNLClassifier>(
                       path_to_model_with_metadata, std::move(resolver)));
  RETURN_IF_ERROR(bert_nl_classifier->InitializeFromMetadata());
  return std::move(bert_nl_classifier);
}

StatusOr<std::unique_ptr<BertNLClassifier>>
BertNLClassifier::CreateBertNLClassifierWithMetadataFromBinary(
    const char* model_with_metadata_buffer_data,
    size_t model_with_metadata_buffer_size,
    std::unique_ptr<tflite::OpResolver> resolver) {
  std::unique_ptr<BertNLClassifier> bert_nl_classifier;
  ASSIGN_OR_RETURN(bert_nl_classifier,
                   core::TaskAPIFactory::CreateFromBuffer<BertNLClassifier>(
                       model_with_metadata_buffer_data,
                       model_with_metadata_buffer_size, std::move(resolver)));
  RETURN_IF_ERROR(bert_nl_classifier->InitializeFromMetadata());
  return std::move(bert_nl_classifier);
}

absl::Status BertNLClassifier::InitializeFromMetadata() {
  // Set up mandatory tokenizer.
  ASSIGN_OR_RETURN(tokenizer_,
                   CreateTokenizerFromMetadata(*GetMetadataExtractor()));

  // Set up optional label vector.
  TrySetLabelFromMetadata(
      GetMetadataExtractor()->GetOutputTensorMetadata(kOutputTensorIndex))
      .IgnoreError();
  return absl::OkStatus();
}

}  // namespace nlclassifier
}  // namespace text
}  // namespace task
}  // namespace support
}  // namespace tflite
