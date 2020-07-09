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

#ifndef TENSORFLOW_LITE_SUPPORT_CC_TASK_TEXT_NLCLASSIFIER_BERT_NL_CLASSIFIER_H_
#define TENSORFLOW_LITE_SUPPORT_CC_TASK_TEXT_NLCLASSIFIER_BERT_NL_CLASSIFIER_H_

#include "tensorflow_lite_support/cc/task/text/nlclassifier/nl_classifier.h"
#include "tensorflow_lite_support/cc/text/tokenizers/tokenizer.h"

namespace tflite {
namespace support {
namespace task {
namespace text {
namespace nlclassifier {

// Classifier API for NLClassification tasks with Bert models, categorizes
// string into different classes.
//
// The API expects a Bert based TFLite model with metadata populated.
// The metadata should contain the following information:
//   - input_process_units for Wordpiece/Sentencepiece Tokenizer
//   - 3 input tensors with names "ids", "mask" and "segment_ids"
//   - 1 output tensor of type float32[1, 2], with a optionally attached label
//     file. If a label file is attached, the file should be a plain text file
//     with one label per line, the number of labels should match the number of
//     categories the model outputs.
// A suitable model is:
//   path/to/model

class BertNLClassifier : public NLClassifier {
 public:
  static constexpr int kMaxQueryLen = 64;
  static constexpr int kMaxSeqLen = 128;
  static constexpr char kIdsTensorName[] = "ids";
  static constexpr char kMaskTensorName[] = "mask";
  static constexpr char kSegmentIdsTensorName[] = "segment_ids";
  static constexpr char kScoreTensorName[] = "probability";
  using NLClassifier::NLClassifier;

  static StatusOr<std::unique_ptr<BertNLClassifier>>
  CreateBertNLClassifierWithMetadata(
      const std::string& path_to_model_with_metadata,
      std::unique_ptr<tflite::OpResolver> resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());

  static StatusOr<std::unique_ptr<BertNLClassifier>>
  CreateBertNLClassifierWithMetadataFromBinary(
      const char* model_with_metadata_buffer_data,
      size_t model_with_metadata_buffer_size,
      std::unique_ptr<tflite::OpResolver> resolver =
          absl::make_unique<tflite::ops::builtin::BuiltinOpResolver>());

 protected:
  absl::Status Preprocess(const std::vector<TfLiteTensor*>& input_tensors,
                          const std::string& input) override;

  StatusOr<std::vector<core::Category>> Postprocess(
      const std::vector<const TfLiteTensor*>& output_tensors,
      const std::string& input) override;

 private:
  // Initialize the API with the tokenizer and label files set in the metadata.
  absl::Status InitializeFromMetadata();
  std::unique_ptr<tflite::support::text::tokenizer::Tokenizer> tokenizer_;
};

}  // namespace nlclassifier
}  // namespace text
}  // namespace task
}  // namespace support
}  // namespace tflite

#endif  // TENSORFLOW_LITE_SUPPORT_CC_TASK_TEXT_NLCLASSIFIER_BERT_NL_CLASSIFIER_H_
