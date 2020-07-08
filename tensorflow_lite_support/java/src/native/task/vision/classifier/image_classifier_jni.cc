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

#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/vision/core/frame_buffer.h"
#include "tensorflow_lite_support/cc/task/vision/image_classifier.h"
#include "tensorflow_lite_support/cc/task/vision/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_classifier_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/cc/utils/jni_utils.h"

namespace {

using ::tflite::support::StatusOr;
using ::tflite::support::task::vision::ClassificationResult;
using ::tflite::support::task::vision::Classifications;
using ::tflite::support::task::vision::FrameBuffer;
using ::tflite::support::task::vision::ImageClassifier;
using ::tflite::support::task::vision::ImageClassifierOptions;
using ::tflite::support::utils::GetMappedFileBuffer;
using ::tflite::support::utils::kAssertionError;
using ::tflite::support::utils::ThrowException;

std::vector<std::string> ParseListOfStrings(JNIEnv* env, jobject arrayList) {
  jclass listClass = env->FindClass("Ljava/util/List;");
  jmethodID list_size = env->GetMethodID(listClass, "size", "()I");
  jmethodID list_get =
      env->GetMethodID(listClass, "get", "(I)Ljava/lang/Object;");
  jint len = env->CallIntMethod(arrayList, list_size);
  std::vector<std::string> result;
  result.reserve(len);
  for (jint i = 0; i < len; i++) {
    jstring element =
        static_cast<jstring>(env->CallObjectMethod(arrayList, list_get, i));
    const char* pchars = env->GetStringUTFChars(element, nullptr);
    result.emplace_back(pchars);
    env->ReleaseStringUTFChars(element, pchars);
    env->DeleteLocalRef(element);
  }

  return result;
}

// Creates an ImageClassifierOptions in proto based on the one in Java.
ImageClassifierOptions ConvertToProtoOptions(JNIEnv* env,
                                             jobject java_options) {
  ImageClassifierOptions proto_options;
  jclass java_options_class = env->FindClass(
      "org/tensorflow/lite/task/vision/classifier/"
      "ImageClassifier$ImageClassifierOptions");

  jfieldID display_names_locale_id = env->GetFieldID(
      java_options_class, "displayNamesLocale", "Ljava/lang/String;");
  jstring display_names_locale = static_cast<jstring>(
      env->GetObjectField(java_options, display_names_locale_id));
  const char* pchars = env->GetStringUTFChars(display_names_locale, nullptr);
  proto_options.set_display_names_locale(pchars);
  env->ReleaseStringUTFChars(display_names_locale, pchars);

  jfieldID max_results_id =
      env->GetFieldID(java_options_class, "maxResults", "I");
  jint max_results = env->GetIntField(java_options, max_results_id);
  proto_options.set_max_results(max_results);

  jfieldID is_score_threshold_set_id =
      env->GetFieldID(java_options_class, "isScoreThresholdSet", "Z");
  jboolean is_score_threshold_set =
      env->GetBooleanField(java_options, is_score_threshold_set_id);
  if (is_score_threshold_set) {
    jfieldID score_threshold_id =
        env->GetFieldID(java_options_class, "scoreThreshold", "F");
    jfloat score_threshold =
        env->GetFloatField(java_options, score_threshold_id);
    proto_options.set_score_threshold(score_threshold);
  }

  jfieldID white_list_id = env->GetFieldID(
      java_options_class, "classNameWhiteList", "Ljava/util/List;");
  jobject white_list = env->GetObjectField(java_options, white_list_id);
  auto white_list_vector = ParseListOfStrings(env, white_list);
  for (const auto& class_name : white_list_vector) {
    proto_options.add_class_name_whitelist(class_name);
  }

  jfieldID black_list_id = env->GetFieldID(
      java_options_class, "classNameBlackList", "Ljava/util/List;");
  jobject black_list = env->GetObjectField(java_options, black_list_id);
  auto black_list_vector = ParseListOfStrings(env, black_list);
  for (const auto& class_name : black_list_vector) {
    proto_options.add_class_name_blacklist(class_name);
  }

  return proto_options;
}

jobject ConvertToClassificationResults(JNIEnv* env,
                                       const ClassificationResult& results) {
  // jclass and init of Classifications.
  jclass classifications_class = env->FindClass(
      "org/tensorflow/lite/task/vision/classifier/Classifications");
  jmethodID classifications_create =
      env->GetStaticMethodID(classifications_class, "create",
                             "(Ljava/util/List;I)Lorg/tensorflow/lite/"
                             "task/vision/classifier/Classifications;");

  // jclass and init of Category.
  jclass catrgory_class =
      env->FindClass("org/tensorflow/lite/support/label/Category");
  jmethodID catrgory_init =
      env->GetMethodID(catrgory_class, "<init>", "(Ljava/lang/String;F)V");

  // jclass, init, and add of ArrayList.
  jclass array_list_class = env->FindClass("java/util/ArrayList");
  jmethodID array_list_init =
      env->GetMethodID(array_list_class, "<init>", "(I)V");
  jmethodID array_list_add_method =
      env->GetMethodID(array_list_class, "add", "(Ljava/lang/Object;)Z");

  jobject classifications_list =
      env->NewObject(array_list_class, array_list_init,
                     static_cast<jint>(results.classifications_size()));
  for (int i = 0; i < results.classifications_size(); i++) {
    auto classifications = results.classifications(i);
    jobject category_list = env->NewObject(array_list_class, array_list_init,
                                           classifications.classes_size());
    for (int j = 0; j < classifications.classes_size(); j++) {
      auto category = classifications.classes(j);
      std::string label = category.display_name().empty()
                              ? category.class_name()
                              : category.display_name();
      jstring class_name = env->NewStringUTF(label.c_str());
      jobject jcategory = env->NewObject(catrgory_class, catrgory_init,
                                         class_name, (float)category.score());
      env->DeleteLocalRef(class_name);
      env->CallBooleanMethod(category_list, array_list_add_method, jcategory);
    }
    jobject jclassifications = env->CallStaticObjectMethod(
        classifications_class, classifications_create, category_list,
        classifications.head_index());
    env->CallBooleanMethod(classifications_list, array_list_add_method,
                           jclassifications);
  }
  return classifications_list;
}

extern "C" JNIEXPORT void JNICALL
Java_org_tensorflow_lite_task_core_BaseTaskApi_deinitJni(JNIEnv* env,
                                                         jobject thiz,
                                                         jlong native_handle) {
  delete reinterpret_cast<ImageClassifier*>(native_handle);
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_tensorflow_lite_task_vision_classifier_ImageClassifier_initJniWithModelPath(
    JNIEnv* env, jclass thiz, jint file_descriptor,
    jlong file_descriptor_length, jlong file_descriptor_offset,
    jobject java_options) {
  ImageClassifierOptions proto_options =
      ConvertToProtoOptions(env, java_options);
  auto file_descriptor_meta = proto_options.mutable_model_file_with_metadata()
                                  ->mutable_file_descriptor_meta();
  file_descriptor_meta->set_fd(file_descriptor);
  file_descriptor_meta->set_length(file_descriptor_length);
  file_descriptor_meta->set_offset(file_descriptor_offset);

  StatusOr<std::unique_ptr<ImageClassifier>> status =
      ImageClassifier::CreateFromOptions(proto_options);
  if (status.ok()) {
    return reinterpret_cast<jlong>(status->release());
  } else {
    ThrowException(env, kAssertionError,
                   "Error occurred when initializing ImageClassifier: ",
                   status.status().message().data());
    return 0;
  }
}

extern "C" JNIEXPORT jobject JNICALL
Java_org_tensorflow_lite_task_vision_classifier_ImageClassifier_classifyNative(
    JNIEnv* env, jclass thiz, jlong native_handle, jobject image_buffer,
    jint width, jint height) {
  auto* classifier = reinterpret_cast<ImageClassifier*>(native_handle);
  auto image = GetMappedFileBuffer(env, image_buffer);
  std::unique_ptr<FrameBuffer> frame_buffer = CreateFromRgbRawBuffer(
      (const uint8*)image.data(), FrameBuffer::Dimension{width, height});
  auto results = classifier->Classify(*frame_buffer);
  if (results.ok()) {
    return ConvertToClassificationResults(env, results.value());
  } else {
    ThrowException(env, kAssertionError,
                   "Error occurred when classifying the image: ",
                   results.status().message().data());
    return 0;
  }
}
}  // namespace
