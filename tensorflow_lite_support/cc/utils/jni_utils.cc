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

#include "tensorflow_lite_support/cc/utils/jni_utils.h"

namespace tflite {
namespace support {
namespace utils {

std::string JStringToString(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) {
    return std::string();
  }
  const char* cstring = env->GetStringUTFChars(jstr, nullptr);
  std::string result(cstring);
  env->ReleaseStringUTFChars(jstr, cstring);
  return result;
}

std::vector<std::string> StringListToVector(JNIEnv* env, jobject list_object) {
  jobject j_iterator = env->CallObjectMethod(
      list_object, env->GetMethodID(env->GetObjectClass(list_object),
                                    "iterator", "()Ljava/util/Iterator;"));
  std::vector<std::string> result;
  jmethodID has_next =
      env->GetMethodID(env->GetObjectClass(j_iterator), "hasNext", "()Z");
  jmethodID get_next = env->GetMethodID(env->GetObjectClass(j_iterator), "next",
                                        "()Ljava/lang/Object;");
  while (env->CallBooleanMethod(j_iterator, has_next)) {
    jstring jstr =
        static_cast<jstring>(env->CallObjectMethod(j_iterator, get_next));
    const char* raw_str = env->GetStringUTFChars(jstr, JNI_FALSE);
    result.emplace_back(std::string(raw_str));
    env->ReleaseStringUTFChars(jstr, raw_str);
  }
  return result;
}

absl::string_view GetMappedFileBuffer(JNIEnv* env, const jobject& file_buffer) {
  return absl::string_view(
      static_cast<char*>(env->GetDirectBufferAddress(file_buffer)),
      static_cast<size_t>(env->GetDirectBufferCapacity(file_buffer)));
}

void ThrowException(JNIEnv* env, const char* clazz, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  const size_t max_msg_len = 512;
  auto* message = static_cast<char*>(malloc(max_msg_len));
  if (message && (vsnprintf(message, max_msg_len, fmt, args) >= 0)) {
    env->ThrowNew(env->FindClass(clazz), message);
  } else {
    env->ThrowNew(env->FindClass(clazz), "");
  }
  if (message) {
    free(message);
  }
  va_end(args);
}

}  // namespace utils
}  // namespace support
}  // namespace tflite
