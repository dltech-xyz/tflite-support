# TensorFlow Lite Support

TFLite Support is a toolkit that helps users to develop ML and deploy TFLite
models onto mobile devices. It works cross-Platform and is supported on Java,
C++ (WIP), and Swift (WIP). The TFLite Support project consists of the following
major components:

*   **TFLite Support Util Library**: a cross-platform library that helps to
    deploy TFLite models onto mobile devices.
*   **TFLite Model Metadata**: (metadata populator and metadata extractor
    library): includes both human and machine readable information about what a
    model does and how to use the model.
*   **TFLite Support Codegen Tool**: an executable that generates model wrapper
    automatically based on the Support Library and the metadata.
*   **TFLite Support Task Library**: a flexible and ready-to-use library for
    common machine learning model types, such as classification and detection.

TFLite Support library serves different tiers of deployment requirements from
easy onboarding to fully customizable. There are three major use cases that
TFLite Support targets at:

*   **Provide ready-to-use APIs for users to interact with the model**. \
    This is achieved by the TFLite Support Codegen tool, where users can get the
    model interface (contains ready-to-use APIs) simply by passing the model to
    the codegen tool. The automatic codegen strategy is designed based on the
    TFLite metadata.

*   **Provide optimized model interface for popular ML tasks**. \
    The model interfaces provided by the TFLite Support Task Library are
    specifically optimized compared to the codegen version in terms of both
    usability and performance. Users can also swap their own custom models with
    the default models in each task.

*   **Provide the flexibility to customize model interface and build inference
    pipelines**. \
    The TFLite Support Util Library contains varieties of util methods and data
    structures to perform pre/post processing and data conversion. It is also
    designed to match the behavior of TensorFlow modules, such as TF.Image and
    TF.text, ensuring consistency from training to inferencing.