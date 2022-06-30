#include <iostream>
#if !defined(__CLING__)
#include "onnxruntime_cxx_api.h"
#endif

void Inference_cnn(){
    
    Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "mnist_cnn"};
    Ort::Session session(env, "./mnist_cnn.onnx", Ort::SessionOptions(nullptr));

    const char* input_names[] = {"input_1"};
    const char* output_names[] = {"dense_1"};
    std::array<float, 28*28> input_image{};
    input_image.fill(0.2);
    std::array<float, 10> results{};
    std::array<int64_t, 4> input_shape{1, 28, 28, 1};
    std::array<int64_t, 2> output_shape{1, 10};

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());
    
    auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    float* intarr = output_tensor.front().GetTensorMutableData<float>();
    std::vector<float> output_tensor_values {intarr, intarr + 10};
    for(int i{}; i < output_tensor_values.size(); i++){
        std::cout << output_tensor_values[i] << std::endl;
    }

}