#include <iostream>
#if !defined(__CLING__)
#include "onnxruntime_cxx_api.h"
#endif

void Inference_xgb(){
    
    Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "xgb"};
    Ort::Session session(env, "./xgboost_boston.onnx", Ort::SessionOptions(nullptr));

    const char* input_names[] = {"float_input"};
    const char* output_names[] = {"variable"};
    std::array<float, 13> input{};
    input.fill(0.2);
    std::array<float, 1> results{};
    std::array<int64_t, 2> input_shape{1, 13};
    std::array<int64_t, 2> output_shape{1, 1};

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input.data(), input.size(), input_shape.data(), input_shape.size());
    
    auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    float* intarr = output_tensor.front().GetTensorMutableData<float>();
    std::vector<float> output_tensor_values {intarr, intarr+1};
    for(int i{}; i < output_tensor_values.size(); i++){
        std::cout << output_tensor_values[i] << std::endl;
    }

}