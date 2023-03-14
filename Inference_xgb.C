#include <iostream>
#include <chrono>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#if !defined(__CLING__)
#endif

void Inference_xgb(){
    
    Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "xgb"};
    Ort::SessionOptions* sessionOptions = new Ort::SessionOptions();
	sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	sessionOptions->SetIntraOpNumThreads(1);
    sessionOptions->SetInterOpNumThreads(1);
    Ort::Session session(env, "./xgboost_boston.onnx", *sessionOptions);

    const char* input_names[] = {"float_input"};
    const char* output_names[] = {"variable"};
    std::array<float, 13> input{};
    input.fill(0.2);
    std::array<float, 1> results{};
    std::array<int64_t, 2> input_shape{1, 13};
    std::array<int64_t, 2> output_shape{1, 1};

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input.data(), input.size(), input_shape.data(), input_shape.size());
    //timer
    auto begin = std::chrono::high_resolution_clock::now();
    for(int i{}; i < 10000; i++){
        auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken by function: "
         << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / (1000.0*10000.0)
         << " ms" << std::endl;
    // float* intarr = output_tensor.front().GetTensorMutableData<float>();
    // std::vector<float> output_tensor_values {intarr, intarr+1};
    // for(int i{}; i < output_tensor_values.size(); i++){
    //     std::cout << output_tensor_values[i] << std::endl;
    // }

}