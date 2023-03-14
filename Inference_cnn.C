#include <iostream>
#include <chrono>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#if !defined(__CLING__)
#endif

void Inference_cnn(){
    
    Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "mnist_cnn"};
    Ort::SessionOptions* sessionOptions = new Ort::SessionOptions();
	sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	sessionOptions->SetIntraOpNumThreads(1);
    sessionOptions->SetInterOpNumThreads(1);
    Ort::Session session(env, "./mnist_cnn.onnx", *sessionOptions);

    const char* input_names[] = {"input_1"};
    const char* output_names[] = {"dense_1"};
    std::array<float, 28*28> input_image{};
    input_image.fill(0.2);
    std::array<float, 10> results{};
    std::array<int64_t, 4> input_shape{1, 28, 28, 1};
    std::array<int64_t, 2> output_shape{1, 10};

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());
    auto begin = std::chrono::high_resolution_clock::now();
    for(int i{}; i < 10000; i++){
        auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken by function: "
         << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / (1000.0*10000.0)
         << " ms" << std::endl;
    // auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    // float* intarr = output_tensor.front().GetTensorMutableData<float>();
    // std::vector<float> output_tensor_values {intarr, intarr + 10};
    // for(int i{}; i < output_tensor_values.size(); i++){
    //     std::cout << output_tensor_values[i] << std::endl;
    // }

}