#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "input_denoise_single.h"
#include <chrono>


int Inf(){
  
  Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "denoise"};
  Ort::SessionOptions session_options;
  //session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  //session_options.EnableProfiling("denoise");
  Ort::Session session(env, "./mixed.onnx", session_options);

  const char* input_names[] = {"input_1"};
  const char* output_names[] = {"conv2d_6"};
  std::array<float, 72*32> input_image = input_denoise::single_input;
  //input_image.fill(0.2);
  std::array<int64_t, 4> input_shape{1, 72, 32, 1};
  std::array<int64_t, 4> output_shape{1, 72, 32, 1};

  auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());
  
  auto begin = std::chrono::high_resolution_clock::now();
  
  
  // input_image = input_denoise::single_input;
  // auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
  // float* intarr = output_tensor.front().GetTensorMutableData<float>();
  // std::vector<float> output_tensor_values {intarr, intarr + 72*32};

  input_image = input_denoise::single_input;
  for(int i{}; i < 1000; i++){
    //input_image.fill(0.0);
    auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    //float* intarr = output_tensor.front().GetTensorMutableData<float>();
    //std::vector<float> output_tensor_values {intarr, intarr + 72*32};
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);

  std::cout << "time for each time window: " << elapsed.count()/(1000.0*1000.0) << " ms" << std::endl;

  // for(int i{}; i < output_tensor_values.size(); i++){
  //   if(output_tensor_values[i] > 0.01) std::cout << std::setprecision(18) << output_tensor_values[i] << std::endl;
  // }
  
  return 0;

}
