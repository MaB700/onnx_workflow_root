#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "input_denoise_single.h"
#include <chrono>

float run_task(int intra, int inter, ExecutionMode mode);

int Inf_gnn(){
  
  std::vector<int> intra{1, 2, 4, 8, 16};//
  std::vector<int> inter{1, 2, 4, 8, 16};//
  std::vector<ExecutionMode> mode{ORT_SEQUENTIAL};
  // loop over all combinations of intra, inter and mode and print the best 5 results
  std::vector<std::tuple<float, int, int, ExecutionMode>> results;
  for(auto i : intra){
    for(auto j : inter){
      for(auto k : mode){
        results.push_back(std::make_tuple(run_task(i, j, k), i, j, k));
      }
    }
  }
  std::sort(results.begin(), results.end());
  for(int i = 0; i < 5; i++){
    std::cout << "Intra: " << std::get<1>(results[i]) << " Inter: " << std::get<2>(results[i]) << " Mode: " << std::get<3>(results[i]) << " Time: " << std::get<0>(results[i]) << std::endl;
  }
  
  
  return 0;

}

float run_task(int intra, int inter, ExecutionMode mode){
  Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "denoise"};
  Ort::SessionOptions session_options;
  //session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options.SetIntraOpNumThreads(intra);
	session_options.SetInterOpNumThreads(inter);
  session_options.SetExecutionMode(mode);
  //session_options.EnableProfiling("denoise");
  Ort::Session session(env, "./gnn_5_16.onnx", session_options);

  std::vector<const char*> input_names = {"nodes", "edge_index"};
  std::vector<const char*> output_names = {"output"};
  
  std::array<float, 50*3> node_input; 
  std::generate(node_input.begin(), node_input.end(), [](){return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);});
  std::array<int64_t, 2*100> edge_input;
  std::generate(edge_input.begin(), edge_input.end(), [](){return rand() % 30;});

  std::array<int64_t, 2> node_shape{50, 3};
  std::array<int64_t, 2> edge_shape{2, 100};
  std::array<int64_t, 2> output_shape{50, 1};

  auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<Ort::Value> inputs;
  inputs.push_back(Ort::Value::CreateTensor<float>(allocator_info, node_input.data(), node_input.size(), node_shape.data(), node_shape.size()));
  inputs.push_back(Ort::Value::CreateTensor<int64_t>(allocator_info, edge_input.data(), edge_input.size(), edge_shape.data(), edge_shape.size()));

  auto begin = std::chrono::high_resolution_clock::now();
  
  for(int i{}; i < 1000; i++){
    //input_image.fill(0.0);
    std::vector<Ort::Value> output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), input_names.size(), output_names.data(), output_names.size());
    //float* intarr = output_tensor.front().GetTensorMutableData<float>();
    //std::vector<float> output_tensor_values {intarr, intarr + 72*32};
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
  std::cout << "Settings: " << "Intra: " << intra << " Inter: " << inter << " Mode: " << mode << std::endl;
  std::cout << "time for each time window: " << elapsed.count()/(1000.0*1000.0) << " ms" << std::endl;

  // for(int i{}; i < output_tensor_values.size(); i++){
  //   if(output_tensor_values[i] > 0.01) std::cout << std::setprecision(18) << output_tensor_values[i] << std::endl;
  // }
  return elapsed.count()/(1000.0*1000.0);
}
