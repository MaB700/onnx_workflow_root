#include <iostream>
#if !defined(__CLING__)
#include "onnxruntime_cxx_api.h"
#endif

void Inference(){
    
    Ort::Env env{OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "mnist_cnn"};
    Ort::Session session(env, "./mnist/mnist_cnn.onnx", Ort::SessionOptions(nullptr));

    const char* input_names[] = {"input_1"};
    const char* output_names[] = {"dense_1"};
    std::array<float, 28*28> input_image{};
    std::array<float, 10> results{};
    std::array<int64_t, 4> input_shape{1, 28, 28, 1};
    std::array<int64_t, 2> output_shape{1, 10};

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_image.data(), input_image.size(), input_shape.data(), input_shape.size());
    //Ort::Value output_tensor = Ort::Value::CreateTensor<float>(allocator_info, results.data(), results.size(), output_shape.data(), output_shape.size());
    
    auto output_tensor = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    float* intarr = output_tensor.front().GetTensorMutableData<float>();
    std::vector<float> output_tensor_values {intarr, intarr + 10};
    for(int i{}; i < output_tensor_values.size(); i++){
        std::cout << output_tensor_values[i] << std::endl;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    // {
    //     Ort::AllocatorWithDefaultOptions allocator;

    //     // print number of model input nodes
    //     size_t num_input_nodes = session.GetInputCount();
    //     std::vector<const char*> input_node_names(num_input_nodes);
    //     std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
    //                                     // Otherwise need vector<vector<>>
        
    //     //printf("Number of inputs = %zu\n", num_input_nodes);

    //     // iterate over all input nodes
    //     for (int i = 0; i < num_input_nodes; i++) {
    //     // print input node names
    //     char* input_name = session.GetInputName(i, allocator);
    //     printf("Input %d : name=%s\n", i, input_name);
    //     input_node_names[i] = input_name;

    //     /* // print input node types
    //     Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    //     auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    //     ONNXTensorElementDataType type = tensor_info.GetElementType();
    //     //printf("Input %d : type=%d\n", i, type);

    //     // print input shapes/dims
    //     input_node_dims = tensor_info.GetShape(); */
    //     //printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    //     //for (int j = 0; j < input_node_dims.size(); j++)
    //     //  printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
    //     }
    // }

    // {
    //     Ort::AllocatorWithDefaultOptions allocator;

    //     // print number of model input nodes
    //     size_t num_output_nodes = session.GetOutputCount();
    //     std::vector<const char*> output_node_names(num_output_nodes);
    //     std::vector<int64_t> output_node_dims; 

    //     //printf("Number of outputs = %zu\n", num_output_nodes);

    //     // iterate over all input nodes
    //     for (int i = 0; i < num_output_nodes; i++) {
    //     // print input node names
    //     char* output_name = session.GetOutputName(i, allocator);
    //     printf("Input %d : name=%s\n", i, output_name);
    //     output_node_names[i] = output_name;

    //     /* // print input node types
    //     Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    //     auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    //     ONNXTensorElementDataType type = tensor_info.GetElementType();
    //     //printf("Output %d : type=%d\n", i, type);

    //     // print input shapes/dims
    //     output_node_dims = tensor_info.GetShape(); */
    //     //printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
    //     //for (int j = 0; j < output_node_dims.size(); j++)
    //     //  printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    //     }
    // }

}