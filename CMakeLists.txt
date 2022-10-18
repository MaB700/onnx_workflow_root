project(onnxruntime_test)
cmake_minimum_required(VERSION 3.14.0 FATAL_ERROR)
cmake_policy(VERSION 3.14...3.23)

# Define compiler on macosx to avoid problems when comparing to compilers
# used for FairSoft/FairRoot compilation
if(APPLE)
  if(NOT DEFINED CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER clang)
  endif()
  if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER clang++)
  endif()
endif()

# Default CMake settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT DEFINED ONNXPATH)
  message(FATAL_ERROR "Please pass the path to the onnxruntime installation as -DONNXPATH")
endif()

list(PREPEND CMAKE_PREFIX_PATH ${ONNXPATH})
message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
find_package(ONNXRuntime CONFIG REQUIRED)

get_target_property(bla ONNXRuntime::ONNXRuntime IMPORTED_LOCATION)
message("IMPORTED_LOCATION: ${bla}")

get_target_property(bla ONNXRuntime::ONNXRuntime INTERFACE_INCLUDE_DIRECTORIES)
message("INTERFACE_INCLUDE_DIRECTORIES: ${bla}")

add_executable(Inference_cnn Inference_cnn.cxx)
target_link_libraries(Inference_cnn ONNXRuntime::ONNXRuntime)

add_executable(Inference_xgb Inference_xgb.cxx)
target_link_libraries(Inference_xgb ONNXRuntime::ONNXRuntime)

