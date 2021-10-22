cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example)
set(CMAKE_CXX_STANDARD 14)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.5")
set(${CUDNN_INCLUDE_PATH} "/usr/local/cuda-11.5/include")
set(${CUDNN_LIBRARY_PATH} "/usr/local/cuda-11.5/lib64")

list(APPEND CMAKE_PREFIX_PATH "~/miniconda3/envs/ml/lib/python3.8/site-packages/torch")

find_package(Torch REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main ${TORCH_LIBRARIES})