cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(example)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")
#add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.5")
#set(${CUDNN_INCLUDE_PATH} "/usr/local/cuda-11.5/include")
#set(${CUDNN_LIBRARY_PATH} "/usr/local/cuda-11.5/lib64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread -Wall")


#list(APPEND CMAKE_PREFIX_PATH "~/miniconda3/envs/ml/lib/python3.8/site-packages/torch")
list(APPEND CMAKE_PREFIX_PATH "/home/sm/Documents/libs/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp)
target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})
#set_property(TARGET example PROPERTY CXX_STANDARD 17)
