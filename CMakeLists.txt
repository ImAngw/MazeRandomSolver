cmake_minimum_required(VERSION 3.23)
project(CudaMy CUDA)

find_package(CUDAToolkit)

set(CMAKE_CUDA_STANDARD 14)

add_executable(CudaMy main.cu maze_func.cu maze_func.cuh)

set_target_properties(CudaMy PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
