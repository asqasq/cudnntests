cmake_minimum_required (VERSION 2.8)
project (lenet)

find_package(CUDA 9.2 REQUIRED)

# Uncomment the following line to use gflags
#set(USE_GFLAGS 1)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-g;-lineinfo;-Xcompiler;-ggdb)
else()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_52,code=sm_52;-gencode;arch=compute_50,code=compute_50;-std=c++11;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG)
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Addresses a bug where code is not compiled as C++11 in non-CUDA code and older g++ versions
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(USE_GFLAGS)
  add_definitions(-DUSE_GFLAGS)
endif()



include_directories(~/abaglada/nvidia/cuda/include/)
link_directories(~/abaglada/nvidia/cuda/lib64)

include_directories(.)


cuda_add_executable(trainlenet lenet.cu readubyte.cpp readmnist.cpp)
cuda_add_cublas_to_target(trainlenet)

cuda_add_executable(simplemnist simplemnist.cpp readmnist.cpp)
target_link_libraries(simplemnist m)

if(USE_GFLAGS)
  target_link_libraries(trainlenet gflags cudnn)
else()
  target_link_libraries(trainlenet cudnn)
endif()


