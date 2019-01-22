cmake_minimum_required (VERSION 2.8)
project (edgedetector)

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

find_package( OpenCV REQUIRED )
#find_package( HDF5 10.3.1 REQUIRED )

#string(REPLACE ":" ";" LIBRARY_DIRS $ENV{LD_LIBRARY_PATH})
#find_library (hdf5_10_3_1_LIBRARY libhdf5.so.10.3.1 PATHS ${LIBRARY_DIRS})


cuda_add_executable(edgedetector edgedetector.cu)
cuda_add_cublas_to_target(edgedetector)

include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(edgedetector ${OpenCV_LIBS} )



if(USE_GFLAGS)
    target_link_libraries(edgedetector gflags cudnn opencv_imgproc opencv_core hdf5 tiff)
else()
  target_link_libraries(edgedetector cudnn opencv_imgproc opencv_core hdf5 tiff)
endif()

message("VERSION isch ${OpenCV_VERSION_MAJOR}")

if(${OpenCV_VERSION_MAJOR} EQUAL "3")
  message("Adding opencv_imgproc for OpenCV version 3")
  target_link_libraries(edgedetector opencv_imgproc)
endif()

