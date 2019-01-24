#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCudaErrors(status) {         \
  if ((status) != 0) {                    \
    printf("\nCUDA error %d\n", status);  \
    exit(status);                         \
  }                                       \
};


#define checkCudnnErrors(status) {            \
  if ((status) != CUDNN_STATUS_SUCCESS) {     \
    printf("\nCuDNN error %d: %s\n",          \
           status,                            \
           cudnnGetErrorString(status));      \
    exit(status);                             \
  }                                           \
};

int main(int argc, char **argv)
{

  int gpuid = 0;
  int nr_gpus;
  cudnnHandle_t cudnnHandle;
  cublasHandle_t cublasHandle;

  
  printf("Starting...\n");
  checkCudaErrors(cudaGetDeviceCount(&nr_gpus));
  printf("Available number of GPUs: %d\n", nr_gpus);
  if (nr_gpus < 1) {
    printf("No GPU available. Terminating.\n");
    exit(-1);
  }

  checkCudaErrors(cudaSetDevice(gpuid));
  checkCudaErrors(cublasCreate(&cublasHandle));
  checkCudnnErrors(cudnnCreate(&cudnnHandle));

  cudnnDestroy(cudnnHandle);
  checkCudaErrors(cublasDestroy(cublasHandle));

  return (0);
}

