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

#define n 4
#define k 2
#define m 3

// A is a nxk matrix
// B is a kxm matrix
// Result C is nxm matrix


static void print_matrix(float *M, int rows, int columns)
{
  printf("\n");
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      printf("%f ", M[i * columns + j]);
    }
    printf("\n");
  }
}

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

  // crete handles
  checkCudaErrors(cudaSetDevice(gpuid));
  checkCudaErrors(cublasCreate(&cublasHandle));
  checkCudnnErrors(cudnnCreate(&cudnnHandle));

  float *A = (float*)malloc(sizeof(float) * n * k);
  float *B = (float*)malloc(sizeof(float) * k * m);
  float *C = (float*)malloc(sizeof(float) * n * m);

  float v = 1.0f;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k; j++) {
      A[i * k + j] = v;
      v = v + 1.0f;
    }
  }

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < m; j++) {
      B[i * m + j] = v;
      v = v + 1.0f;
    }
  }


  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      float sum = 0.0f;
      for (int e = 0; e < k; e++) {
        sum += A[i * k + e] * B[e * m + j];
      }
      C[i * m + j] = sum;
      //C[0] = sum;
    }
  }



  print_matrix(A, n, k);
  print_matrix(B, k, m);
  print_matrix(C, n, m);





  //destroy handles
  cudnnDestroy(cudnnHandle);
  checkCudaErrors(cublasDestroy(cublasHandle));

  return (0);
}

