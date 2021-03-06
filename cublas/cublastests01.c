#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// Matrix mulitplication
// C = A * B
static void matrix_multiplication(float *A, int rowA, int colA, float *B, int rowB, int colB, float *C)
{
  // C = A * B
  for (int i = 0; i < rowA; i++) {
    for (int j = 0; j < colB; j++) {
      float sum = 0.0f;
      for (int e = 0; e < colA; e++) {
        sum += A[i * colA + e] * B[e * colB + j];
      }
      C[i * colB + j] = sum;
    }
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
  float *C2 = (float*)malloc(sizeof(float) * n * m);
  memset(C2, 0, sizeof(float) * n * m);

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


  // C = A * B
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

  matrix_multiplication(A, n, k, B, k, m, C);

  print_matrix(A, n, k);
  print_matrix(B, k, m);
  print_matrix(C, n, m);


  // allocate A, B and C on the GPU
  float *d_A, *d_B, *d_C;
  cudaMalloc((void**)&d_A, sizeof(float) * n * k);
  cudaMalloc((void**)&d_B, sizeof(float) * k * m);
  cudaMalloc((void**)&d_C, sizeof(float) * n * m);

  checkCudaErrors(cublasSetMatrix(n, k, sizeof(float), A, n, d_A, n));
  checkCudaErrors(cublasSetMatrix(k, m, sizeof(float), B, k, d_B, k));

  float alpha = 1.0f;
  float beta = 0.0f;

  //cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, m, &alpha,
  //    d_A, n, d_B, k, &beta, d_C, n);
  cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
      d_B, m, d_A, k, &beta, d_C, m);


  cublasGetMatrix(n, m, sizeof(float), d_C, n, C, n);

  printf("\nCUDA result:\n");
  print_matrix(C, n, m);


  // allocate A, B and C on the GPU
  float *d_A2, *d_B2, *d_C2;
  cudaMalloc((void**)&d_A2, sizeof(float) * n * k);
  cudaMalloc((void**)&d_B2, sizeof(float) * k * m);
  cudaMalloc((void**)&d_C2, sizeof(float) * n * m);

//  checkCudaErrors(cudaMemcpyAsync(d_A2, A, sizeof(float) * n * k, cudaMemcpyHostToDevice));
//  checkCudaErrors(cudaMemcpyAsync(d_B2, B, sizeof(float) * k * m, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A2, A, sizeof(float) * n * k, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_B2, B, sizeof(float) * k * m, cudaMemcpyHostToDevice));

  //cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, k, m, &alpha,
  //    d_A, n, d_B, k, &beta, d_C, n);
  cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
      d_B2, m, d_A2, k, &beta, d_C2, m);

//  checkCudaErrors(cudaMemcpyAsync(C2, d_C2, sizeof(float) * n * m, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(C2, d_C2, sizeof(float) * n * m, cudaMemcpyDeviceToHost));
  
  printf("\nCUDA result memcpy:\n");
  print_matrix(C2, n, m);

  // free memory
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_B));
  checkCudaErrors(cudaFree(d_C));
  checkCudaErrors(cudaFree(d_A2));
  checkCudaErrors(cudaFree(d_B2));
  checkCudaErrors(cudaFree(d_C2));
  free(A);
  free(B);
  free(C);
  free(C2);

  //destroy handles
  cudnnDestroy(cudnnHandle);
  checkCudaErrors(cublasDestroy(cublasHandle));

  return (0);
}

