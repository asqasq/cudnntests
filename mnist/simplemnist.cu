#include <cudnn.h>
#include <stdio.h>
#include <readmnist.h>


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

static void matrix_add(float *A, int rowA, int colA, float *B, int rowB, int colB, float *C)
{
    // C = A + B
    for (int i = 0; i < (rowA * colA); i++) {
        C[i] = A[i] + B[i];
    }
}


static void forward_propagation(float *input,
                                float *weight1, float *bias1,
                                float *fc1out, float *fc1biasout
                                float *weight2, float *bias2,
                                float *fc2out, float *fc2biasout)
{
    matrix_multiplication(input, 1, 784,
                          weight1, 784, 50,
                          fc1out);

    matrix_add(fc1out, 1, 50, bias1, 1, 50, fc1biasout);
    
    matrix_multiplication(fc1biasout, 1, 50,
                          weight1, 50, 10,
                          fc2out);

    matrix_add(fc2out, 1, 10, bias2, 1, 10, fc2biasout);
}


int create_simple_network(char *trainimg, char *trainlb, char *tstimg, char *tstlb)
{
    int res;
    int gpu_id = 0;

    // load data    
    struct mnist_img_desc traindesc;
    struct mnist_img_desc testdesc;

    //train
    res = read_train_mnist(trainimg, trainlb, &traindesc);
    if (res != 0) {
       printf("Error %d\n", res);
       return res;
    }

    //test
    res = read_train_mnist(tstimg, tstlb, &testdesc);
    if (res != 0) {
       printf("Error %d\n", res);
       return res;
    }

// 784, 50, 10

/*
1x784 * 784x50 = 1x50
1x50 + 1x50 = 1x50

1x50 * 50x10 = 1x10
1x10 * 1x10 = 1x10
*/

    float weights1[784, 50];
    float bias1[50];
    float weights2[50, 10];
    float bias2[10];
    
    float fc1out[50];
    float fc1biasout[50];
    
    float fc2out[10];
    float fc2biasout[10];
    
}


void bla() {
    cudaSetDevice(gpu_id);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/128,
                                          /*channels=*/1,
                                          /*image_height=*/traindesc->rows,
                                          /*image_width=*/traindesc->cols));
}

