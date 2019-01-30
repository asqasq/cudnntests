#include <cudnn.h>
#include <stdio.h>
#include <readmnist.h>

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

static void matrix_scaling(float v, float *A, int rowA, int colA)
{
    for (int i = 0; i < (rowA * colA); i++) {
        A[i] = v * A[i];
    }
}


static void forward_propagation(float *input,
                                float *weight1, float *bias1,
                                float *fc1out, float *fc1biasout,
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

    float weights1[784 * 50];
    float bias1[50];
    float weights2[50 * 10];
    float bias2[10];
    
    float fc1out[50];
    float fc1biasout[50];
    
    float fc2out[10];
    float fc2biasout[10];
    
    
    
    forward_propagation(traindesc.databufferf,
                        weights1, bias1,
                        fc1out, fc1biasout,
                        weights2, bias2,
                        fc2out, fc2biasout);
    
    return 0;
}

#if 0
void bla() {
    int gpu_id = 0;

    cudaSetDevice(gpu_id);

    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    checkCudnnErrors(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCudnnErrors(cudnnSetTensor4dDescriptor(input_descriptor,
                                          /*format=*/CUDNN_TENSOR_NHWC,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/128,
                                          /*channels=*/1,
                                          /*image_height=*/traindesc->rows,
                                          /*image_width=*/traindesc->cols));
}


#include <stdio.h>
#include <math.h>

int main () {
   double x = 0;
  
   printf("The exponential value of %lf is %lf\n", x, exp(x));
   printf("The exponential value of %lf is %lf\n", x+1, exp(x+1));
   printf("The exponential value of %lf is %lf\n", x+2, exp(x+2));
   
   return(0);
}

#endif

