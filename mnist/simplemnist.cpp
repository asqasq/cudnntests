#include <cudnn.h>
#include <stdio.h>
#include <readmnist.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

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

static void matrix_add(float *A, int rowA, int colA, float *B, int rowB, int colB, float *C)
{
    // C = A + B
    for (int i = 0; i < (rowA * colA); i++) {
        C[i] = A[i] + B[i];
    }
}

static void matrix_scaling(float v, float *A, int rowA, int colA, float *C)
{
    for (int i = 0; i < (rowA * colA); i++) {
        C[i] = v * A[i];
    }
}

static void matrix_sigma(float *A, int rowA, int colA, float *C)
{
    for (int i = 0; i < (rowA * colA); i++) {
        C[i] = (1.0f / (1.0f + exp(-A[i])));
    }
}

static void matrix_random_init(float *A, int rows, int cols)
{
    int n = rows * cols;

    for (int i = 0; i < n; i++) {
        A[i] = (float)random()/((float)RAND_MAX);
    }
}

static void init_random_generator(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    srandom((int)tv.tv_usec);
}



static void forward_propagation(float *input,
                                float *weight1, float *bias1,
                                float *fc1out, float *fc1biasout,
                                float *fc1activationout,
                                float *weight2, float *bias2,
                                float *fc2out, float *fc2biasout,
                                float *fc2activationout)
{
    matrix_multiplication(input, 1, 784,
                          weight1, 784, 50,
                          fc1out);

    matrix_add(fc1out, 1, 50, bias1, 1, 50, fc1biasout);

    matrix_sigma(fc1biasout, 1, 50, fc1activationout);
    
    matrix_multiplication(fc1activationout, 1, 50,
                          weight2, 50, 10,
                          fc2out);

    matrix_add(fc2out, 1, 10, bias2, 1, 10, fc2biasout);

    matrix_sigma(fc2biasout, 1, 10, fc2activationout);
}


static void backward_propagation()
{
}



static int predict(float *image,
                   float *weight1, float *bias1,
                   float *fc1out, float *fc1biasout,
                   float *fc1activationout,
                   float *weight2, float *bias2,
                   float *fc2out, float *fc2biasout,
                   float *fc2activationout)
{
    float current_value = 0.0f;
    int pred = 0;

    forward_propagation(image,
                        weight1, bias1,
                        fc1out, fc1biasout,
                        fc1activationout,
                        weight2, bias2,
                        fc2out, fc2biasout,
                        fc2activationout);

    for (int i = 0; i < 10; i++) {
        printf("output activation %d: %f\n", i, fc2activationout[i]);
        if (fc2activationout[i] > current_value) {
            pred = i;
            current_value = fc2activationout[i];
        }
    }
    return (pred);
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
    float fc1activationout[50];
    
    float fc2out[10];
    float fc2biasout[10];
    float fc2activationout[10];

    int prediction = -1;



    init_random_generator();

    matrix_random_init(weights1, 784, 50);
    matrix_random_init(bias1, 1, 50);

    matrix_random_init(weights2, 50, 10);
    matrix_random_init(bias2, 1, 10);


    
    
    forward_propagation(traindesc.databufferf,
                        weights1, bias1,
                        fc1out, fc1biasout,
                        fc1activationout,
                        weights2, bias2,
                        fc2out, fc2biasout,
                        fc2activationout);


    prediction = predict(&(testdesc.databufferf[0]),
                         weights1, bias1,
                         fc1out, fc1biasout,
                         fc1activationout,
                         weights2, bias2,
                         fc2out, fc2biasout,
                         fc2activationout);
    
    printf("\nPredicted number: %d\n", prediction);
    return 0;
}

static int tests()
{
    const int rows = 5;
    const int cols = 3;

    float *A = (float*)malloc(sizeof(float) * rows * cols);
    float *B = (float*)malloc(sizeof(float) * rows * cols);
    float *C = (float*)malloc(sizeof(float) * rows * cols);

    for (int i = 0; i < rows * cols; i++) {
        A[i] = 1.0f;
    }

    for (int i = 0; i < rows * cols; i++) {
        B[i] = (float)i;
    }
    print_matrix(A, rows, cols);
    print_matrix(B, rows, cols);

    matrix_sigma(A, rows, cols, C);
    print_matrix(C, rows, cols);

    matrix_add(A, rows, cols, B, rows, cols, C);
    print_matrix(C, rows, cols);


    print_matrix(A, rows, cols);
    print_matrix(B, cols, rows);
    //C_3x3 = B'_3x5 * A_5x3
    matrix_multiplication(B, cols, rows, A, rows, cols, C);
    print_matrix(C, cols, cols);

    matrix_scaling(0.711f, A, rows, cols, C);
    print_matrix(C, rows, cols);
}

int main(int argc, char **argv)
{
    if (argc == 5) {
        create_simple_network(argv[1], argv[2], argv[3], argv[4]);
    } else {
        tests();
    }
    return 0;
}

