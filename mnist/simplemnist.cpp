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


#define checkMatrixOp(status) {         \
  if (!(status)) {                      \
    printf("\nMatrix operation failed: %s %s %d\n", __FILE__, __func__, __LINE__); \
  } \
};


struct matrix {
    float *M;
    int rows;
    int columns;
    bool transposed;
};

static struct matrix* allocate_matrix(int rows, int columns)
{
    struct matrix *tmp = (struct matrix*)malloc(sizeof(struct matrix));
    tmp->M = (float*)malloc(sizeof(float) * rows * columns);
    tmp->rows = rows;
    tmp->columns = columns;
    tmp->transposed = false;
    return tmp;
}

static void free_matrix(struct matrix *M)
{
    free(M->M);
    free(M);
}


static void matrix_transpose(struct matrix *M)
{
    M->transposed = !M->transposed;
}

static inline int matrix_get_array_idx(struct matrix *M, int row, int column)
{
    if (!(M->transposed)) {
        return (row * M->columns + column);
    } else {
        return (column * M->columns + row);
    }
}

static inline int matrix_get_rows(struct matrix *M)
{
    if (!(M->transposed)) {
        return (M->rows);
    } else {
        return (M->columns);
    }
}

static inline int matrix_get_columns(struct matrix *M)
{
    if (!(M->transposed)) {
        return (M->columns);
    } else {
        return (M->rows);
    }
}



static void print_matrix(struct matrix *M)
{
  printf("\n");
  for (int i = 0; i < matrix_get_rows(M); i++) {
    for (int j = 0; j < matrix_get_columns(M); j++) {
      printf("%f ", M->M[matrix_get_array_idx(M, i, j)]);
    }
    printf("\n");
  }
}


// Matrix mulitplication
// C = A * B
static bool matrix_multiplication(struct matrix *A, struct matrix *B, struct matrix *C)
{
  if ((matrix_get_columns(A) != matrix_get_rows(B)) ||
      (matrix_get_rows(A) != matrix_get_rows(C)) ||
      (matrix_get_columns(B) != matrix_get_columns(C))) {
        return false;
  }

  // C = A * B
  for (int i = 0; i < matrix_get_rows(A); i++) {
    for (int j = 0; j < matrix_get_columns(B); j++) {
      float sum = 0.0f;
      for (int e = 0; e < matrix_get_columns(A); e++) {
        sum += A->M[matrix_get_array_idx(A, i, e)] * B->M[matrix_get_array_idx(B, e, j)];
      }
      C->M[matrix_get_array_idx(C, i, j)] = sum;
    }
  }
  return true;
}

static bool matrix_add(struct matrix *A, struct matrix *B, struct matrix *C)
{
    if ((matrix_get_rows(A) != matrix_get_rows(B)) ||
        (matrix_get_rows(A) != matrix_get_rows(C)) ||
        (matrix_get_columns(A) != matrix_get_columns(B)) ||
        (matrix_get_columns(A) != matrix_get_columns(C))) {
          return false;
    }

    // C = A + B
    for (int i = 0; i < (A->rows * A->columns); i++) {
        C->M[i] = A->M[i] + B->M[i];
    }
    return true;
}

static bool matrix_scaling(float v, struct matrix *A, struct matrix *C)
{
    if ((matrix_get_rows(A) != matrix_get_rows(C)) ||
        (matrix_get_columns(A) != matrix_get_columns(C))) {
          return false;
    }

    for (int i = 0; i < (A->rows * A->columns); i++) {
        C->M[i] = v * A->M[i];
    }
    return true;
}

static bool matrix_sigma(struct matrix *A, struct matrix *C)
{
    if ((matrix_get_rows(A) != matrix_get_rows(C)) ||
        (matrix_get_columns(A) != matrix_get_columns(C))) {
          return false;
    }

    for (int i = 0; i < (A->rows * A->columns); i++) {
        C->M[i] = (1.0f / (1.0f + exp(-A->M[i])));
    }
    return true;
}

static void matrix_random_init(struct matrix *A)
{
    int n = A->rows * A->columns;

    for (int i = 0; i < n; i++) {
        A->M[i] = (float)random()/((float)RAND_MAX);
    }
}

static void init_random_generator(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    srandom((int)tv.tv_usec);
}



static void forward_propagation(struct matrix *input,
                                struct matrix *weight1, struct matrix *bias1,
                                struct matrix *fc1out, struct matrix *fc1biasout,
                                struct matrix *fc1activationout,
                                struct matrix *weight2, struct matrix *bias2,
                                struct matrix *fc2out, struct matrix *fc2biasout,
                                struct matrix *fc2activationout)
{
    checkMatrixOp(matrix_multiplication(input,
                                        weight1,
                                        fc1out));

    checkMatrixOp(matrix_add(fc1out, bias1, fc1biasout));

    checkMatrixOp(matrix_sigma(fc1biasout, fc1activationout));

    checkMatrixOp(matrix_multiplication(fc1activationout,
                                        weight2,
                                        fc2out));

    checkMatrixOp(matrix_add(fc2out, bias2, fc2biasout));

    checkMatrixOp(matrix_sigma(fc2biasout, fc2activationout));
}

/*
forward:

[w5 w6] [outh1] = [w5*out_h1 + w6*out_h2] = [net_o1]
[w7 w8] [outh2] = [w7*out_h1 + w8*out_h2] = [net_o2]


backwards:
[v1] * [out_h1 out_h2] = [w5, w6]
[v2]                     [w7, w8]

*/

static void backward_propagation(struct matrix *target /*y=label*/,
                                struct matrix *weight1, struct matrix *bias1,
                                struct matrix *fc1out, struct matrix *fc1biasout,
                                struct matrix *fc1activationout,
                                struct matrix *weight2, struct matrix *bias2,
                                struct matrix *fc2out, struct matrix *fc2biasout,
                                struct matrix *fc2activationout,
                                struct matrix *fc2v, struct matrix *dweight2)
{
    fc2v->M[0] = -(target->M[0] - fc2activationout->M[0]) * fc2activationout->M[0]*(1 - fc2activationout->M[0]);
    fc2v->M[1] = -(target->M[1] - fc2activationout->M[1]) * fc2activationout->M[1]*(1 - fc2activationout->M[1]);
    fc2v->M[2] = -(target->M[2] - fc2activationout->M[2]) * fc2activationout->M[2]*(1 - fc2activationout->M[2]);
    fc2v->M[3] = -(target->M[3] - fc2activationout->M[3]) * fc2activationout->M[3]*(1 - fc2activationout->M[3]);
    fc2v->M[4] = -(target->M[4] - fc2activationout->M[4]) * fc2activationout->M[4]*(1 - fc2activationout->M[4]);
    fc2v->M[5] = -(target->M[5] - fc2activationout->M[5]) * fc2activationout->M[5]*(1 - fc2activationout->M[5]);
    fc2v->M[6] = -(target->M[6] - fc2activationout->M[6]) * fc2activationout->M[6]*(1 - fc2activationout->M[6]);
    fc2v->M[7] = -(target->M[7] - fc2activationout->M[7]) * fc2activationout->M[7]*(1 - fc2activationout->M[7]);
    fc2v->M[8] = -(target->M[8] - fc2activationout->M[8]) * fc2activationout->M[8]*(1 - fc2activationout->M[8]);
    fc2v->M[9] = -(target->M[9] - fc2activationout->M[9]) * fc2activationout->M[9]*(1 - fc2activationout->M[9]);

    checkMatrixOp(matrix_multiplication(fc2v,
                                        fc1activationout,
                                        dweight2));
}



static int predict(struct matrix *image,
                   struct matrix *weight1, struct matrix *bias1,
                   struct matrix *fc1out, struct matrix *fc1biasout,
                   struct matrix *fc1activationout,
                   struct matrix *weight2, struct matrix *bias2,
                   struct matrix *fc2out, struct matrix *fc2biasout,
                   struct matrix *fc2activationout)
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
        printf("output activation %d: %f\n", i, fc2activationout->M[i]);
        if (fc2activationout->M[i] > current_value) {
            pred = i;
            current_value = fc2activationout->M[i];
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

    struct matrix* weights1 = allocate_matrix(784, 50);
    struct matrix* bias1 = allocate_matrix(50, 1);
    struct matrix* weights2 = allocate_matrix(50, 10);
    struct matrix* bias2 = allocate_matrix(10, 1);

    struct matrix* fc1out = allocate_matrix(50, 1);
    struct matrix* fc1biasout = allocate_matrix(50, 1);
    struct matrix* fc1activationout = allocate_matrix(50, 1);

    struct matrix* fc2out = allocate_matrix(10, 1);
    struct matrix* fc2biasout = allocate_matrix(10, 1);
    struct matrix* fc2activationout = allocate_matrix(10, 1);

    int prediction = -1;



    init_random_generator();

    matrix_random_init(weights1);
    matrix_random_init(bias1);

    matrix_random_init(weights2);
    matrix_random_init(bias2);

    struct matrix input_image;
    input_image.rows = 28;
    input_image.columns = 28;
    
    input_image.M = &(traindesc.databufferf[0]);
    forward_propagation(&input_image,
                        weights1, bias1,
                        fc1out, fc1biasout,
                        fc1activationout,
                        weights2, bias2,
                        fc2out, fc2biasout,
                        fc2activationout);


    input_image.M = &(testdesc.databufferf[0]);
    prediction = predict(&input_image,
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
    struct matrix* A = allocate_matrix(rows, cols);
    struct matrix* B = allocate_matrix(rows, cols);
    struct matrix* C = allocate_matrix(rows, cols);
    struct matrix* D = allocate_matrix(rows, rows);
    struct matrix* E = allocate_matrix(cols, cols);

    for (int i = 0; i < A->rows * A->columns; i++) {
        A->M[i] = 1.0f;
    }

    for (int i = 0; i < B->rows * B->columns; i++) {
        B->M[i] = (float)i;
    }
    
    printf("\nA:\n");
    print_matrix(A);
    printf("B:\n");
    print_matrix(B);

    printf("signma(A):\n");
    checkMatrixOp(matrix_sigma(A, C));
    print_matrix(C);

    printf("A+B:\n");
    checkMatrixOp(matrix_add(A, B, C));
    print_matrix(C);


    printf("A * B:\n");
    print_matrix(A);
    matrix_transpose(B);
    print_matrix(B);
    //D_5x5 = A_5x3 * B'_3x5
    checkMatrixOp(matrix_multiplication(A, B, D));
    print_matrix(D);
    //E_3x3 = B'_3x5 * A_5x3
    checkMatrixOp(matrix_multiplication(B, A, E));
    print_matrix(E);
    matrix_transpose(B);

    checkMatrixOp(matrix_scaling(0.711f, A, C));
    print_matrix(C);

    print_matrix(B);
    matrix_transpose(B);
    print_matrix(B);
    matrix_transpose(B);
    print_matrix(B);
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

