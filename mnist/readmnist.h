#ifndef READMNIST_H_
#define READMNIST_H_

struct mnist_img_desc {
    int columns;
    int rows;
    int nr_images;
    unsigned char *databuffer;
    unsigned char *labelbuffer;
    float *databufferf;
    float *labelbufferf;
};

int read_train_mnist(char *trainingdata, char *traininglabels, struct mnist_img_desc *desc);



#endif

