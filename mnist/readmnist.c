#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <readmnist.h>

#define BUFFERSIZE (4 * 1024*1024*1024UL)
int read_train_mnist(char *trainingdata, char *traininglabels, struct mnist_img_desc *desc)
{
    unsigned char *buffer = (unsigned char*)malloc(BUFFERSIZE);
    if (buffer == NULL) {
        return -1;
    }
    
/******************************************************************************/    
/* training images                                                            */
/******************************************************************************/    
    FILE *f = fopen(trainingdata, "r");
    fread(buffer, BUFFERSIZE, 1, f);
    fclose(f);
    
    int idx = 0;
    
    unsigned int header = 0;
    for (int i = 0; i < 4; i++) {
        header = header << 8;
        header |= buffer[idx++];
    }
    unsigned int nr_images = 0;
    for (int i = 0; i < 4; i++) {
        nr_images = nr_images << 8;
        nr_images |= buffer[idx++];
    }
    
    unsigned int nr_rows = 0;
    for (int i = 0; i < 4; i++) {
        nr_rows = nr_rows << 8;
        nr_rows |= buffer[idx++];
    }
    unsigned int nr_cols = 0;
    for (int i = 0; i < 4; i++) {
        nr_cols = nr_cols << 8;
        nr_cols |= buffer[idx++];
    }
    
    printf("\nheader = 0x%08x, nr images = %u, nr rows = %u, nr columns = %u\n",
        header, nr_images, nr_rows, nr_cols);
    
    unsigned int dataoffset = idx;
    unsigned int datasize = nr_rows * nr_cols * nr_images;
    printf("\ndata size = %u\n", datasize);
    
    
    desc->databuffer = (unsigned char*)malloc(datasize);
    if (desc->databuffer == NULL) {
        free(buffer);
        return -2;
    }
    memcpy(desc->databuffer, buffer + dataoffset, datasize);
    desc->columns = nr_cols;
    desc->rows = nr_rows;
    desc->nr_images = nr_images;



/******************************************************************************/    
/* training labels                                                            */
/******************************************************************************/    
    f = fopen(traininglabels, "r");
    fread(buffer, BUFFERSIZE, 1, f);
    fclose(f);
    
    idx = 0;
    
    header = 0;
    for (int i = 0; i < 4; i++) {
        header = header << 8;
        header |= buffer[idx++];
    }
    unsigned int nr_labels = 0;
    for (int i = 0; i < 4; i++) {
        nr_labels = nr_labels << 8;
        nr_labels |= buffer[idx++];
    }
    
    printf("\nheader = 0x%08x, nr labels = %u\n",
        header, nr_labels);
    
    dataoffset = idx;
    datasize = 1 * nr_labels; // Labels are 1 bytes in size
    printf("\ndata size = %u\n", datasize);

    desc->labelbuffer = (unsigned char*)malloc(datasize);
    if (desc->labelbuffer == NULL) {
        free(desc->databuffer);
        free(buffer);
        return -3;
    }
    memcpy(desc->labelbuffer, buffer + dataoffset, datasize);

    free(buffer);
    return 0;
}



void free_mnist_desc(struct mnist_img_desc *desc)
{
    if (desc->databuffer) {
        free(desc->databuffer);
    }
    if (desc->labelbuffer) {
        free(desc->labelbuffer);
    }
}

void dump_image(struct mnist_img_desc *desc, int idx)
{
    int imgoffset = idx * desc->columns * desc->rows;
    printf("\n");
    for (int i = 0; i < desc->rows; i++) {
        for (int j = 0; j < desc->columns; j++) {
            printf("%02x ", desc->databuffer[imgoffset + i * desc->columns + j]);
        }
        printf("\n");
    }
    printf("\n*****\n  %d  \n*****\n", desc->labelbuffer[idx]);
}



void test_read_mnist()
{
     struct mnist_img_desc traindesc;
     struct mnist_img_desc testdesc;
     
     //train
     int res = read_train_mnist("~/data/mnist/train-images-idx3-ubyte",
                                "~/data/mnist/train-labels-idx1-ubyte",
                                &traindesc);
     if (res != 0) {
        printf("Error %d\n", res);
     } else {
        printf("ok\n");
        printf("Nr images = %d, nr columns + %d, nr rows = %d, image size = %d\n",
            traindesc.nr_images, traindesc.columns, traindesc.rows, 
            traindesc.columns * traindesc.rows);
        for (int i = 0; i < 10; i++) {
            dump_image(&traindesc, i);
        }
     }
     
     //test
     res = read_train_mnist("~/data/mnist/t10k-images-idx3-ubyte",
                                "~/data/mnist/t10k-labels-idx1-ubyte",
                                &testdesc);
     if (res != 0) {
        printf("Error %d\n", res);
     } else {
        printf("ok\n");
        printf("Nr images = %d, nr columns + %d, nr rows = %d, image size = %d\n",
            testdesc.nr_images, testdesc.columns, testdesc.rows, 
            testdesc.columns * testdesc.rows);
        for (int i = 0; i < 10; i++) {
            dump_image(&testdesc, i);
        }
     }
}


