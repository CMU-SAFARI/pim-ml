// Logistic Regression with GD GPU baseline based on cuBLAS 
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h> 
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) 

#define T float

#define MAXCHAR 500

// Read training dataset from Skin_NonSkin.txt 
static int read_input_Skin(T* X, T* Y, T* W, unsigned int m_size, unsigned int n_size) {
    printf("Reading training dataset from Skin_NonSkin.csv...\n");

    FILE* fp;
    char row[MAXCHAR];
    char* token;
    unsigned int m = 0, n = 0;

    fp = fopen("/home/yuxguo/Skin_NonSkin.csv", "r"); // add file path here 
    if (fp == NULL) {
        perror("Can't open file!");
        return(-1);
    }

    while (fgets(row, MAXCHAR, fp)) {
        token = strtok(row, ",");
        n = 0;
        while (n < n_size) {//(token != NULL) {
            X[IDX2C(m, n, m_size)] = atof(token);
            token = strtok(NULL, ",");
            n++;
        }
        char temp = atoi(token);
        if (temp == 1)
            Y[m] = 1.0;
        else
            Y[m] = 0.0;
        m++;
    }
    fclose(fp);
    printf("\nSuccessfully generate input data. m = %d\n", m);
    if (m != m_size) {
        printf("Error: invalid input m_size!\n");
        return -1;
    }
    return 0; 
}

// Read training dataset from SUSY.csv 
static int read_input_SUSY(T* X, T* Y, T* W, unsigned int m_size, unsigned int n_size) {
    printf("Reading training dataset from SUSY...\n");

    FILE* fp;
    char row[MAXCHAR];
    char* token;
    unsigned int m = 0, n = 0;

    fp = fopen("/home/yuxguo/SUSY.csv", "r"); // add file path here 
    if (fp == NULL) {
        perror("Can't open file!");
        return(-1);
    }

    while (fgets(row, MAXCHAR, fp) != NULL) {//m < m_size) {
        //fgets(row, MAXCHAR, fp); 
        token = strtok(row, ",");
        Y[m] = atof(token);

        n = 0;
        token = strtok(NULL, ",");
        while (token != NULL) {
            X[IDX2C(m, n, m_size)] = atof(token);
            token = strtok(NULL, ",");
            n++;
        }
        m++;
    }

    fclose(fp);
    printf("\nSuccessfully generate input data. m = %d\n", m);
    if (m != m_size) {
        printf("Error: invalid input m_size!\n");
        return -1;
    }
    return 0; 
}

// Create synthetic input arrays in host, X is column-major 
static void read_input(T* X, T* Y, T* W, unsigned int m_size, unsigned int n_size) {
    srand(0); 
    printf("Predefined weight: ");
    for (unsigned int w = 0; w < n_size; ++w) {
        W[w] = (T)(w + 1);
        printf("%d, ", (int)W[w]);
    }
    for (unsigned int m = 0; m < m_size; ++m) {
        for (unsigned int n = 0; n < n_size; ++n) {
            X[IDX2C(m, n, m_size)] = (float) ((float)(rand()%100000) - 50000) / 10000; 
        }
    }
    for (unsigned int m = 0; m < m_size; ++m) {
        T dot_product = 0;
        for (unsigned int n = 0; n < n_size; ++n) {
            dot_product += X[IDX2C(m, n, m_size)] * W[n] + (((float)(rand()%400) - 200)/100); 
        }
        double sigmoid_temp = 1.0 / (1.0 + exp((double)(-dot_product)));
        Y[m] = sigmoid_temp >= 0.5 ? 1.0 : 0.0;
    }
    printf("\nSuccessfully generate input data.\n");
}

// error = sigmoid(error) - Y 
__global__ void SigmoidSubY(T error[], const T Y[], unsigned int m_size, unsigned int thread_num) {
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    //for (unsigned int i = thread_id; i < m_size, i += thread_num;) {
    if (thread_id < thread_num) {
        error[thread_id] = (1.0 / (1.0 + expf(-error[thread_id]))) - Y[thread_id];
    }
}

// Train the model at GPU 
static void GD_GPU(cublasHandle_t handle, T* X, T* Y, T* W,
    uint32_t m_size, uint32_t n_size, uint32_t iter_time, float lr) {
    printf("-----Start traing by cuBLAS, float-----\n");

    T* error = (T*)malloc(m_size * sizeof(T));
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // init wirght with random value
    for (uint32_t n = 0; n < n_size; ++n)
        W[n] = (T)1.0;

    T temp = (lr / m_size);
    T neg_temp = -temp;
    T one = 1.0, neg_one = -1.0, zero = 0.0;

    int threadPerBlock = 256;
    int blockNumber = (m_size + threadPerBlock - 1) / threadPerBlock;
    printf("block number: %d, thread per block: %d\n", blockNumber, threadPerBlock);

    // init GPU memory 
    cudaEventRecord(start, 0);
    T* X_dev, * Y_dev, * W_dev, * error_dev, * gradient_dev;

    cudaMalloc((void**)&X_dev, m_size * n_size * sizeof(T));
    cublasSetMatrix(m_size, n_size, sizeof(T), X, m_size, X_dev, m_size);

    cudaMalloc((void**)&Y_dev, m_size * sizeof(T));
    cublasSetVector(m_size, sizeof(T), Y, 1, Y_dev, 1);

    cudaMalloc((void**)&W_dev, n_size * sizeof(T));
    cublasSetVector(n_size, sizeof(T), W, 1, W_dev, 1);

    cudaMalloc((void**)&error_dev, m_size * sizeof(T));
    cudaMalloc((void**)&gradient_dev, n_size * sizeof(T));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // init data transfer time 
    printf("inti data transfer time: %.2f\n", time); 

    // Start epochs
    cudaEventRecord(start, 0);
    for (uint32_t i = 0; i < iter_time; ++i) {
        // error = Xw
        cublasSgemv(handle, CUBLAS_OP_N,
            m_size, n_size, &one, X_dev, m_size, W_dev, 1, &zero, error_dev, 1);

        // error = sigmoid(error) - Y 
        SigmoidSubY<<<blockNumber, threadPerBlock >>> (error_dev, Y_dev, m_size, blockNumber * threadPerBlock);

        // gradient = X_trans * error (*lr/m_size) 
        // cublasSgemv(handle, CUBLAS_OP_T, m_size, n_size, &temp, X_dev, m_size, error_dev, 1, &zero, gradient_dev, 1);
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE); 
        for (uint32_t j = 0; j < n_size; ++j) {
            cublasSdot(handle, m_size, X_dev + j*m_size, 1, error_dev, 1, gradient_dev + j); 
        }
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST); 

        // W -= gradient*lr/m_size  
        cublasSaxpy(handle, n_size, &neg_temp, gradient_dev, 1, W_dev, 1); 

        if (i % 100 == 0)
            printf("iter %d...\n", i); 
    } // end epochs 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // GPU kernel time 
    printf("kernel time: %.2f\n", time); 

    // upload W_dev to host
    cudaEventRecord(start, 0); 
    cublasGetVector(n_size, sizeof(T), W_dev, 1, W, 1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // final transfer time 
    printf("final data transfer time: %.2f\n", time); 

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(X_dev);
    cudaFree(Y_dev);
    cudaFree(W_dev);
    cudaFree(error_dev);
    cudaFree(gradient_dev);
} // end GD_GPU 

void compute_error_rate(const T* X, const T* Y, const T* W, int m_size, int n_size,
    const char* comment) {
    uint32_t reduction = 0;
    uint32_t sum_of_Y = 0;

    for (int m = 0; m < m_size; ++m) {
        float dot_product = 0.0;
        for (int n = 0; n < n_size; ++n) {
            dot_product += (float)X[IDX2C(m, n, m_size)] * W[n];
        }
        double sigmoid_temp = 1 / (1 + exp((double)(-dot_product)));
        int32_t predict_temp = sigmoid_temp >= 0.5 ? 1 : 0;
        if (predict_temp != (int)Y[m]) {
            reduction++;
        }
        sum_of_Y += (int32_t)Y[m];
    }
    printf("error rate on %s = %.2f%%, reduction: %d, sum_of_Y: %d\n", comment, \
        ((float)reduction / m_size) * 100, reduction, sum_of_Y);
}

int main(void) {
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    unsigned int iter_time;
    float        learning_rate;
    unsigned int m_size;
    unsigned int n_size;

    int dataset = 1;

    // printf("Select training dataset...\n");
    // printf("1 for synthetic data, 2 for SUSY, 3 for Skin Segmentation:\n"); 
    // scanf("%d", &dataset); 
    for(dataset = 1; dataset < 4; dataset++)
    {
    if (dataset == 1) { // synthetic 
        printf("Run synthetic dataset, 6291456 x 16\n"); 
        iter_time = 500; 
        learning_rate = 0.1;
        m_size = 6291456;
        n_size = 16; 
    } 
    else if (dataset == 2) { // SUSY 
        printf("Run SUSY dataset, 5000000 x 18\n");
        iter_time = 1000;
        learning_rate = 0.1;
        m_size = 5000000;
        n_size = 18;
    }
    else if (dataset == 3) { // Skin  
        printf("Run Skin Segmentation dataset, 245057 x 3\n");
        iter_time = 500;
        learning_rate = 0.0001;
        m_size = 245057;
        n_size = 3;
    }
    else {
        printf("Please choose valid number!");
        return 1; 
    }
    printf("i = %d, lr = %.4f, m = %d, n = %d\n", iter_time, learning_rate, m_size, n_size);

    int deviceID = 0;
    cudaGetDevice(&deviceID);

    cudaDeviceProp deviceProp;

    cudaStat = cudaGetDeviceProperties(&deviceProp, deviceID);

    if (cudaStat != cudaSuccess) {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", cudaStat,
            __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d, global memory: %d GB\n", deviceID,
        deviceProp.name, deviceProp.major, deviceProp.minor, deviceProp.totalGlobalMem / 1024 / 1024 / 1024);

    // create handle of cuBLAS library 
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    // Pointer declaration
    T* X = (T*)malloc(m_size * n_size * sizeof(T));
    T* Y = (T*)malloc(m_size * sizeof(T));
    T* W = (T*)malloc(n_size * sizeof(T));

    if (dataset == 1) { // synthetic 
        read_input(X, Y, W, m_size, n_size); 
    }
    else if (dataset == 2) { // SUSY 
        read_input_SUSY(X, Y, W, m_size, n_size); 
    }
    else if (dataset == 3) { // Skin  
        read_input_Skin(X, Y, W, m_size, n_size); 
    }

    // start training on GPU 
    GD_GPU(handle, X, Y, W, m_size, n_size, iter_time, learning_rate);

    printf("Trained weight at GPU: ");
    for (uint32_t x = 0; x < n_size; ++x) {
        printf("%.4f, ", W[x]);
    }
    printf("\n");
    compute_error_rate(X, Y, W, m_size, n_size, "GPU");
    // printf("Training time on GPU (ms) = %.2f\n", time);

    free(X);
    free(Y);
    free(W);

    printf("--------------------------\n\n");
    }

    cublasDestroy(handle); // destroy handle 

    return EXIT_SUCCESS;
}