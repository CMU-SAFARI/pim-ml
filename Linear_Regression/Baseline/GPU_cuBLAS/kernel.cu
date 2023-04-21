// Linear Regression with GD GPU baseline based on cuBLAS 
//-----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> 
#include <math.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i)) 
#define T float   
#define MAXCHAR 500

// Read training dataset from SUSY.csv, X is column-major 
static int read_input_SUSY(T* X, T* Y, T* W, unsigned int m_size, unsigned int n_size) {
    printf("Reading training dataset from SUSY.csv...\n"); 

    FILE* fp;
    char row[MAXCHAR];
    char* token;
    unsigned int m = 0, n = 0;

    fp = fopen("/home/yuxguo/SUSY.csv", "r"); // add file path here 
    if (fp == NULL) {
        perror("Can't open file!");
        return(-1);
    }

    while (fgets(row, MAXCHAR, fp)) {
        token = strtok(row, ",");
        X[IDX2C(m, 0, m_size)] = atof(token); 

        token = strtok(NULL, ",");
        Y[m] = atof(token);

        n = 1; 
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

// Create input arrays in host, X is column-major 
static void read_input(T* X, T* Y, T* W, unsigned int m_size, unsigned int n_size) {
    srand(0);
    printf("Predefined weight: ");
    for (unsigned int w = 0; w < n_size; ++w) {
        W[w] = (T)(w + 1);
        printf("%d, ", (int)W[w]); 
    }
    for (unsigned int m = 0; m < m_size; ++m) {
        for (unsigned int n = 0; n < n_size; ++n) {
            X[IDX2C(m, n, m_size)] = ((float) (rand()%10000)) / 10000; 
        }
    }
    for (unsigned int m = 0; m < m_size; ++m) {
        T tmp = 0;
        for (unsigned int n = 0; n < n_size; ++n) {
            tmp += X[IDX2C(m, n, m_size)] * W[n] + ((float) (rand()%300)) / 1000; 
        }
        Y[m] = tmp;
    }
    printf("\nSuccessfully generate input data.\n");
}

// Train the model at GPU 
static void GD_GPU(cublasHandle_t handle, T* X, T* Y, T* W, 
                   uint32_t m_size, uint32_t n_size, uint32_t iter_time, float lr) {
    printf("-----Start traing by cuBLAS, float-----\n"); 

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

    // init GPU memory 
    cudaEventRecord(start, 0);
    T *X_dev, *Y_dev, *W_dev, *error_dev, *gradient_dev;

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
        // error = Y 
        cublasScopy(handle, m_size, Y_dev, 1, error_dev, 1); 

        // error = (Xw - error) * lr / m_size  
        cublasSgemv(handle, CUBLAS_OP_N, m_size, n_size, &temp, X_dev, m_size, W_dev, 1, &neg_temp, error_dev, 1); 

        // gradient = X_trans * error
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE); 
        for (uint32_t j = 0; j < n_size; ++j) {
            cublasSdot(handle, m_size, X_dev + j*m_size, 1, error_dev, 1, gradient_dev + j); 
        }
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST); 

        // W -= gradient 
        cublasSaxpy(handle, n_size, &(neg_one), gradient_dev, 1, W_dev, 1); 

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
    cudaFree(W_dev);
    cudaFree(error_dev); 
    cudaFree(gradient_dev); 
} // end GD_GPU 

static void compute_mae(const T* X, const T* Y, const T* W, int m_size, int n_size, const char* comment) {
    float reduction = 0;
    float sum_of_Y = 0;
    for (int m = 0; m < m_size; ++m) {
        float dot_product = 0.0;
        for (int n = 0; n < n_size; ++n) {
            dot_product += X[IDX2C(m,n,m_size)] * W[n]; 
        }
        reduction += (float)(fabsf(Y[m] - dot_product)) / m_size;
        sum_of_Y += Y[m] / m_size; 
    }
    // float mae = (float) reduction / m_size; 
    printf("MAE on %s = %.4f, avg Y = %.4f, error rate = %.2f%%\n", comment, reduction, sum_of_Y, \
        (reduction / sum_of_Y) * 100);
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
    // printf("1 for synthetic data, 2 for SUSY:\n"); 
    // scanf("%d", &dataset); 
    for(dataset = 1; dataset < 3; dataset++)
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
    else {
        printf("Please choose valid number!\n");
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

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", deviceID,
        deviceProp.name, deviceProp.major, deviceProp.minor);

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

    GD_GPU(handle, X, Y, W, m_size, n_size, iter_time, learning_rate); 

    printf("Trained weight at GPU: ");
    for (uint32_t x = 0; x < n_size; ++x) {
        printf("%.2f, ", W[x]);
    }
    printf("\n");

    compute_mae(X, Y, W, m_size, n_size, "GPU");

    // printf("Training time on GPU (ms) = %.2f\n", time); 

    free(X); 
    free(Y); 
    free(W); 

    printf("--------------------------\n\n"); 
    } 

    cublasDestroy(handle); // destroy handle 

    return EXIT_SUCCESS;
}