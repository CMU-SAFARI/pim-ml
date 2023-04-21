/* C source code is found in dgemm_example.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mkl.h"

#include "params.h" 

#define T float 

#define MAXCHAR 500

// Create input arrays
static void read_input(T* X, T* Y, T* W, unsigned int m_size, unsigned int n_size) {
    srand(0);

    printf("Predefined weight: ");
    for (unsigned int w = 0; w < n_size; ++w) {
        W[w] = (T) (w+1); 
        printf("%d, ", (int) W[w]); 
    }

    for (unsigned int i = 0; i < m_size * n_size; ++i) {
        // X[i] = (rand()%50) + ((float) (rand()%100))/100; 
        X[i] = ((float) (rand()%10000)) / 10000; 
    }

    for (unsigned int j = 0; j < m_size; ++j) {
        float tmp = 0; 
        for (unsigned int k = 0; k < n_size; ++k) {
            tmp += X[j*n_size + k] * W[k] + ((float) (rand()%300))/1000; 
        }
        Y[j] = tmp; 
    }
    printf("\nSuccessfully generate input data (float).\n");
}

static int read_input_SUSY(T* X, T* Y, T* W, unsigned int m_size, unsigned int n_size) {
    printf("Reading training dataset from file...\n"); 

    FILE* fp;
    char row[MAXCHAR];
    char* token;
    unsigned int m = 0, n = 0; 

    fp = fopen("/home/rain/桌面/ETH-SAFARI/SRC/LogReg/Baseline/SUSY.csv", "r"); 
    if (fp == NULL) {
        perror("Can't open file!");
        return(-1);
    } 

    while (fgets(row, MAXCHAR, fp)) {
        token = strtok(row, ",");
        X[m*n_size] = atof(token); 

        token = strtok(NULL, ",");
        Y[m] = atof(token); 

        n = 1; 
        token = strtok(NULL, ",");
        while (token != NULL) {
            X[m*n_size + n] = atof(token); 
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

static void GD_host(T* X, T* Y, T* W, uint32_t m_size, uint32_t n_size, uint32_t iter_time, float lr) {
    printf("-----Start traing at host, float-----\n");

    // init wirght with random value
    for (uint32_t n = 0; n < n_size; ++n)
        W[n] = (T) 1; 

    for (uint32_t i = 0; i < iter_time; ++i) {
        // calculate gradient 
        T* gradient_tmp = (T*) calloc(n_size, sizeof(T)); 

        for (uint32_t j = 0; j < m_size; ++j) {
            T dot_product = 0; 
            for (unsigned int k = 0; k < n_size; ++k) {
                dot_product += X[j*n_size + k] * W[k]; 
            }

            for (unsigned int l = 0; l < n_size; ++l) {
                gradient_tmp[l] -= X[j*n_size + l] * (Y[j] - dot_product) / ((int) m_size); // avoid overflow 
            }
        } // gradient done 
        
        // update weight
        for (uint32_t m = 0; m < n_size; ++m) {
            W[m] = W[m] - (gradient_tmp[m] * lr); 
        }

        free(gradient_tmp); 
    } // end iteration
} // end GD_host 

static void GD_MKL(T* X, T* Y, T* W, uint32_t m_size, uint32_t n_size, uint32_t iter_time, float lr) {
    printf("-----Start traing by MKL, float-----\n");
    double s_initial, s_elapsed;

    // init wirght with random value
    for (uint32_t n = 0; n < n_size; ++n)
        W[n] = (T) 1.0; 

    T temp = (lr / m_size); 

    T* error = (T*) mkl_malloc(m_size * sizeof(T), 64); 
    
    // Start epochs
    for (uint32_t i = 0; i < iter_time; ++i) {
        // error = Y 
        cblas_scopy(m_size, Y, 1, error, 1); 
        
        // error = (Xw - error) * lr / m_size  
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 
                    m_size, n_size, temp, X, n_size, W, 1, -temp, error, 1); 

        T* gradient_tmp = (T*) malloc(n_size*sizeof(T)); 

        // gradient = X_trans * error
        // s_initial = dsecnd();
        cblas_sgemv(CblasRowMajor, CblasTrans, 
                    m_size, n_size, 1.0, X, n_size, error, 1, 0.0, gradient_tmp, 1); 
        // s_elapsed = (dsecnd() - s_initial); 
        // printf("time (ms) of gemv: %.4f\n", s_elapsed*1000); 

        // W -= gradient 
        for (int n = 0; n < n_size; ++n){
            W[n] = W[n] - gradient_tmp[n]; 
        }

        free(gradient_tmp); 
    } // end epochs 
    mkl_free(error); 
} // end GD_MKL 

static void compute_mae(const T* X, const T* Y, const T* W, int m_size, int n_size, const char* comment) { 
    float reduction = 0; 
    float sum_of_Y = 0; 
    for (int m = 0; m < m_size; ++m) {
        float dot_product = 0.0;
        for (int n = 0; n < n_size; ++n) {
            dot_product += (float) X[m*n_size + n] * W[n]; 
        }
        reduction += (float) (fabsf(Y[m] - dot_product)) / m_size; 
        sum_of_Y += (float) abs(Y[m]) / m_size; 
    }
    // float mae = (float) reduction / m_size; 
    printf("MAE on %s = %.4f, avg Y = %.4f, error rate = %.2f%%\n", comment, reduction, sum_of_Y, \
        (reduction/sum_of_Y)*100); 
}

int main(int argc, char **argv)
{
    // set training data size and hyperparameters 
    struct Params p = input_params(argc, argv);

    unsigned int iter_time = p.iter_time;
    float        learning_rate = p.learning_rate; 
    unsigned int m_size = p.m_size;
    unsigned int n_size = p.n_size;
    unsigned int thread_num = p.thread_num; 

    if (thread_num != 0){// && thread_num <= mkl_get_max_threads()){
        mkl_set_num_threads(thread_num); 
        mkl_set_dynamic(0); 
    } 

    printf("t = %d (dynamic:%d), i = %d, lr = %.4f, m = %d, n = %d\n\n", 
        mkl_get_max_threads(), mkl_get_dynamic(), iter_time, learning_rate, m_size, n_size); 

    // printf("max thread number: %d\n", mkl_get_max_threads());

    // Pointer declaration
    T* X = (T*) mkl_malloc(m_size*n_size*sizeof(T), 64); 
    T* Y = (T*) mkl_malloc(m_size*sizeof(T), 64);
    T* W = (T*) mkl_malloc(n_size*sizeof(T), 64); 

    double s_initial, s_elapsed; 

    if (X == NULL || Y == NULL || W == NULL) {
      printf( "\n ERROR: Can't allocate memory for training dataset and W. Aborting... \n\n");
      mkl_free(X);
      mkl_free(Y);
      mkl_free(W); 
      return 1;
    }

    // init training dataset and initial host W
    read_input(X, Y, W, m_size, n_size); 

    s_initial = dsecnd();
    GD_host(X, Y, W, m_size, n_size, iter_time, learning_rate); 
    s_elapsed = (dsecnd() - s_initial); 

    const char* comment = "CPU_no_mkl"; 
    compute_mae(X, Y, W, (int) m_size, (int) n_size, comment); 

    printf("Trained weight at host: ");
    for (uint32_t x = 0; x < n_size; ++x) {
        printf("%.4f, ", (float) W[x]); 
    }

    printf("\ntraining time without MKL (ms) = %.4f\n", s_elapsed * 1000); 
    printf("\n"); 

    s_initial = dsecnd();
    GD_MKL(X, Y, W, m_size, n_size, iter_time, learning_rate); 
    s_elapsed = (dsecnd() - s_initial); 

    comment = "CPU_mkl"; 
    compute_mae(X, Y, W, (int) m_size, (int) n_size, comment); 

    printf("Trained weight at host: ");
    for (uint32_t x = 0; x < n_size; ++x) {
        printf("%.4f, ", (float) W[x]); 
    }
    printf("\n"); 

    printf("training time with MKL (ms) = %.4f\n", s_elapsed * 1000); 

    mkl_free(X);
    mkl_free(Y);
    mkl_free(W); 

    return 0;
}