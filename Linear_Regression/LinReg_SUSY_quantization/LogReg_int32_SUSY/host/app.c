/**
* app.c
* LRGD Host Application Source File, quantized SUSY 
* int32 stable implementation  
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h> 
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

#if ENERGY
#include <dpu_probe.h>
#endif

#define MAXCHAR 500 
#define SCALE (33.1*2/(65536)) 

// Pointer declaration
static T* X;
static T* Y;
static T* W; 


// Read training dataset from Skin_NonSkin.txt 
static int read_input_SUSY(T* X, T* Y, unsigned int m_size, unsigned int n_size) {
    printf("Reading training dataset from csv...\n"); 
    FILE* fp;
    char row[MAXCHAR];
    char* token;
    unsigned int m = 0, n = 0;

    // add file path 
    // fp = fopen("/home/rain/SUSY_int8.csv", "r"); 
    fp = fopen("/home/upmem0013/gyuxin/SUSY_int16.csv", "r"); // add file path 
    if (fp == NULL) {
        perror("Can't open file!");
        return(-1);
    }

    while (fgets(row, MAXCHAR, fp)) {
        token = strtok(row, ",");
        X[m*n_size] = atoi(token); 

        token = strtok(NULL, ",");
        Y[m] = atoi(token); 

        n = 1; 
        token = strtok(NULL, ",");
        while (token != NULL) {
            X[m*n_size + n] = atoi(token); 
            token = strtok(NULL, ","); 
            n++; 
        } 
        m++; 
    }
    fclose(fp);
    printf("Successfully generate input data. m = %d\n", m);
    if (m != m_size) {
        printf("Error: invalid input m_size!\n");
        return -1;
    }
    // for (int i = 0; i < n_size; i++){
    //     printf("%d ", X[i]); 
    // }
    // printf("\n"); 
    return 0; 
}

// Read training dataset from Skin_NonSkin.txt 
static int read_input_SUSY_float(float* X, float* Y, unsigned int m_size, unsigned int n_size) {
    printf("Reading training dataset from csv (float)...\n"); 
    FILE* fp;
    char row[MAXCHAR];
    char* token;
    unsigned int m = 0, n = 0;

    // fp = fopen("/home/rain/SUSY.csv", "r"); 
    fp = fopen("/home/upmem0013/gyuxin/SUSY.csv", "r");  // add file path 
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
    printf("Successfully generate input data. m = %d\n", m);
    if (m != m_size) {
        printf("Error: invalid input m_size!\n");
        return -1;
    }
    // for (int i = 0; i < n_size; i++){
    //     printf("%f ", X[i]); 
    // }
    // printf("\n"); 
    return 0; 
}

static void GD_host_fp(T* X, T* Y, T* W, float* W_float, uint32_t m_size, uint32_t n_size, 
    uint32_t iter_time, float lr) 
{
    printf("-----Start traing of SUSY at host, int32-----\n"); 

    // init wirght with random value
    for (uint32_t n = 0; n < n_size; ++n) {
        W[n] = 1;// << SHIFT_AMOUNT; 
        W_float[n] = 1.0; 
    }

    for (uint32_t i = 0; i < iter_time; ++i) {
        // calculate gradient 
        int64_t* gradient_tmp = (int64_t*) calloc(n_size, sizeof(int64_t)); 

        for (uint32_t j = 0; j < m_size; ++j) {
            int64_t dot_product = 0; 
            for (unsigned int k = 0; k < n_size; ++k) {
                dot_product += X[j*n_size + k] * W[k]; 
            }
	    if (i == 0 && j < 4){
		printf("dot product = %ld\n", dot_product); 
	    }
            for (unsigned int l = 0; l < n_size; ++l) {
                gradient_tmp[l] -= X[j*n_size + l] * ((Y[j]<<SHIFT_AMOUNT) - dot_product) 
                >> OVERFLOW_SHIFT; 
            }
        } // gradient done 
        
        // update weight
        for (uint32_t m = 0; m < n_size; ++m) {
            W[m] = W[m] - (gradient_tmp[m] * lr / ((int) m_size>>OVERFLOW_SHIFT)); 
            W_float[m] = (float) W[m] / (1<<SHIFT_AMOUNT); 
        }
        printf("%d: gradient0 = %ld, W0 = %d, W_float0 = %f\n", i, gradient_tmp[0], W[0], W_float[0]); 

        free(gradient_tmp); 
    } // end iteration
} // end GD_host 

static void compute_mae(const T* X, const float* Y, const float* W, int m_size, int n_size, const char* comment) { 
    float reduction = 0; 
    float sum_of_Y = 0; 
    float error_rate = 0.0; 
    for (int m = 0; m < m_size; ++m) {
        float dot_product = 0.0;
        for (int n = 0; n < n_size; ++n) {
            dot_product += (float) X[m*n_size + n] * W[n]; 
        }
        dot_product *= SCALE; 
        reduction += fabsf(Y[m] - dot_product); 
        error_rate += fabsf(Y[m] - dot_product) / fabsf(Y[m]); 
        sum_of_Y += fabsf(Y[m]) / m_size; 
    }
    float mae = (float) reduction / m_size; 
    printf("True result, MAE on %s = %.4f, avg Y = %.4f, error rate = %.2f%%\n", \
    comment, mae, sum_of_Y, (error_rate/m_size)*100); 
}

static void compute_mae_quant(const T* X, const T* Y, const float* W, 
int m_size, int n_size, const char* comment) { 
    float reduction = 0; 
    float sum_of_Y = 0; 

    for (int m = 0; m < m_size; ++m) {
        float dot_product = 0.0;
        for (int n = 0; n < n_size; ++n) {
            dot_product += (float) X[m*n_size + n] * W[n]; 
        } 
        reduction += fabsf((float) Y[m] - dot_product);
        sum_of_Y += fabsf((float) Y[m]) / m_size; 
    }
    float mae = (float) reduction / m_size; 
    printf("Quantization result, MAE on %s = %.4f, avg Y = %.4f\n", comment, mae, sum_of_Y); 
} 

static void init_argument_tasklet(uint32_t tasklet_id, uint32_t nr_rows, uint32_t* rows_per_tasklet, uint32_t* start_row){
    unsigned int element_per_cacheY = 8 >> DIV; 
    unsigned int chunks = nr_rows / (NR_TASKLETS * element_per_cacheY);
    unsigned int dbl_chunks = chunks * element_per_cacheY;  
    *rows_per_tasklet = dbl_chunks; // rows per tasklet is multiple of element_per_cacheY
    unsigned int rest_rows = nr_rows % (NR_TASKLETS * element_per_cacheY); 

    if ((tasklet_id * element_per_cacheY) < rest_rows)
        *rows_per_tasklet += element_per_cacheY;
    if (rest_rows > 0) {
        if ((tasklet_id * element_per_cacheY) >= rest_rows) {
            if ((rest_rows % element_per_cacheY) != 0)
                *start_row = roundup(rest_rows, element_per_cacheY) + tasklet_id * dbl_chunks; 
            else
                *start_row = rest_rows + tasklet_id * dbl_chunks; 
        } else 
            *start_row = tasklet_id * (dbl_chunks + element_per_cacheY);
    } else {
        *start_row = tasklet_id * (dbl_chunks);
    }

    // printf("tasklet: %d, start_row: %d, row/tasklet: %d\n", tasklet_id, *start_row, *rows_per_tasklet); 
}


// Main of the Host Application
int main(int argc, char **argv) {

    struct Params p = input_params(argc, argv);

    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    // Allocate DPUs and load binary
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set)); 
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("Allocated %d DPU(s)\n", nr_of_dpus);

    unsigned int i = 0;

    unsigned int iter_time = p.iter_time;
    float learning_rate = p.learning_rate; 

    unsigned int m_size = p.m_size;
    unsigned int n_size = p.n_size;

    printf("i = %d, lr = %.4f, m = %d, n = %d\n", iter_time, learning_rate, m_size, n_size); 

    // Initialize help data
    dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t));
    dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
    uint32_t max_rows_per_dpu = 0;
    uint32_t n_size_pad = ((n_size*sizeof(T)) % 8) == 0 ? n_size : roundup(n_size, (8/sizeof(T))); 

    DPU_FOREACH(dpu_set, dpu, i) {
        uint32_t rows_per_dpu;
        uint32_t prev_rows_dpu = 0;
        uint32_t chunks = m_size / nr_of_dpus;
        rows_per_dpu = chunks;
        uint32_t rest_rows = m_size % nr_of_dpus;
        if (i < rest_rows)
            rows_per_dpu++;

        if (rest_rows > 0) {
            if (i >= rest_rows)
                prev_rows_dpu = rest_rows + i * chunks; 
                // prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
            else
                prev_rows_dpu = i * (chunks + 1);
        } else {
            prev_rows_dpu = i * chunks;
        }

        // Keep max rows for parallel transfers
        uint32_t rows_per_dpu_pad = ((rows_per_dpu*sizeof(T)) % 8) == 0 ? rows_per_dpu : roundup(rows_per_dpu, (8/sizeof(T))); 
        if (rows_per_dpu_pad > max_rows_per_dpu)
            max_rows_per_dpu = rows_per_dpu_pad;

        dpu_info[i].rows_per_dpu = rows_per_dpu;
        dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
        dpu_info[i].prev_rows_dpu = prev_rows_dpu;

        // Copy input arguments to DPU
        input_args[i].n_size = n_size;
        input_args[i].n_size_pad = n_size_pad;
        input_args[i].nr_rows = rows_per_dpu;

        // Init arguments for each tasklet
        for(uint32_t id = 0; id < NR_TASKLETS; ++id) {
            init_argument_tasklet(id, rows_per_dpu, &input_args[i].rows_per_tasklet[id], \
                &input_args[i].start_row[id]); 
        }
    }

    // Input/output allocation
    X = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T)); 
    Y = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T)); 
    W = malloc(n_size_pad * sizeof(int32_t)); 

    // init trainging dataset and weight for host 
    T *bufferX = X;
    T *bufferY = Y;
    T *bufferW_host = W; 
    float* bufferW_float = (float*) malloc(n_size_pad * sizeof(float)); 

    read_input_SUSY(bufferX, bufferY, m_size, n_size); 

    // init Weight for DPU 
    float* W_dpu_float  = (float*) malloc(n_size*sizeof(float));
    T* W_dpu_fp = malloc(n_size_pad * sizeof(T)); 
    for (uint32_t n = 0; n < n_size_pad; ++n) {
        W_dpu_fp[n] = 1;// << SHIFT_AMOUNT; 
    }

    // temp dpu gradient  
    int64_t* gradient_dpu_tmp = malloc(n_size_pad * nr_of_dpus * sizeof(int64_t)); 

    // Timer declaration
    Timer timer;

    // Train the model on host
    start(&timer, 0, 0);
    //GD_host_fp(bufferX, bufferY, bufferW_host, bufferW_float, m_size, n_size, iter_time, learning_rate); 
    stop(&timer, 0); 

    // Transfer input arguments and training dataset to DPU
    printf("Load input data to DPUs\n");
    start(&timer, 1, 0); // CPU-DPU transfer time start
    // Input arguments
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        // Copy input arguments to DPU
        input_args[i].max_rows = max_rows_per_dpu;

        DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, \
        sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

    // Copy X  
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, bufferX + dpu_info[i].prev_rows_dpu * n_size));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, \
        max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT)); 
    
    // Copy y 
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, bufferY + dpu_info[i].prev_rows_dpu)); 
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
        max_rows_per_dpu * n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT)); 

    stop(&timer, 1); // CPU-DPU transfer time stop

    // Iteration at DPU
    printf("Run program on DPU(s)...\n"); 
    for(uint32_t rep = 0; rep < iter_time; ++rep) {
        // Copy W 
        start(&timer, 2, rep); // CPU-DPU transfer time start
        i = 0; 
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, W_dpu_fp)); 
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
            max_rows_per_dpu * n_size_pad * sizeof(T) + max_rows_per_dpu * sizeof(T), \
            n_size_pad * sizeof(T), DPU_XFER_DEFAULT)); 
        stop(&timer, 2); // CPU-DPU transfer time stop 

        // Run DPU kernel
        start(&timer, 3, rep); 
        #if ENERGY
        DPU_ASSERT(dpu_probe_start(&probe)); 
        #endif
        
        // Launch kernel 
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS)); 

        stop(&timer, 3);
        #if ENERGY
        DPU_ASSERT(dpu_probe_stop(&probe));
        #endif

#if PRINT
        {
            if (rep%200 == 0) {
                unsigned int each_dpu = 0;
                printf("Display DPU Logs\n");
                DPU_FOREACH (dpu_set, dpu) {
		    if (each_dpu==0){
                    printf("DPU#%d:\n", each_dpu);
                    DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
		    } 
                    each_dpu++;
                }
            }
        }
#endif
        // Retrive result
        start(&timer, 4, rep); // DPU-CPU time 
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, gradient_dpu_tmp + i * n_size_pad)); 
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, n_size_pad * sizeof(int64_t), \
            DPU_XFER_DEFAULT));
        stop(&timer, 4); // DPU-CPU time 

        start(&timer, 5, rep); // CPU reduction 
        int64_t* gradient_dpu = calloc(n_size, sizeof(int64_t)); 
        i = 0; 
        DPU_FOREACH(dpu_set, dpu, i) {
            for (uint32_t x = 0; x < n_size; ++x) {
                gradient_dpu[x] += gradient_dpu_tmp[i*n_size_pad + x];// >> OFFSET; 
            }
        } 
        
        // Update weight 
        for (uint32_t m = 0; m < n_size; ++m) {  
            W_dpu_fp[m] = W_dpu_fp[m] - (gradient_dpu[m]*learning_rate) / (m_size >> OVERFLOW_SHIFT); 
        }
        // printf("iter: %d, gradient_dpu: %d, W_dpu_fp: %d\n", rep, gradient_dpu[0], W_dpu_fp[0]); 
        free(gradient_dpu); 
        stop(&timer, 5); // Update weight at CPU 

        if (rep % 100 == 0)
            printf("DPU iter %d...\n", rep); 
    } // iter end 

    // Print trained weight at host 
    printf("Trained weight at host: ");
    for (uint32_t x = 0; x < n_size; ++x) {
        printf("%.2f, ", bufferW_float[x]); 
    }
    printf("\n"); 

    // Print DPU trained result 
    printf("Trained weight at DPU: ");
    for (uint32_t m = 0; m < n_size; ++m) {
        W_dpu_float[m] = (float) W_dpu_fp[m] / (SHIFT_MASK + 1); 
        printf("%.2f, ", W_dpu_float[m]); 
    }
    printf("\n"); 

    // Print timing results
    printf("CPU "); 
    print(&timer, 0, 1);
    printf("init C-D ");
    print(&timer, 1, 1);
    printf("syn C-D ");
    print(&timer, 2, 1); 
    printf("DPU kernel ");
    print(&timer, 3, 1);
    printf("D-C ");
    print(&timer, 4, 1);
    printf("CPU reduction ");
    print(&timer, 5, 1);

// #if ENERGY
//     double energy;
//     DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
//     printf("DPU Energy (J): %f\t", energy);
// #endif	

    // Check output
    bool status = true; 
    for (uint32_t each_attr = 0; each_attr < n_size; ++each_attr) {
        if ((bufferW_float[each_attr] - W_dpu_float[each_attr] > 0.01) || 
            (bufferW_float[each_attr] - W_dpu_float[each_attr] < -0.01)) 
        {
            status = false; 
        }
    }

    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Compute quantized MAE
    compute_mae_quant(bufferX, bufferY, bufferW_float, m_size, n_size, "host"); 
    compute_mae_quant(bufferX, bufferY, W_dpu_float, m_size, n_size, "DPU"); 

    // Compute training error rate
    float* X_float = malloc(m_size * n_size * sizeof(float)); 
    float* Y_float = malloc(m_size * sizeof(float)); 
    read_input_SUSY_float(X_float, Y_float, m_size, n_size); 
    compute_mae(bufferX, Y_float, W_dpu_float, (int) m_size, (int) n_size, "DPU"); 


    // Deallocation
    free(input_args); 
    free(X);
    free(Y);
    free(W);
    free(bufferW_float); 
    free(W_dpu_fp); 
    free(W_dpu_float); 
    free(gradient_dpu_tmp); 
    DPU_ASSERT(dpu_free(dpu_set));
	
    return status ? 0 : -1;
}
