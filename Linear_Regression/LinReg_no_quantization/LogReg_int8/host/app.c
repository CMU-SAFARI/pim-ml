/**
* app.c
* LRGD_2.0 Host Application Source File
* int8 (hybrid) stable implementation  
* no builtin
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

// Pointer declaration
static T* X;
static int16_t* Y;
static T* W;

// Create input arrays
static void read_input(T* X, int16_t* Y, T* W, unsigned int m_size, unsigned int n_size) {
    srand(0);

    printf("Predefined weight: ");
    for (unsigned int w = 0; w < n_size; ++w) {
        W[w] = (T) (w+1); 
        // W[w] = (T) (rand()%(n_size*2)); 
        printf("%d, ", (int) W[w]); 
    }

    for (unsigned int i = 0; i < m_size * n_size; ++i) {
        X[i] = (T) (rand()%50); 
    }

    for (unsigned int j = 0; j < m_size; ++j) {
        int16_t tmp = 0;
        for (unsigned int k = 0; k < n_size; ++k) {
            tmp += X[j*n_size + k] * W[k] + (rand()%30); 
        }
        Y[j] = tmp; 
    }
    printf("\nSuccessfully generate input data.\n");
}

static void compute_mae(const T* X, const int16_t* Y, const float* W, int m_size, int n_size, char* comment) { 
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

// Train weight coefficients in the host, fixed-point arithmetic 
static void GD_host_fp(T* X, int16_t* Y, T* W, int16_t* W_fp, uint32_t m_size, uint32_t n_size, uint32_t iter_time, float lr) {
    printf("-----Start traing at host, int-----\n");

    // init weight with random value
    for (uint32_t n = 0; n < n_size; ++n){
        W[n] = (T) 1; 
        W_fp[n] = W[n] << SHIFT_AMOUNT; // fix-pointed arithmetic 
    }

    for (uint32_t i = 0; i < iter_time; ++i) {
        // calculate gradient 
        int32_t* gradient_tmp = calloc(n_size, sizeof(int32_t)); 
        for (uint32_t j = 0; j < m_size; ++j) {
            // dot product 
            int32_t dot_product = 0; 
            for (unsigned int k = 0; k < n_size; ++k) {
                dot_product += X[j*n_size + k] * W_fp[k]; 
            }

            for (unsigned int l = 0; l < n_size; ++l) {
                // avoid overflow
                gradient_tmp[l] -= X[j*n_size + l] * ((Y[j]<<SHIFT_AMOUNT)-dot_product) >> OVERFLOW_SHIFT; 
                // if (j == 0) 
                //     printf("ind gradient: %d, x: %d, y:%d, dot: %d\n", \
                //         X[j*n_size + l]*(Y[j] - dot_product) >> OVERFLOW_SHIFT, \
                //         X[j*n_size + l], Y[j], dot_product); 
            }
            // if(j < 4){
            //     printf("dot_product at host: %d\n", dot_product);
            //     printf("X at host: "); 
            //     for (uint32_t each_attribute = 0; each_attribute < n_size; each_attribute++) {
            //         printf("%d, ", X[j*n_size + each_attribute]); 
            //     }
            //     printf("\n"); 
            // } 
        } // gradient done 
        
        // update weight
        for (uint32_t m = 0; m < n_size; ++m) {
            // W_fp[m] = W_fp[m] - (gradient_tmp[m] * lr / (m_size>>OVERFLOW_SHIFT)) * SHIFT_AMOUNT; 
            // W[m] = W_fp[m] / SHIFT_AMOUNT; 
            W_fp[m] = W_fp[m] - (gradient_tmp[m] * lr) / (m_size>>OVERFLOW_SHIFT); 
            W[m] = W_fp[m] >> SHIFT_AMOUNT; 
        }
        // printf("i: %d, g: %d, g*lr: %.4f, w_fp: %d, w: %d\n", i, gradient_tmp[0], \
        //     (float)gradient_tmp[0]*lr/(m_size>>OVERFLOW_SHIFT)), W_fp[0], W[0]); 
        free(gradient_tmp); 
    } // end iteration
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
            // printf("%d, start row %d, row/tasklet %d\n", input_args[i].start_row[id], \
            //     input_args[i].rows_per_tasklet[id]); 
        }
    }

    // Input/output allocation
    X = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T)); 
    Y = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(int16_t)); 
    W = malloc(n_size_pad * sizeof(T)); 

    // init trainging dataset and weight for host 
    T *bufferX = X;
    int16_t *bufferY = Y;
    T *bufferW_host = W; 
    int16_t* bufferW_fp = malloc(n_size_pad * sizeof(int16_t)); 

    read_input(bufferX, bufferY, bufferW_host, m_size, n_size); // init training dataset and initial host W

    // init Weight for DPU 
    T* W_dpu = malloc(n_size_pad * sizeof(T)); 
    int16_t* W_dpu_fp = malloc(n_size_pad * sizeof(int16_t)); 
    for (uint32_t n = 0; n < n_size_pad; ++n) {
        W_dpu[n] = (T) 1; 
        W_dpu_fp[n] = W_dpu[n] << SHIFT_AMOUNT; 
    }

    // temp dpu gradient  
    int32_t* gradient_dpu_tmp = malloc(n_size_pad * nr_of_dpus * sizeof(int32_t)); 

    // Timer declaration
    Timer timer;

    // Train the model on host
    start(&timer, 0, 0);
    GD_host_fp(bufferX, bufferY, bufferW_host, bufferW_fp, m_size, n_size, iter_time, learning_rate); 
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
        max_rows_per_dpu * n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(int16_t), DPU_XFER_DEFAULT));

    stop(&timer, 1); // CPU-DPU transfer time stop

    // Iteration at DPU
    printf("Run program on DPU(s)...\n"); 
    for(uint32_t rep = 0; rep < iter_time; ++rep) {
        // Copy W 
        start(&timer, 1, rep+1); // CPU-DPU transfer time start
        i = 0; 
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, W_dpu_fp)); 
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
            max_rows_per_dpu * n_size_pad * sizeof(T) + max_rows_per_dpu * sizeof(int16_t), \
            n_size_pad * sizeof(int16_t), DPU_XFER_DEFAULT)); 
        stop(&timer, 1); // CPU-DPU transfer time stop 

        // Run DPU kernel
        start(&timer, 2, rep); 
        #if ENERGY
        DPU_ASSERT(dpu_probe_start(&probe)); 
        #endif
        
        // Launch kernel 
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS)); 

        stop(&timer, 2);
        #if ENERGY
        DPU_ASSERT(dpu_probe_stop(&probe));
        #endif

#if PRINT
        {
            if (rep%200 == 0) {
                unsigned int each_dpu = 0;
                printf("Display DPU Logs\n");
                DPU_FOREACH (dpu_set, dpu) {
                    printf("DPU#%d:\n", each_dpu);
                    DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                    each_dpu++;
                }
            }
        }
#endif
        // Retrive result
        start(&timer, 3, rep); // DPU-CPU time 
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, gradient_dpu_tmp + i * n_size_pad)); 
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, n_size_pad * sizeof(int32_t), \
            DPU_XFER_DEFAULT));
        // stop(&timer, 3); // DPU-CPU time 

        // start(&timer, 4, rep); // Update weight at CPU 
        int32_t* gradient_dpu = calloc(n_size, sizeof(int32_t)); 
        i = 0; 
        DPU_FOREACH(dpu_set, dpu, i) {
            for (uint32_t x = 0; x < n_size; ++x) {
                // gradient_dpu[x] += gradient_dpu_tmp[i*n_size_pad + x] >> OVERFLOW_SHIFT; 
                gradient_dpu[x] += gradient_dpu_tmp[i*n_size_pad + x] >> OFFSET; 
            }
        } 

        // Update weight 
        for (uint32_t m = 0; m < n_size; ++m) {  
            // W_dpu_fp[m] = W_dpu_fp[m] - (gradient_dpu[m]*learning_rate/(m_size>>OVERFLOW_SHIFT))*SHIFT_AMOUNT; 
            // W_dpu[m] = W_dpu_fp[m] / SHIFT_AMOUNT; 
            W_dpu_fp[m] = W_dpu_fp[m] - (gradient_dpu[m]*learning_rate) / (m_size >> OVERFLOW_SHIFT); 
            W_dpu[m] = W_dpu_fp[m] >> SHIFT_AMOUNT; 
        }
        // printf("iter: %d, gradient_dpu: %d, W_dpu_fp: %d\n", rep, gradient_dpu[0], W_dpu_fp[0]); 
        free(gradient_dpu); 

        stop(&timer, 3); // Update weight at CPU 

        if (rep % 100 == 0)
            printf("DPU iter %d...\n", rep); 

    } // iter end 

    // Express fix-pointed coefficients in float
    float* W_host_float = (float*) malloc(n_size*sizeof(float));
    float* W_dpu_float  = (float*) malloc(n_size*sizeof(float));

    // Print trained weight at host 
    printf("Trained weight at host: ");
    for (uint32_t x = 0; x < n_size; ++x) {
        W_host_float[x] = (float) bufferW_fp[x] / (SHIFT_MASK + 1); 
        printf("%.2f, ", W_host_float[x]); 
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
    printf("CPU-DPU ");
    print(&timer, 1, 1);
    printf("DPU Kernel ");
    print(&timer, 2, 1); 
    printf("DPU-CPU ");
    print(&timer, 3, 1);

// #if ENERGY
//     double energy;
//     DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
//     printf("DPU Energy (J): %f\t", energy);
// #endif	

    // Check output
    bool status = true; 
    for (uint32_t each_attr = 0; each_attr < n_size; ++each_attr) {
        // if ((bufferW_host[each_attr] - W_dpu[each_attr] > 1) || 
        //     (bufferW_host[each_attr] - W_dpu[each_attr] < -1)) 
        if (bufferW_host[each_attr] != W_dpu[each_attr]) 
        {
            status = false; 
            // # if PRINT
            // printf("host: %.2f, dpu: %.2f\n", (float) bufferW_host[each_attr], (float) W_dpu[each_attr]); 
            // #endif
        }
    }

    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Compute MSE
    // for(unsigned int n = 0; n < n_size; ++n) 
    //     bufferW_host[n] = 1; 
    compute_mae(bufferX, bufferY, W_host_float, m_size, n_size, "host"); 
    compute_mae(bufferX, bufferY, W_dpu_float, m_size, n_size, "DPU"); 

    // Deallocation
    // free(input_args); 
    free(X);
    free(Y);
    free(W);
    free(bufferW_fp); 
    free(W_dpu); 
    free(W_dpu_fp); 
    free(gradient_dpu_tmp); 
    free(W_host_float);
    free(W_dpu_float); 
    DPU_ASSERT(dpu_free(dpu_set));
	
    return status ? 0 : -1;
}
