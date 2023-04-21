/*
 * LogReg_2.4
 * Compute gradient of MSE loss function with multiple tasklet 
 * Symmetric 16-bit look-up-table for sogmoid, load to MRAM and transfered to WRAM at the beginning 
 */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;  

// if the boundary is -+20, size of LUT = 20*1024*2 = 40KB for int16 
__mram_noinit int16_t SIGMOID_LUT[LUT_SIZE]; 
__dma_aligned int16_t LUT_buffer[LUT_SIZE]; 

__mram_noinit T DPU_RESULTS[MAX_ROWS]; // partial gradient in each DPU, max number of rows = 16 

__dma_aligned T gradient_tmp[MAX_ROWS*NR_TASKLETS]; 

char lut_flag = 0; // 0: LUT not loaded into WRAM, 1: LUT loaded into WRAM 

// Dot product 
static T dot_product(T *bufferX, T *bufferW, uint32_t length) {
    T result = 0; 
    for (unsigned int i = 0; i < length; i++) {
        result += bufferX[i] * bufferW[i]; 
    }
    return result; 
}

# ifdef FLOAT 
// Sigmoid function by Taylor serious 
static float sigmoid_dpu(T x){
    if(x >= 15)
        return (T) 1; 
    else if (x <= -15) 
        return (T) 0; 
    else if (x == 0.0)
        return 0.5; 

    float sum = 1.0;
    float temp = 1.0; 
    // iter 100 times 
    for(uint32_t i = 1; i < 101; ++i){
        temp = temp * (-x) / i;
        sum = sum + temp; 
    }
    // printf("exp: %f\n", sum);
    return (1.0 / (1.0 + sum)); 
} 

# else // int 
static T sigmoid_LUT(T x){
    // if input is too large/small/zero 
    if(x > (SIGMOID_BOUNDARY<<SHIFT_AMOUNT)) 
        return (1<<SHIFT_AMOUNT);
    else if (x < (-(SIGMOID_BOUNDARY<<SHIFT_AMOUNT)))
        return (T) 0; 
    else if (x == 0)
        return (1<<(SHIFT_AMOUNT-1)); 

    // Query LUT 
    uint32_t lut_index = (x < 0) ? ((-x) - 1) : (x - 1); 
    int16_t ans = LUT_buffer[lut_index]; 
    ans = (x < 0) ? ((1<<SHIFT_AMOUNT) - ans) : (ans); 

    return (T) ans; 
}
# endif 

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// main
int main() {
    unsigned int tasklet_id = me();
    if (tasklet_id == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap
        // printf("lut flag: %d\n", lut_flag); 
    }
    
    barrier_wait(&my_barrier); // Barrier

    uint32_t n_size = DPU_INPUT_ARGUMENTS.n_size;
    uint32_t n_size_pad = DPU_INPUT_ARGUMENTS.n_size_pad;
    uint32_t nr_rows = DPU_INPUT_ARGUMENTS.nr_rows;
    uint32_t max_rows = DPU_INPUT_ARGUMENTS.max_rows;

    // arguments for each tasklet 
    uint32_t rows_per_tasklet = DPU_INPUT_ARGUMENTS.rows_per_tasklet[tasklet_id]; 
    uint32_t start_row = DPU_INPUT_ARGUMENTS.start_row[tasklet_id];

    // Load LUT from MRAM to WRAM 
    if (lut_flag == 0) {
        // Transfer 2048 bytes each time 
        uint32_t required_iter = (LUT_SIZE << 1) >> 11; // lut_size_in_byte / 2048 = 20 
        for(uint32_t i = tasklet_id; i < required_iter; i += NR_TASKLETS){
            mram_read((__mram_ptr void const*) (SIGMOID_LUT + (i << 10)), LUT_buffer + (i << 10), 2048); 
        } 
        barrier_wait(&my_barrier); // Barrier

        if (tasklet_id == 0){
            lut_flag = 1; // set flag 
        }
    }

    // Clear global arrays in WRAM 
    uint32_t tasklet_offset = tasklet_id * n_size_pad; 
    for (uint32_t each_attribute = 0; each_attribute < n_size_pad; each_attribute++) {
        gradient_tmp[tasklet_offset + each_attribute] = 0; 
    } 

    // Address of the current row in MRAM
    uint32_t n_size_byte = n_size << MUL;//* sizeof(T);
    uint32_t n_size_pad_byte = n_size_pad << MUL;//* sizeof(T); 

    uint32_t mram_base_addr_X = (uint32_t) (DPU_MRAM_HEAP_POINTER + start_row * n_size_byte);
    uint32_t mram_base_addr_Y = (uint32_t) (DPU_MRAM_HEAP_POINTER + max_rows * n_size_pad_byte + (start_row << MUL)); 
    uint32_t mram_base_addr_W = (uint32_t) (DPU_MRAM_HEAP_POINTER + max_rows * n_size_pad_byte + (max_rows << MUL)); 
    
    uint32_t mram_temp_addr_X = mram_base_addr_X;
    uint32_t mram_temp_addr_Y = mram_base_addr_Y;

    // Inititalize a local cache to store the MRAM block 
    T *cache_X = (T *) mem_alloc(BLOCK_SIZE); 
    T *cache_Y = (T *) mem_alloc(BLOCK_SIZE); 
    T *cache_W = (T *) mem_alloc(n_size_pad_byte); 

    // read W from MRAM 
    mram_read((__mram_ptr void const*) (mram_base_addr_W), cache_W, n_size_pad_byte); 

    // Iterate over nr_rows
    uint32_t rows_per_cache = BLOCK_SIZE / n_size_byte; 
    # if PRINT 
    if(tasklet_id == NR_TASKLETS-1) {
        printf("tasklet: %d, n_size: %d, n_size_pad: %d, row/tasklet: %d, nr_rows: %d, max_rows: %d\n", \
            tasklet_id, n_size, n_size_pad, rows_per_tasklet, nr_rows, max_rows); 
        printf("cache size: %d, rows_per_cache: %d\n", BLOCK_SIZE, rows_per_cache); 
    }
    # endif
    for (unsigned int row_index = 0; row_index < rows_per_tasklet;) {
        mram_temp_addr_X = mram_base_addr_X + row_index * n_size_byte; 
        mram_temp_addr_Y = mram_base_addr_Y + (row_index << MUL); 

        // read X and Y from MRAM 
        mram_read((__mram_ptr void const*) (mram_temp_addr_X), cache_X, BLOCK_SIZE); 
        mram_read((__mram_ptr void const*) (mram_temp_addr_Y), cache_Y, BLOCK_SIZE); 

        // Iterate over cache 
        uint32_t x_index = 0; 
        for(unsigned int y_index = 0; (y_index<rows_per_cache) && (row_index<rows_per_tasklet); y_index++, \
            row_index++){ 
            if(row_index+start_row >= nr_rows){
                row_index = rows_per_tasklet; 
                break; 
            }

            // compute dot product
            T dot_product_t = dot_product(cache_X + x_index, cache_W, n_size); 
            # ifdef FLOAT
            T sigmoid = sigmoid_dpu(dot_product_t); 
            # else 
            // sigmoid is in the range of [0, 2048]
            // T sigmoid = sigmoid_dpu_fp((float) dot_product_t / (SHIFT_MASK+1)); 
            T sigmoid = sigmoid_LUT(dot_product_t); 
            # endif

            // compute gradient 
            // TODO: unroll the loop 
            for (unsigned int l = 0; l < n_size; ++l) {
                #ifdef FLOAT 
                gradient_tmp[tasklet_offset + l] += cache_X[x_index + l] * (sigmoid - cache_Y[y_index]); 

                #else // int, fixed-pointed  
                gradient_tmp[tasklet_offset + l] += cache_X[x_index + l] * (sigmoid - \
                    (cache_Y[y_index]<<SHIFT_AMOUNT)) >> SHIFT_AMOUNT; 
                #endif
            }
            # if PRINT
            if(row_index < 100) {
                printf("dot_product dpu: %d, sigmoid_dpu: %d, %d\n", dot_product_t, sigmoid, \
                    (-dot_product_t) + (SIGMOID_BOUNDARY<<SHIFT_AMOUNT)); 
                // printf("X at DPU: "); 
                // for (uint32_t each_attribute = 0; each_attribute < n_size_pad; each_attribute++) {
                //     printf("%d, ", cache_X[x_index+each_attribute]); 
                // }
                // printf("\n");
            }
            # endif
            x_index += n_size; 
        } // end cache_X 
    } // accessed all rows 

    // Barrier
    barrier_wait(&my_barrier);

    // Reduction 
    // if (tasklet_id > 0) {
    //     for (unsigned int each_attribute = 0; each_attribute < n_size; each_attribute++) {
    //         gradient_tmp[each_attribute] += gradient_tmp[tasklet_id*n_size_pad + each_attribute]; 
    //     }
    // }
    // barrier_wait(&my_barrier);
    // if (tasklet_id == 0) {
    //     // partial result of gradient in this DPU
    //     mram_write((const void *) gradient_tmp, (__mram_ptr void *) DPU_RESULTS, n_size_pad_byte); 
    // }
    if (tasklet_id == 0) {
        for (unsigned int each_tasklet = 1; each_tasklet < NR_TASKLETS; each_tasklet++){
            for (unsigned int each_attribute = 0; each_attribute < n_size; each_attribute++) {
                gradient_tmp[each_attribute] += gradient_tmp[each_tasklet*n_size_pad + each_attribute]; 
            }
        }
        # if PRINT
        if(tasklet_id == NR_TASKLETS-1)
            printf("gradient_tmp at dpu: %d\n", gradient_tmp[0]); 
        # endif
        
        // partial result of gradient in this DPU
        mram_write((const void *) gradient_tmp, (__mram_ptr void *) DPU_RESULTS, n_size_pad_byte); 
    }

    return 0;
}
