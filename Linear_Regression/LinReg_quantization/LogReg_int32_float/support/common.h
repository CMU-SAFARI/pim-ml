#ifndef _COMMON_H_
#define _COMMON_H_

// Structures used by both the host and the dpu to communicate information 
typedef struct {
    uint32_t n_size;
    uint32_t n_size_pad;
    uint32_t nr_rows;
    uint32_t max_rows;
    uint32_t start_row[NR_TASKLETS]; 
    uint32_t rows_per_tasklet[NR_TASKLETS]; 
} dpu_arguments_t;

// Specific information for each DPU
struct dpu_info_t {
    uint32_t rows_per_dpu;
    uint32_t rows_per_dpu_pad;
    uint32_t prev_rows_dpu;
};
struct dpu_info_t *dpu_info;

// Transfer size between MRAM and WRAM
#ifdef BL
#define BLOCK_SIZE_LOG2 BL
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#else
#define BLOCK_SIZE_LOG2 8
#define BLOCK_SIZE (1 << BLOCK_SIZE_LOG2)
#define BL BLOCK_SIZE_LOG2
#endif

// Data type
#ifdef INT32
#define T int32_t
#define MUL 2 // Shift left to divide by sizeof(T)
#define DIV 2 // Shift right to divide by sizeof(T)
#elif FLOAT
#define T float
#define MUL 2 // Shift left to divide by sizeof(T)
#define DIV 2 // Shift right to divide by sizeof(T)
#endif

#ifndef ENERGY
#define ENERGY 0
#endif
#define PRINT 0

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define divceil(n, m) (((n)-1) / (m) + 1)
#define roundup(n, m) ((n / m) * m + m)

// fixed point arithmetic 
#define SHIFT_AMOUNT 10
#define SHIFT_MASK ((1 << SHIFT_AMOUNT) - 1) 

// avoid overflow
#define OFFSET 0
#define OVERFLOW_SHIFT (SHIFT_AMOUNT + OFFSET) 

#define MAX_ROWS 24

#endif
