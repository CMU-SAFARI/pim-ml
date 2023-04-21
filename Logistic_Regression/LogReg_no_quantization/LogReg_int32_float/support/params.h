#ifndef _PARAMS_H_
#define _PARAMS_H_

#include "common.h"

typedef struct Params {
    unsigned int   iter_time;
    float          learning_rate;
    unsigned int   m_size;
    unsigned int   n_size;
    int   n_warmup;
    int   n_reps;
    int   exp;
}Params;

static void usage() {
    fprintf(stderr,
        "\nUsage:  ./program [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -w <W>    # of untimed warmup iterations (default=1)"
        "\n    -e <E>    # of timed repetition iterations (default=3)"
        "\n    -x <X>    Weak (0) or strong (1) scaling (default=0)"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -i <I>    iteration time (default=100)"
        "\n    -l <L>    learning rate (default=0.0001)"
        "\n    -m <M>    m_size (default=8192)"
        "\n    -n <N>    n_size (default=16, max=16)"
        "\n");
}

struct Params input_params(int argc, char **argv) {
    struct Params p;

    p.iter_time      = 100;
    p.learning_rate  = 0.0001;
    p.m_size         = 8192;
    p.n_size         = 16;

    p.n_warmup       = 1;
    p.n_reps         = 3;
    p.exp            = 1;

    int opt;
    while((opt = getopt(argc, argv, "hi:l:m:n:w:e:x:")) >= 0) {
        switch(opt) {
        case 'h':
        usage();
        exit(0);
        break;
        case 'i': p.iter_time     = atoi(optarg); break;
        case 'l': p.learning_rate = atof(optarg); break;
        case 'm': p.m_size        = atoi(optarg); break;
        case 'n': p.n_size        = atoi(optarg); break;
        case 'w': p.n_warmup      = atoi(optarg); break;
        case 'e': p.n_reps        = atoi(optarg); break;
        case 'x': p.exp           = atoi(optarg); break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }
    assert(NR_DPUS > 0 && "Invalid # of dpus!");
    if (p.n_size > MAX_ROWS) {
        printf("Max num of rows is 24!\n"); 
        exit(0); 
    }

    return p;
}
#endif
