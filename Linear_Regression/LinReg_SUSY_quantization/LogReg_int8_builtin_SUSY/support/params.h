#ifndef _PARAMS_H_
#define _PARAMS_H_

#include "common.h"

typedef struct Params {
    unsigned int   iter_time;
    float          learning_rate;
    unsigned int   m_size;
    unsigned int   n_size;
}Params;

static void usage() {
    fprintf(stderr,
        "\nUsage:  ./program [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -i <I>    iteration time (default=600)"
        "\n    -l <L>    learning rate (default=0.002)"
        "\n    -m <M>    m_size (default=5000000)"
        "\n    -n <N>    n_size (default=18, max=24)"
        "\n");
}

struct Params input_params(int argc, char **argv) {
    struct Params p;

    p.iter_time      = 600;
    p.learning_rate  = 0.002;
    p.m_size         = 5000000;
    p.n_size         = 18;

    int opt;
    while((opt = getopt(argc, argv, "hi:l:m:n:")) >= 0) {
        switch(opt) {
        case 'h':
        usage();
        exit(0);
        break;
        case 'i': p.iter_time     = atoi(optarg); break;
        case 'l': p.learning_rate = atof(optarg); break;
        case 'm': p.m_size        = atoi(optarg); break;
        case 'n': p.n_size        = atoi(optarg); break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }
    assert(NR_DPUS > 0 && "Invalid # of dpus!");
    if (p.n_size > MAX_ROWS) {
        printf("max num of rows is %d!\n", MAX_ROWS); 
        exit(0);
    }

    return p;
}
#endif
