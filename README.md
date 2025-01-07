# PIM-ML
PIM-ML is a benchmark for training machine learning algorithms on the [UPMEM](https://www.upmem.com/) architecture, which is the first publicly-available real-world processing-in-memory (PIM) architecture. The UPMEM architecture integrates DRAM memory banks and general-purpose in-order cores, called DRAM Processing Units (DPUs), in the same chip. 

PIM-ML is designed to understand the potential of modern general-purpose PIM architectures to accelerate machine learning training. 
PIM-ML implements several representative classic machine learning algorithms: 
- Linear Regression 
- Logistic Regression 
- Decision Tree
- K-means Clustering

## Citation
Please cite the following papers if you find this repository useful.

ISPASS2023 paper version:

Juan Gómez-Luna, Yuxin Guo, Sylvan Brocard, Julien Legriel, Remy Cimadomo, Geraldo F. Oliveira, Gagandeep Singh, and Onur Mutlu, "[Evaluating Machine Learning Workloads on Memory-Centric Computing Systems](https://ispass.org/ispass2023/)", 2023 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS), 2023.

Bibtex entry for citation:
```
@inproceedings{gomezluna2022ispass,
  title={{Evaluating Machine Learning Workloads on Memory-Centric Computing Systems}}, 
  author={Juan Gómez-Luna and Yuxin Guo and Sylvan Brocard and Julien Legriel and Remy Cimadomo and Geraldo F. Oliveira and Gagandeep Singh and Onur Mutlu},
  year={2023},
  booktitle = {ISPASS}
}
```

ISVLSI2023 paper version:

Juan Gómez-Luna, Yuxin Guo, Sylvan Brocard, Julien Legriel, Remy Cimadomo, Geraldo F. Oliveira, Gagandeep Singh, and Onur Mutlu, "[Machine Learning Training on a Real Processing-in-Memory System](https://doi.org/10.1109/ISVLSI54635.2022.00064)". 2022 IEEE Computer Society Annual Symposium on VLSI (ISVLSI), 2022.

Bibtex entry for citation:
```
@inproceedings{gomezluna2022isvlsi,
      title={{Machine Learning Training on a Real Processing-in-Memory System}}, 
      author={Juan Gómez-Luna and Yuxin Guo and Sylvan Brocard and Julien Legriel and Remy Cimadomo and Geraldo F. Oliveira and Gagandeep Singh and Onur Mutlu},
      booktitle={ISVLSI}, 
      year={2022}
}
```

arXiv paper version:

Juan Gómez-Luna, Yuxin Guo, Sylvan Brocard, Julien Legriel, Remy Cimadomo, Geraldo F. Oliveira, Gagandeep Singh, and Onur Mutlu, "[An Experimental Evaluation of Machine Learning Training on a Real Processing-in-Memory System](https://arxiv.org/abs/2207.07886)", arXiv:2207.07886 [cs.AR], 2022. 

Bibtex entries for citation:
```
@misc{gomezluna2023experimental,
      title={{An Experimental Evaluation of Machine Learning Training on a Real Processing-in-Memory System}}, 
      author={Juan Gómez-Luna and Yuxin Guo and Sylvan Brocard and Julien Legriel and Remy Cimadomo and Geraldo F. Oliveira and Gagandeep Singh and Onur Mutlu},
      year={2023},
      eprint={2207.07886},
      archivePrefix={arXiv},
      primaryClass={cs.AR}
}
```

## Installation
### Prerequisites
Running PIM-ML requires installing the [UPMEM SDK](https://sdk.upmem.com/). This benchmark designed to run on a server with real UPMEM modules, but they are also able to be run by the functional simulator in the UPMEM SDK.

###  Getting Started 
Clone the repository:
```sh
git clone https://github.com/CMU-SAFARI/pim-ml.git
cd pim-ml
```

## Repository Structure
### Linear Regression and Logistic Regression
All benchmark folders of Linear Regression and Logistic Regression are similar to the one shown in `/Linear_Regression/LinReg_no_quantization/LogReg_int32_float`. 
```
.
├── Linear_Regression
│   ├── Baseline
│   │   ├── CPU_MKL
│   │   │   ├── Makefile
│   │   │   ├── linear_regression_mkl.c
│   │   │   └── params.h
│   │   └── GPU_cuBLAS
│   │       ├── Makefile
│   │       └── kernel.cu
│   ├── LinReg_SUSY_quantization
│   │   ├── LogReg_int32_SUSY
│   │   │   ├── ...
│   │   ├── LogReg_int8_SUSY
│   │   │   ├── ...
│   │   └── LogReg_int8_builtin_SUSY
│   │       ├── ...
│   ├── LinReg_no_quantization
│   │   ├── LogReg_int32_float
│   │   │   ├── Makefile
│   │   │   ├── dpu
│   │   │   │   └── task.c
│   │   │   ├── host
│   │   │   │   └── app.c
│   │   │   ├── run_strong.py
│   │   │   ├── run_tasklet.py
│   │   │   ├── run_weak.py
│   │   │   └── support
│   │   │       ├── common.h
│   │   │       ├── params.h
│   │   │       └── timer.h
│   │   ├── LogReg_int8
│   │   │   ├── ...
│   │   └── LogReg_int8_builtin
│   │       ├── ...
│   └── LinReg_quantization
│       ├── LogReg_int32_float
│       │   ├── ...
│       ├── LogReg_int8
│       │   ├── ...
│       └── LogReg_int8_builtin
│           ├── ...
├── Logistic_Regression
│   ├── Baseline
│   │   ├── ...
│   ├── LogReg_SUSY_quantization
│   │   ├── LogReg_int32_SUSY
│   │   │   ├── ...
│   │   └── LogReg_int8_SUSY
│   │       ├── ...
│   ├── LogReg_no_quantization
│   │   ├── LogReg_int32LUTMRAM_float
│   │   │   ├── ...
│   │   ├── LogReg_int32LUT_float
│   │   │   ├── ...
│   │   ├── LogReg_int32_float
│   │   │   ├── ...
│   │   ├── LogReg_int8
│   │   │   ├── ...
│   │   └── LogReg_int8_builtin
│   │       ├── ...
│   └── LogReg_quantization
│       ├── LogReg_int32LUTMRAM_float
│       │   ├── ...
│       ├── LogReg_int32LUT_float
│       │   ├── ...
│       ├── LogReg_int32_float
│       │   ├── ...
│       ├── LogReg_int8
│       │   ├── ...
│       └── LogReg_int8_builtin
│           ├── ...
│── README.md
└── LICENSE
```

### Decision Tree and K-Means Clustering
PIM-ML implementations of Decision Tree and K-Means Clustering can be found in the following repositories.
- Decision Tree: https://github.com/upmem/scikit-dpu
- K-Means Clustering: https://github.com/upmem/dpu_kmeans

## Running PIM-ML 
### Linear Regression and Logistic Regression
Each benckmark folder includes Python scripts to run experiments automatically:
- `run_strong.py`: Strong scaling experiment for benchmark using 4 to 32 rank of UPMEM DPUs (256 to 2048 DPUs).
- `run_tasklet.py`: Experiment benchmark using 1 UPMEM DPU with various tasklets per DPU (1 to 24). 
- `run_weak.py`: Weak scaling experiment for benchmark using 1 rank of UPMEM DPUs (1 to 64 DPUs).

To use these scripts, update `rootdir` in the beginning of each script.
```sh
cd pim-ml/LRGD_int32_float
# Weak scaling experiments for BFS 
python3 run_weak.py LRGD
```

Inside each benchmark folder, one can compile and run each benchmark with different input parameters. 
```sh
cd pim-ml/LRGD_int32_float

# Compile LRGD_int32_float for 32 DPUs and 16 tasklets (i.e., software threads) per DPU
NR_DPUS=32 NR_TASKLETS=16 make all

# -i set the number of epochs, -l set the value of learning rate, -m set the number of samples of the dataset, -n set the number of features 
./bin/host_code -i 100 -l 0.001 -m 1024 -n 16
```

By default, Linear Regression and Logistic Regression benchmarks use synthetic dataset. To use real dataset, the ```BENCHMARK_FOLDER/host/app.c``` should be editted to read the real dataset from a specific file. 

### Decision Tree and K-Means Clustering
Check the following documents.
- Decision Tree: https://github.com/upmem/scikit-dpu/blob/master/README.rst
- K-Means Clustering: https://github.com/upmem/dpu_kmeans/blob/master/README.md

## Getting Help 
If you have any suggestions for improvement, please contact yuxin.guo.007 at gmail dot com and el1goluj at gmail dot com. 
If you find any bugs or have further questions or requests, please post an issue at the [issue page](https://github.com/CMU-SAFARI/pim-ml/issues).

## Acknowledgments 
We acknowledge the generous gifts provided by our industrial partners, including ASML, Facebook, Google, Huawei, Intel, Microsoft, and VMware. We acknowledge support from the Semiconductor Research Corporation, the ETH Future Computing Laboratory, and the BioPIM project. 
