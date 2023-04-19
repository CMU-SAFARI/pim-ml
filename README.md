# pim-ml
PIM-ML is a benchmark for training machine learning algorithms on [UPMEM](https://www.upmem.com/) architecture, which is a real-world- processing-in-memory architecture. The UPMEM architecture integrate tranditional DRAM memory and general-purpose in-order cores, called DRAM Processing Units (DPUs), in the same chip. 

PIM-ML includes four machine learning algorithms: Linear Regression, Logistic Regression, K-means, and Decision Trees.

## Citation
Juan Gómez-Luna, Yuxin Guo, Sylvan Brocard, Julien Legriel, Remy Cimadomo, Geraldo F. Oliveira, Gagandeep Singh, Onur Mutlu, "[An Experimental Evaluation of Machine Learning Training on a Real Processing-in-Memory System](https://arxiv.org/abs/2207.07886)", arXiv:2207.07886 [cs.AR], 2022. 

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
All benchmark folders of Linear Regression and Logistic Regression are similar to the one shown in int32 and float version of Linear Regression (LRGD_int32_float). 
```
+-- LRGD_int32_float/
|   +-- dpu/
|   +-- host/
|   +-- support/
|   +-- Makefile
|   +-- run_strong.py
|   +-- run_tasklet.py
|   +-- run_weak.py
```

### K-means and Decision Trees 
TODO

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

By default, Linear Regression and Logistic Regression benchmarks use synthetic dataset. To use real dataset, the BENCHMARK_FOLDER/host/app.c file should be editted to read the real dataset from a specific file. 

### K-means and Decision Trees 
TODO

### Getting Help 
If you have any suggestions for improvement, please contact xxx. 
If you find any bugs or have further questions or requests, please post an issue at the [issue page](https://github.com/CMU-SAFARI/pim-ml/issues).

## Acknowledgments 
