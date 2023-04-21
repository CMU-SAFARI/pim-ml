CXX = dpcpp
MKLROOT = /home/rain/intel/oneapi/mkl/2022.0.1
CXXFLAGS = -O2 -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 \
-lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

# CXX = gcc
# MKLROOT = /home/rain/intel/oneapi/mkl/2022.0.1
# CXXFLAGS = -O2 -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread \
# -lmkl_core -liomp5 -lpthread -lm -ldl

BUFFER_EXE_NAME = lr_mkl
BUFFER_SOURCES = linear_regression_mkl.c

all: build_buffers

build_buffers:
	$(CXX) $(CXXFLAGS) -o $(BUFFER_EXE_NAME) $(BUFFER_SOURCES)

run: 
	./$(BUFFER_EXE_NAME)

clean: 
	rm -f $(BUFFER_EXE_NAME) 
