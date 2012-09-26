CXX=g++
CXXFLAGS=-O2

NVCC=nvcc
NVCCFLAGS=

# OS DEPENDENCIES
OSTYPE = $(shell uname)
ifeq ($(OSTYPE),Linux)
ifndef CUDA_INSTALL_PATH
	CUDA_INSTALL_PATH=/usr/local/cuda
endif # CUDA_INSTALL_PATH
	INCLUDES=-I$(CUDA_INSTALL_PATH)/include
	LDFLAGS=-L$(CUDA_INSTALL_PATH)/lib64
endif # Linux

LIBS=-lcudart
OBJS=CudaPitch.o
CUOBJS=CudaPitch.cuo
EXEC=CudaPitch

all: $(EXEC) 

$(EXEC): $(OBJS) $(CUOBJS)
	$(CXX) $(LDFLAGS) -o $(EXEC) $(OBJS) $(CUOBJS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< 
	
%.cuo: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	$(RM) *.o *.cuo *.so *.a $(EXEC)

