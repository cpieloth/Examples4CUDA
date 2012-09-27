CXX=g++
CXXFLAGS=-O2

# OS DEPENDENCIES
OSTYPE = $(shell uname)
ifeq ($(OSTYPE),Linux)
ifndef CUDA_INSTALL_PATH
	CUDA_INSTALL_PATH=/usr/local/cuda
endif # CUDA_INSTALL_PATH
	INCLUDES=-I$(CUDA_INSTALL_PATH)/include
	LDFLAGS=-L$(CUDA_INSTALL_PATH)/lib64
endif # Linux

LIBS=-lcublas
OBJS=MatrixMultiplication.o
EXEC=MatrixMultiplication

all: $(EXEC) 

$(EXEC): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< 

clean:
	$(RM) *.o *.so *.a $(EXEC)
       