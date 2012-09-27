CUDA Examples
=============

A collection of CUDA example code.

This repository collects a few brief CUDA examples. The aim is to to explain a chosen functionality or principle in a few lines of code. 
Plain makefiles show used libraries and header files and allow customization of existing build scripts.

Requirements:

* C++ compiler e.g. `g++`
* `make` tool
* CUDA SDK
* CUDA enabled graphics card

How to compile and run:
* switch to the directory e.g. `cd <path to repository>/src/cudaInfo`
* run `make`
* run the executable e.g. `./CudaInfo`

Possible problems:

1. CUDA libraries or header files not found. You may have to set `CUDA_INSTALL_PATH`. More information on
`http://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/CUDA_C_Getting_Started_Windows.pdf`
`http://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/CUDA_C_Getting_Started_Linux.pdf`
`http://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/CUDA_C_Getting_Started_Mac.pdf`

