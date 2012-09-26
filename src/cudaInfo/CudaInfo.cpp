/**
 * This program prints information about the available CUDA devices.
 */

#include <cstdlib>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

int main( int argc, char** argv )
{
    std::cout << "::: CUDA Device Info :::" << std::endl;
    CUresult r = cuInit( 0 );
    if( r != CUDA_SUCCESS )
        return EXIT_FAILURE;

    int devCount = 0;
    r = cuDeviceGetCount( &devCount );
    if( r != CUDA_SUCCESS )
        return EXIT_FAILURE;

    std::cout << "Device Count: " << devCount << std::endl;
    cudaDeviceProp props;

    for( int i = 0; i < devCount; i++ )
    {
        std::cout << std::endl;

        cudaGetDeviceProperties( &props, i );

        std::cout << "#" << i << ": " << props.name << std::endl;
        std::cout << "\tcompute capability: " << props.major << "." << props.minor << std::endl;
        std::cout << "\tmultiProcessorCount: " << props.multiProcessorCount << std::endl;
        std::cout << "\tmaxGridSize[0]: " << props.maxGridSize[0] << std::endl;
        std::cout << "\tmaxThreadsDim[0]: " << props.maxThreadsDim[0] << std::endl;
        std::cout << "\tmaxThreadsPerBlock: " << props.maxThreadsPerBlock << std::endl;
        std::cout << "\tmaxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "\tregsPerBlock: " << props.regsPerBlock << std::endl;
        std::cout << "\twarpSize: " << props.warpSize << std::endl;
        std::cout << "\ttotalGlobalMem: " << props.totalGlobalMem / 1024 << " KB" << std::endl;
        std::cout << "\ttotalConstMem: " << props.totalConstMem / 1024 << " KB" << std::endl;
        std::cout << "\tsharedMemPerBlock: " << props.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "\tclockRate: " << props.clockRate / 1000 << " MHz" << std::endl;
        std::cout << "\tmemoryClockRate: " << props.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "\tconcurrentKernels: " << props.concurrentKernels << std::endl;
        std::cout << "\tcomputeMode: " << props.computeMode << std::endl;
        std::cout << "\tintegrated: " << props.integrated << std::endl;
    }

    return EXIT_SUCCESS;
}
