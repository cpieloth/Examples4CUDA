#include <iostream>

#include <cuda_runtime.h>

#include "VectorAddition.hpp"

// to prevent IDE complains about unknown CUDA keywords
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#define __shared__
uint3 threadIdx;
uint3 blockIdx;
dim3 blockDim;
dim3 gridDim;
int warpSize;
#define CUDA_KERNEL_DIM(...)
#else
#define CUDA_KERNEL_DIM(...)  <<< __VA_ARGS__ >>>
#endif

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( err != cudaSuccess )
    {
        std::cerr << "CUDA call failed at " << file << ":" << line << " : " << cudaGetErrorString( err ) << std::endl;
        exit (EXIT_FAILURE);
    }
}

__global__ void cudaVecAdd( ScalarT* const C, const ScalarT* const A, const ScalarT* const B, size_t N )
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if( gid < N )
        C[gid] = A[gid] + B[gid];
}

void vecAdd( ScalarT* const c, const ScalarT* const a, const ScalarT* const b, size_t N )
{
    cudaError status;
    const size_t size = N * sizeof(ScalarT);
    // Prepare device memory //
    ScalarT* dev_A;
    ScalarT* dev_B;
    ScalarT* dev_C;

    status = cudaMalloc( &dev_A, size );
    CudaSafeCall( status );

    status = cudaMalloc( &dev_B, size );
    CudaSafeCall( status );

    status = cudaMalloc( &dev_C, size );
    CudaSafeCall( status );

    status = cudaMemcpy( dev_A, a, size, cudaMemcpyHostToDevice );
    CudaSafeCall( status );

    status = cudaMemcpy( dev_B, b, size, cudaMemcpyHostToDevice );
    CudaSafeCall( status );

    // Call CUDA kernel //
    size_t threadsPerBlock = 32;
    size_t blocksPerGrid = ( N + threadsPerBlock - 1 ) / threadsPerBlock;
    std::cout << "Threads/Block: " << threadsPerBlock << std::endl;
    std::cout << "Blocks/Grid: " << blocksPerGrid << std::endl;

    cudaVecAdd CUDA_KERNEL_DIM( blocksPerGrid, threadsPerBlock )( dev_C, dev_A, dev_B, N );

    // Load result from device //
    status = cudaMemcpy( c, dev_C, size, cudaMemcpyDeviceToHost );
    CudaSafeCall( status );

    // Free CUDA memory //
    status = cudaFree( dev_A );
    CudaSafeCall( status );

    status = cudaFree( dev_B );
    CudaSafeCall( status );

    status = cudaFree( dev_C );
    CudaSafeCall( status );
}
