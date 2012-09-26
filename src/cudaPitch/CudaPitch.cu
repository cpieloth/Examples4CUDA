#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include "CudaPitch.h"

// to prevent IDE complains about unknown CUDA keywords
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#define __shared__
#define __syncthreads();
#define CUDA_KERNEL_DIM(...)

#else
#define CUDA_KERNEL_DIM(...) <<< __VA_ARGS__ >>>

#endif

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
    if( err != cudaSuccess )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
    }
}

__global__ void cudaAddOne( ScalarT* const mem, size_t width, size_t height, size_t pitch )
{
    size_t gid = blockDim.x * blockIdx.x + threadIdx.x;
    if( gid >= width )
        return;

    // BEGIN: Alternative #1
    pitch = pitch / sizeof(ScalarT);
    ScalarT* row = mem;

    for( size_t h = 0; h < height; ++h )
    {
        row[gid] += 1;
        row += pitch;
    }
    // END: Alternative #1

    // BEGIN: Alternative #2
//    ScalarT* row;
//    for( size_t h = 0; h < height; ++h )
//    {
//        row = ( ScalarT* )( ( char* )mem + h * pitch );
//        row[gid] += 1;
//    }
    // END: Alternative #2
}

/**
 * Add 1 to each item.
 */
void addOne( ScalarT* const mem, size_t width, size_t height )
{
    ScalarT* dev_mem = NULL;
    size_t pitch;

    // Allocate an aligned 2D memory
    CudaSafeCall( cudaMallocPitch(&dev_mem, &pitch, width * sizeof(ScalarT), height) );
    // Copy the input on the device, considering the pitched memory!
    CudaSafeCall(
                    cudaMemcpy2D(dev_mem, pitch, mem, width*sizeof(ScalarT), width*sizeof(ScalarT), height, cudaMemcpyHostToDevice) );

    printf( "pitch (bytes): %zd\n", pitch );
    printf( "pitch (elements): %zd\n", pitch / sizeof(ScalarT) );

    size_t threadsPerBlock = 32;
    size_t blocksPerGrid = ( width + threadsPerBlock - 1 ) / threadsPerBlock;

    cudaAddOne CUDA_KERNEL_DIM(blocksPerGrid, threadsPerBlock) (dev_mem, width, height, pitch);

    // Copy the device memory back to host, considering the pitched memory!
    CudaSafeCall(
                    cudaMemcpy2D( mem, width * sizeof(ScalarT), dev_mem, pitch, width * sizeof(ScalarT), height, cudaMemcpyDeviceToHost ) );

    CudaSafeCall( cudaFree((void*)dev_mem) );
}

/**
 * Copies a matrix (height rows of width bytes each) from the memory area pointed to by src to the memory area pointed to by dst,
 * where kind is one of cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, or cudaMemcpyDeviceToDevice,
 * and specifies the direction of the copy.
 * dpitch and spitch are the widths in memory in bytes of the 2D arrays pointed to by dst and src, including any padding added to the end of each row.
 * The memory areas may not overlap. width must not exceed either dpitch or spitch.
 * Calling cudaMemcpy2D() with dst and src pointers that do not match the direction of the copy results in an undefined behavior.
 * cudaMemcpy2D() returns an error if dpitch or spitch exceeds the maximum allowed.
 *
 * \param dst     - Destination memory address
 * \param dpitch  - Pitch of destination memory
 * \param src     - Source memory address
 * \param spitch  - Pitch of source memory
 * \param width   - Width of matrix transfer (columns in bytes)
 * \param height  - Height of matrix transfer (rows)
 * \param kind    - Type of transfer
 *
 * \return ...
 *
 * cudaError_t cudaMemcpy2D( void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height,
                enum cudaMemcpyKind kind );
 */
