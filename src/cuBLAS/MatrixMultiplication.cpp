/**
 * A matrix multiplication using the cuBLAS library.
 */

#include <cstdlib>
#include <iostream>
#include <string>

#include <sys/time.h>

#include <cublas.h>

typedef float ScalarT;

// Some helper functions //
/**
 * Calculates 1D index from row-major order to column-major order.
 */
#define index(r,c,rows) (((c)*(rows))+(r))

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
inline void __cudaSafeCall( cublasStatus err, const char *file, const int line )
{
    if( err != CUBLAS_STATUS_SUCCESS )
    {
        std::cerr << "CUDA call failed at " << file << ":" << line << std::endl;
        exit (EXIT_FAILURE);
    }
}

#define AllocCheck( err ) __allocCheck( err, __FILE__, __LINE__ )
inline void __allocCheck( void* err, const char *file, const int line )
{
    if( err == 0 )
    {
        std::cerr << "Allocation failed at " << file << ":" << line << std::endl;
        exit (EXIT_FAILURE);
    }
}

void printMat( const ScalarT* const mat, size_t rows, size_t columns, std::string prefix = "Matrix:" )
{
    // Maximum to print
    const size_t max_rows = 5;
    const size_t max_columns = 16;

    std::cout << prefix << std::endl;
    for( size_t r = 0; r < rows && r < max_rows; ++r )
    {
        for( size_t c = 0; c < columns && c < max_columns; ++c )
        {
            std::cout << mat[index(r,c,rows)] << " ";
        }
        std::cout << std::endl;
    }
}

float getMilliseconds( struct timeval start, struct timeval end )
{
    return ( float )( ( end.tv_sec - start.tv_sec ) * 1000000 + end.tv_usec - start.tv_usec ) / 1000.0;
}

// Main program //
int main( int argc, char** argv )
{
    size_t HA = 42;
    size_t WA = 23;
    size_t WB = 13;
    size_t HB = WA;
    size_t WC = WB;
    size_t HC = HA;

    size_t r, c;

    struct timeval tAllStart, tAllEnd;
    struct timeval tKernelStart, tKernelEnd;
    float time;

    // Prepare host memory and input data //
    ScalarT* A = ( ScalarT* )malloc( HA * WA * sizeof(ScalarT) );
    AllocCheck( A );
    ScalarT* B = ( ScalarT* )malloc( HB * WB * sizeof(ScalarT) );
    AllocCheck( B );
    ScalarT* C = ( ScalarT* )malloc( HC * WC * sizeof(ScalarT) );
    AllocCheck( C );

    for( r = 0; r < HA; r++ )
    {
        for( c = 0; c < WA; c++ )
        {
            A[index(r,c,HA)] = ( ScalarT )index(r,c,HA);
        }
    }

    for( r = 0; r < HB; r++ )
    {
        for( c = 0; c < WB; c++ )
        {
            B[index(r,c,HB)] = ( ScalarT )index(r,c,HB);
        }
    }

    // Initialize cuBLAS //
    gettimeofday( &tAllStart, 0 );
    cublasStatus status;
    cublasInit();

    // Prepare device memory //
    ScalarT* dev_A;
    ScalarT* dev_B;
    ScalarT* dev_C;

    status = cublasAlloc( HA * WA, sizeof(ScalarT), ( void** )&dev_A );
    CudaSafeCall( status );

    status = cublasAlloc( HB * WB, sizeof(ScalarT), ( void** )&dev_B );
    CudaSafeCall( status );

    status = cublasAlloc( HC * WC, sizeof(ScalarT), ( void** )&dev_C );
    CudaSafeCall( status );

    gettimeofday( &tAllStart, 0 );

    status = cublasSetMatrix( HA, WA, sizeof(ScalarT), A, HA, dev_A, HA );
    CudaSafeCall( status );

    status = cublasSetMatrix( HB, WB, sizeof(ScalarT), B, HB, dev_B, HB );
    CudaSafeCall( status );

    // Call cuBLAS function //
    gettimeofday( &tKernelStart, 0 );

    // Use of cuBLAS constant CUBLAS_OP_N produces a runtime error!
    const char CUBLAS_OP_N = 'n'; // 'n' indicates that the matrices are non-transposed.
    cublasSgemm( CUBLAS_OP_N, CUBLAS_OP_N, HA, WB, WA, 1, dev_A, HA, dev_B, HB, 0, dev_C, HC ); // call for float
//    cublasDgemm( CUBLAS_OP_N, CUBLAS_OP_N, HA, WB, WA, 1, dev_A, HA, dev_B, HB, 0, dev_C, HC ); // call for double
    status = cublasGetError();
    CudaSafeCall( status );

    gettimeofday( &tKernelEnd, 0 );
    time = getMilliseconds( tKernelStart, tKernelEnd );
    std::cout << "time (kernel only): " << time << "ms" << std::endl;

    // Load result from device //
    cublasGetMatrix( HC, WC, sizeof(ScalarT), dev_C, HC, C, HC );
    CudaSafeCall( status );

    gettimeofday( &tAllEnd, 0 );
    time = getMilliseconds( tAllStart, tAllEnd );
    std::cout << "time (incl. data transfer): " << time << "ms" << std::endl;

    // Print result //
    printMat( A, HA, WA, "\nMatrix A:" );
    printMat( B, HB, WB, "\nMatrix B:" );
    printMat( C, HC, WC, "\nMatrix C:" );

    // Free CUDA memory //
    status = cublasFree( dev_A );
    CudaSafeCall( status );

    status = cublasFree( dev_B );
    CudaSafeCall( status );

    status = cublasFree( dev_C );
    CudaSafeCall( status );

    status = cublasShutdown();
    CudaSafeCall( status );

    // Free host memory //
    free( A );
    free( B );
    free( C );

    return EXIT_SUCCESS;
}
