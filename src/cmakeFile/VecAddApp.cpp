/**
 * Simple vector addition on CUDA device.
 */

#include <cstdlib>
#include <ctime>
#include <iostream>

#include "VecAddHelper.hpp"
#include "VectorAddition.h"

int main()
{
    size_t N = 128;
    size_t size = N * sizeof(ScalarT);

    // Allocate host memory
    ScalarT* A = ( ScalarT* )malloc( size );
    AllocCheck( A );
    ScalarT* B = ( ScalarT* )malloc( size );
    AllocCheck( B );
    ScalarT* C = ( ScalarT* )malloc( size );
    AllocCheck( C );

    // Initialize input vectors //
    srand( time( 0 ) );
    for( size_t i = 0; i < N; ++i )
    {
        A[i] = rand() % 500;
        B[i] = rand() % 500;
    }

    vecAdd( C, A, B, N );

    printVec( A, N, "\nVector A:" );
    printVec( B, N, "\nVector B:" );
    printVec( C, N, "\nVector C:" );

    // Free host memory //
    free( A );
    free( B );
    free( C );
}
