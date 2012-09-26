/**
 *  Example for aligned 2D memory access with CUDA Pitch.
 */

#include <cstdlib>
#include <iostream>

#include "CudaPitch.h"

void print( const ScalarT* const array, const size_t width, const size_t height = 1 )
{
    const ScalarT* row = array;
    for( size_t h = 0; h < height; ++h )
    {
        for( size_t w = 0; w < width; ++w )
        {
            std::cout << row[w] << " ";
        }
        row += width;
        std::cout << std::endl;
    }
}

int main()
{
    const size_t WIDTH = 15;
    const size_t HEIGHT = 4;

    // Create and fill a 2D array
    ScalarT* mem = ( ScalarT* )malloc( sizeof(ScalarT) * WIDTH * HEIGHT );
    for( size_t i = 0; i < WIDTH * HEIGHT; ++i )
        mem[i] = i;

    std::cout << "Input array: " << std::endl;
    print( mem, WIDTH, HEIGHT );
    std::cout << std::endl;

    addOne( mem, WIDTH, HEIGHT );

    std::cout << std::endl;
    std::cout << "Result array: " << std::endl;
    print( mem, WIDTH, HEIGHT );

    free( mem );
    return EXIT_SUCCESS;
}
