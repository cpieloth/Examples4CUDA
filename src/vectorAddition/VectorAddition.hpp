#ifndef VECADD_HPP_
#define VECADD_HPP_

#include <string>

typedef int ScalarT;

#define AllocCheck( err ) __allocCheck( err, __FILE__, __LINE__ )
inline void __allocCheck( void* err, const char *file, const int line )
{
    if( err == 0 )
    {
        std::cerr << "Allocation failed at " << file << ":" << line << std::endl;
        exit (EXIT_FAILURE);
    }
}

/**
 * Calculates the sum of the vector A and B.
 *
 * \param C Result vector.
 * \param A Vector A.
 * \param B Vector B.
 * \param N Vector size of A, B and C.
 */
void vecAdd( ScalarT* const C, const ScalarT* const A, const ScalarT* const B, size_t N );

/**
 * Prints the vector to stdout.
 *
 * \param vec Vector to print.
 * \param n Size of the vector,
 * \param prefix Custom string is printed in front of the vector output.
 */
void printVec( const ScalarT* const vec, size_t n, std::string prefix );

#endif /* VECADD_HPP_ */
