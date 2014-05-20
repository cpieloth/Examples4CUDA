#ifndef VECADDHELPER_HPP_
#define VECADDHELPER_HPP_

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
 * Prints the vector to stdout.
 *
 * \param vec Vector to print.
 * \param n Size of the vector,
 * \param prefix Custom string is printed in front of the vector output.
 */
void printVec( const ScalarT* const vec, size_t n, std::string prefix );

inline void printVec( const ScalarT* const vec, size_t n, std::string prefix = "Vector:" )
{
    // Maximum to print
    const size_t max_items = 16;

    std::cout << prefix << std::endl;
    for( size_t i = 0; i < n && i < max_items; ++i )
    {
        std::cout << vec[i] << " ";
    }
    std::cout << std::endl;
}

#endif /* VECADDHELPER_HPP_ */
