#ifndef VECTORADDITION_HPP_
#define VECTORADDITION_HPP_

#include <stddef.h>

typedef int ScalarT;

/**
 * Calculates the sum of the vector A and B.
 *
 * \param C Result vector.
 * \param A Vector A.
 * \param B Vector B.
 * \param N Vector size of A, B and C.
 */
void vecAdd( ScalarT* const c, const ScalarT* const a, const ScalarT* const b, size_t N );

#endif /* VECTORADDITION_HPP_ */
