#ifndef DEFS_H_
#define DEFS_H_

#include <vector>
#include <complex>
#include <thrust/complex.h>


typedef std::complex<double> MyComplexType;
typedef std::vector<MyComplexType> MyComplexVector;
typedef std::vector<MyComplexVector> MyComplexArray;

typedef unsigned long MyLongType;
typedef std::vector<MyLongType> MyLongVector;
typedef std::vector<MyLongVector> MyLongArray;

#endif