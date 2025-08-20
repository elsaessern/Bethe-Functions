#ifndef GPUCODE_H_
#define GPUCODE_H_

#include "defs.h"

MyComplexVector gpuSum(const MyComplexVector a, const MyComplexVector b);
MyLongVector thrustListTest(const MyLongVector input);

std::pair<MyComplexVector, MyComplexVector> gpuComputeBasisTransform(
        const MyComplexVector allBetheRoots, // 2d vector flattened into a nBasis * nUp vector
        const MyLongVector allConfigs, // 2d vector flattened into a nBasis * nUp vector
        const MyComplexVector allGaudinDets, // 1d vector nBasis
        const MyLongVector sigma, // 2d vector flattened into a nPermutations * nUp vector
        const MyComplexType delta
);

MyComplexVector gpuComputeBasisTransformDegen(
        const MyComplexVector allBetheRoots, // 2d vector flattened into a nBasis * nUp vector
        const MyLongVector allConfigs, // 2d vector flattened into a nBasis * nUp vector
        const MyLongVector sigma, // 2d vector flattened into a nPermutations * nUp vector
        const MyComplexType delta,
        const MyLongType nUp
);

#endif