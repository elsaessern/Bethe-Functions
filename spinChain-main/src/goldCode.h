#ifndef GOLDCODE_H_
#define GOLDCODE_H_

#include "defs.h"

MyComplexType sum(const MyComplexType a, const MyComplexType b);

MyComplexVector vectorSum(const MyComplexVector a, const MyComplexVector b);

std::pair<MyComplexArray, MyComplexArray> arraySumSq(const MyComplexArray a, const MyComplexArray b);

MyComplexArray inspectArray(MyComplexArray a);

MyComplexType product(const MyComplexVector vector);

MyComplexType intPower(const MyComplexType base, const unsigned long power, const unsigned long numSites);

MyComplexType computeAmplitude(const MyComplexVector betheRoots, const MyLongVector sigma, const MyComplexType delta);

std::pair<MyComplexArray, MyComplexArray> computeBasisTransform(
        const MyComplexArray allBetheRoots, 
        const MyLongArray allConfigurations, 
        const MyComplexVector allGaudinDets,
        const MyLongArray sigma,
        const MyComplexType delta,
        const unsigned long numSites
);

#endif