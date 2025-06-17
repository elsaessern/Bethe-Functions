#include "defs.h"
#include "goldCode.h"
#include <iostream>

MyComplexType sum(const MyComplexType a, const MyComplexType b) {
    return a+b;
}

MyComplexVector vectorSum(const MyComplexVector a, const MyComplexVector b) {
    MyComplexVector outputVector;
    for (unsigned long i = 0; i<a.size(); i++) {
        outputVector.push_back(a[i] + b[i]);
    }
    return outputVector;
}

std::pair<MyComplexArray, MyComplexArray> arraySumSq(const MyComplexArray a, const MyComplexArray b) {
    MyComplexArray outputArray, outputArraySq;

    for (unsigned long i = 0; i < a.size(); i++) {
        MyComplexVector outputVector;
        MyComplexVector outputVectorSq;
        for (unsigned long j = 0; j < a[0].size(); j++) {
            outputVector.push_back(a[i][j] + b[i][j]);
            outputVectorSq.push_back(a[i][j]*a[i][j] + b[i][j]*b[i][j]);
        }
        outputArray.push_back(outputVector);
        outputArraySq.push_back(outputVectorSq);
    }
    // return std::pair<MyComplexArray, MyComplexArray>(outputArray, outputArraySq);
    return {outputArray, outputArraySq};
}

MyComplexArray inspectArray(MyComplexArray a) {
    std::cout << a.size() << std::endl;
    std::cout << a[0].size() << std::endl;
    a[0][0] = MyComplexType(100,10);
    return a;
}

MyComplexType product(const MyComplexVector vector) {
    //Could be done with std::accumulate()
    MyComplexType result(1);
    for (auto v : vector) {
        result *= v;
    }
    return result;
}

MyComplexType intPower(const MyComplexType base, const unsigned long power, const unsigned long numSites) {
    MyComplexType result(1);
    for (unsigned long i = 0; i < numSites; i++) {
        auto include = (i < power);
        result *= (MyComplexType(include)*base + MyComplexType(!include));
    }
    return result;
}

// Runtime complexity O(N^2)
MyComplexType computeAmplitude(const MyComplexVector betheRoots, const MyLongVector sigma, const MyComplexType delta) {

    MyComplexType amplitude(1);
    
    // Find every inversion and multiply ampMult in for each inversion
    // Do this in a way that uses no conditional so that it will work on the gpu
    for (unsigned long i = 0; i < sigma.size(); i++) {
        for (unsigned long j = i+1; j < sigma.size(); j++) {
            if (sigma[i] > sigma[j]) {
                // std::cout << sigma[i] << " > " << sigma[j] << std::endl;
                auto si = sigma[i];
                auto sj = sigma[j];
                amplitude *= -(MyComplexType(1) + betheRoots[si]*betheRoots[sj] - delta*betheRoots[si])
                            / (MyComplexType(1) + betheRoots[si]*betheRoots[sj] - delta*betheRoots[sj]);
            }
            // auto inverted = (sigma[i] > sigma[j]);
            // auto ampMult = -(MyComplexType(1) + betheRoots[i]*betheRoots[j] - delta*betheRoots[j])
            //                 / (MyComplexType(1) + betheRoots[i]*betheRoots[j] - delta*betheRoots[i]);
            // amplitude *= (MyComplexType(inverted)*ampMult + MyComplexType(!inverted));

            // amplitude *= -(1 + betheRoots[invList[i,0]]*betheRoots[invList[i,1]] - delta*betheRoots[invList[i,0]])
            // /(1 + betheRoots[invList[i,0]]*betheRoots[invList[i,1]] - delta*betheRoots[invList[i,1]])

        }
    }
    return amplitude;
}

std::pair<MyComplexArray, MyComplexArray> computeBasisTransform(
        const MyComplexArray allBetheRoots, 
        const MyLongArray allConfigurations, 
        const MyComplexVector allGaudinDets,
        const MyLongArray sigma,
        const MyComplexType delta,
        const unsigned long numSites
) {
    // These should be done as 2d arrays rather than as vectors of vectors
    MyComplexArray energyToConfig, configToEnergyTrans;

    auto numBasis = allBetheRoots.size();
    auto numUp = allBetheRoots[0].size();

    for (unsigned long energyIndex = 0; energyIndex < numBasis; energyIndex++) {
        auto gaudinDet = allGaudinDets[energyIndex]; // Should be: gaudinDet = np.linalg.det(gaudinLikeMatrix(betheRoots, delta, numSites))
        auto betheProduct = product(allBetheRoots[energyIndex]);
        MyComplexVector e2c(numBasis, 0), c2e(numBasis, 0);


        for (unsigned long permIndex = 0; permIndex < sigma.size(); permIndex++) {
            auto amplitude = computeAmplitude(allBetheRoots[energyIndex], sigma[permIndex], delta);
            // std::cout << "EnergyIndex: " << energyIndex << " permIndex: " << permIndex << " amplitude= " << amplitude << std::endl;

            for (unsigned long positionIndex = 0; positionIndex < numBasis; positionIndex++) {
                MyComplexType accum(1);
                for (unsigned long i = 0; i < numUp; i++) {
                    accum *= intPower(allBetheRoots[energyIndex][sigma[permIndex][i]], allConfigurations[positionIndex][i], numSites);
                }
                // energyToConfig[energyIndex][positionIndex] += amplitude * accum;
                 
                e2c[positionIndex] += amplitude * accum;
                // if ((energyIndex == 0) && (positionIndex == 0)) {
                //     std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXX " << accum << std::endl;
                //     std::cout << "e2c[0,0] = " << e2c[positionIndex] << std::endl;
                // }
                // configToEnergyTrans[energyIndex][positionIndex] += MyComplexType(1) / ( gaudinDet * betheProduct * amplitude * accum );
                c2e[positionIndex] += MyComplexType(1) / ( gaudinDet * betheProduct * amplitude * accum );
            }
        }
        energyToConfig.push_back(e2c);
        configToEnergyTrans.push_back(c2e);
    }
    return {energyToConfig, configToEnergyTrans};
}
