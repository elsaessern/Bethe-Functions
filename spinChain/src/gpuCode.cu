#include "defs.h"
#include "gpuCode.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
// #include <thrust/fill.h>
#include <thrust/transform.h>
// #include <thrust/transform_reduce.h>
// #include <thrust/inner_product.h>
// #include <thrust/iterator/constant_iterator.h>
// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/functional.h>

typedef thrust::complex<double> GPUComplexType;

MyComplexVector gpuSum(const MyComplexVector a, const MyComplexVector b) {
    // thrust::host_vector<GPUComplexType> h_a(a.begin(), a.end());
    // thrust::host_vector<GPUComplexType> h_b(b.begin(), b.end());

    // for (int i = 0; i < a.size(); i++) {
    //     h_a[i] = a[i];
    //     h_b[i] = b[i];
    // }

    thrust::device_vector<GPUComplexType> d_a(a.begin(), a.end());
    thrust::device_vector<GPUComplexType> d_b(b.begin(), b.end());
    thrust::device_vector<GPUComplexType> d_c(a.size());
    

    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(), thrust::plus<GPUComplexType>());
    thrust::host_vector<GPUComplexType> h_result = d_c;
    MyComplexVector result(h_result.begin(), h_result.end());
    return result;
}

MyLongVector thrustListTest(const MyLongVector input) {
    auto r = thrust::counting_iterator<long>(0);
    thrust::device_vector<MyLongType> d_input(input.begin(), input.end());
    thrust::device_vector<MyLongType> d_output(input.size());
	// const MyLongVector* inputPtr = thrust::raw_pointer_cast(&d_input[0]);
    const MyLongType* inputPtr = thrust::raw_pointer_cast(&d_input[0]);
    MyLongType* outputPtr = thrust::raw_pointer_cast(&d_output[0]);

	auto listTest = [inputPtr, outputPtr] __device__ (int i) {
        outputPtr[i] = 2*inputPtr[i];
		// translateParticle(translateDistance, &direction[i*d_nDim], &pos[i*d_nDim], &translatedPos[i*d_nDim]);
	};

	thrust::for_each(r, r+d_input.size(), listTest);
	thrust::host_vector<MyLongType> h_output(d_output);
    MyLongVector output(h_output.begin(), h_output.end());
    return output;
}

// Can't work directly on device vectors, can only use them cast to raw pointers
inline __device__ GPUComplexType gpuProduct(const GPUComplexType* vector, const unsigned long size) {
    GPUComplexType result(1);
    for (unsigned long i = 0; i < size; i++) {
    // for (auto v : vector) {
        result *= vector[i];
    }
    return result;
}

// inline __device__ GPUComplexType intPower(const GPUComplexType base, const unsigned long power) {
inline __device__ GPUComplexType intPower(GPUComplexType base, unsigned long power) {

    GPUComplexType result(1);

    // for (unsigned long i = 0; i < power; i++) { // 19 seconds for L=13, N=6, 630 seconds for L=15, N=7
    //     result *= base;
    // }

    while (power > 0) { // 18.6 seconds for L=13, N=6, 574 seconds for L=15, N=7
        if (power % 2 == 1) {
            result *= base;
        }
        base *= base;
        power /= 2;
    }
    // for (unsigned long i = 0; i < nSites; i++) {
    //     // TODO: Test and remove the optimization to remove if statement.  Let the compiler deal with this instead
    //     // auto include = (i < power); // 65 seconds for L=13, N=6
    //     // result *= (GPUComplexType(include)*base + GPUComplexType(!include));
    //     if (i < power) { // 23 seconds for L=13, N=6
    //         result *= base;
    //     }
    // }
    return result;
}


//The below method, using a denominator and a numerator confers no speed advantage (and also gives wrong answers as written)
inline __device__ GPUComplexType gpuDenominator(const GPUComplexType* betheRoots, const MyLongType nUp, const GPUComplexType delta) {

    GPUComplexType denominator(1);

    for (unsigned long i = 0; i < nUp; i++) {
        for (unsigned long j = i+1; j < nUp; j++) {
            denominator *= (GPUComplexType(1) + betheRoots[i]*betheRoots[j] - delta*betheRoots[j]);
        }
    }
    return denominator;
}

inline __device__ GPUComplexType gpuNumerator(const GPUComplexType* betheRoots, const MyLongType* sigma, const MyLongType nUp, const GPUComplexType delta) {

    GPUComplexType numerator(1);

    for (unsigned long i = 0; i < nUp; i++) {
        for (unsigned long j = i+1; j < nUp; j++) {
            if (sigma[i] > sigma[j]) {
                auto si = sigma[i];
                auto sj = sigma[j];
                numerator *= -(GPUComplexType(1) + betheRoots[si]*betheRoots[sj] - delta*betheRoots[si]);
            }
        }
    }
    return numerator;
}


// Runtime complexity O(N^2)
inline __device__ GPUComplexType gpuAmplitude(const GPUComplexType* betheRoots, const MyLongType* sigma, const MyLongType nUp, const GPUComplexType delta) {

    GPUComplexType amplitude(1);
    
    // Find every inversion and multiply ampMult in for each inversion
    // Do this in a way that uses no conditional so that it will work on the gpu

    // TODO: This is super redundant since we're doing it multiple times for identical data.  Can we do this just once ahead of time? 
    // I guess I could just use the same indexptr stuff to use the precomputed values
    for (unsigned long i = 0; i < nUp; i++) {
        for (unsigned long j = i+1; j < nUp; j++) {
            if (sigma[i] > sigma[j]) {
                auto si = sigma[i];
                auto sj = sigma[j];
                amplitude *= -(GPUComplexType(1) + betheRoots[si]*betheRoots[sj] - delta*betheRoots[si])
                            / (GPUComplexType(1) + betheRoots[si]*betheRoots[sj] - delta*betheRoots[sj]);
            }

            // auto inverted = (sigma[i] > sigma[j]);
            // auto ampMult = -(GPUComplexType(1) + betheRoots[si]*betheRoots[sj] - delta*betheRoots[si])
            //                 / (GPUComplexType(1) + betheRoots[si]*betheRoots[sj] - delta*betheRoots[sj]);
            // amplitude *= (GPUComplexType(inverted)*ampMult + GPUComplexType(!inverted));
        }
    }
    return amplitude;
}


std::pair<MyComplexVector, MyComplexVector> gpuComputeBasisTransform(
        const MyComplexVector allBetheRoots, // 2d vector flattened into a nBasis * nUp vector
        const MyLongVector allConfigs, // 2d vector flattened into a nBasis * nUp vector
        const MyComplexVector allGaudinDets, // 1d vector nBasis
        const MyLongVector sigma, // 2d vector flattened into a nPermutations * nUp vector
        const MyComplexType delta
) {
    auto nBasis = allGaudinDets.size();
    auto nUp = allBetheRoots.size()/nBasis;
    auto nPerm = sigma.size()/nUp;

    //Store the output data here
    thrust::device_vector<GPUComplexType> d_e2c(nBasis*nBasis, MyComplexType(0)), d_c2e(nBasis*nBasis, MyComplexType(0));

    //Turn everything into device vectors
    thrust::device_vector<GPUComplexType> d_allBetheRoots(allBetheRoots.begin(), allBetheRoots.end());
    thrust::device_vector<MyLongType> d_allConfigs(allConfigs.begin(), allConfigs.end());
    thrust::device_vector<GPUComplexType> d_allGaudinDets(allGaudinDets.begin(), allGaudinDets.end());
    thrust::device_vector<MyLongType> d_sigma(sigma.begin(), sigma.end());

    // Cast everything to raw pointers
    GPUComplexType* e2cPtr = thrust::raw_pointer_cast(&d_e2c[0]);
    GPUComplexType* c2ePtr = thrust::raw_pointer_cast(&d_c2e[0]);

    const GPUComplexType* allBetheRootsPtr = thrust::raw_pointer_cast(&d_allBetheRoots[0]);
    const MyLongType* allConfigsPtr = thrust::raw_pointer_cast(&d_allConfigs[0]);
    const GPUComplexType* allGaudinDetsPtr = thrust::raw_pointer_cast(&d_allGaudinDets[0]);
    const MyLongType* sigmaPtr = thrust::raw_pointer_cast(&d_sigma[0]);

    // Create the lambda function to do the actual calculation

    auto r = thrust::counting_iterator<long>(0);

    auto computeBasisVectorMemOpt2 = [=] __device__ (int energyIndex) { // 19 seconds for L=13 N=6

        auto gaudinDet = allGaudinDetsPtr[energyIndex];
        auto betheProduct = gpuProduct(&allBetheRootsPtr[energyIndex*nUp], nUp); // Index raveling
        // Make the vector to hold the temp values
        GPUComplexType e2cVector[1716], c2eVector[1716]; 
        // Zero out the values
        for (unsigned long i = 0; i < 1716; i++) {
            e2cVector[i] = 0;
            c2eVector[i] = 0;
        }

        for (unsigned long permIndex = 0; permIndex < nPerm; permIndex++) {
            auto amplitude = gpuAmplitude(&allBetheRootsPtr[energyIndex*nUp], &sigmaPtr[permIndex*nUp], nUp, delta);
            // std::cout << "EnergyIndex: " << energyIndex << " permIndex: " << permIndex << " amplitude= " << amplitude << std::endl;

            for (unsigned long positionIndex = 0; positionIndex < nBasis; positionIndex++) {
                GPUComplexType accum(1);
                for (unsigned long i = 0; i < nUp; i++) {
                    accum *= intPower(allBetheRootsPtr[energyIndex*nUp + sigmaPtr[permIndex*nUp + i]], allConfigsPtr[positionIndex*nUp +i]);
                }
                e2cVector[positionIndex] += amplitude * accum;
                c2eVector[positionIndex] += GPUComplexType(1) / ( gaudinDet * betheProduct * amplitude * accum );
            }
        }
        for (unsigned long i = 0; i < 1716; i++) {
            e2cPtr[energyIndex*nBasis + i] = e2cVector[i];
            c2ePtr[i*nBasis + energyIndex] = c2eVector[i];
        }
    };

    auto computeBasisVectorMemOpt1 = [=] __device__ (int energyIndex) { // 51 seconds for L=13 N=6
        auto gaudinDet = allGaudinDetsPtr[energyIndex];
        auto betheProduct = gpuProduct(&allBetheRootsPtr[energyIndex*nUp], nUp); // Index raveling

        for (unsigned long positionIndex = 0; positionIndex < nBasis; positionIndex++) {
            GPUComplexType e2cTerm(0), c2eTerm(0);

            for (unsigned long permIndex = 0; permIndex < nPerm; permIndex++) {
                // There's enormous duplication of effort here as we're recomputing these amplitudes for every positionIndex even though we don't have to
                auto amplitude = gpuAmplitude(&allBetheRootsPtr[energyIndex*nUp], &sigmaPtr[permIndex*nUp], nUp, delta);
                // std::cout << "EnergyIndex: " << energyIndex << " permIndex: " << permIndex << " amplitude= " << amplitude << std::endl;

                GPUComplexType accum(1);
                for (unsigned long i = 0; i < nUp; i++) {
                    accum *= intPower(allBetheRootsPtr[energyIndex*nUp + sigmaPtr[permIndex*nUp + i]], allConfigsPtr[positionIndex*nUp +i]);
                }
                e2cTerm += amplitude * accum;
                c2eTerm += GPUComplexType(1) / ( gaudinDet * betheProduct * amplitude * accum );
            }
            e2cPtr[energyIndex*nBasis + positionIndex] = e2cTerm;
            c2ePtr[positionIndex*nBasis + energyIndex] = c2eTerm;
        }
    };

    // Computational complexity O( N! * (L choose N) * N * Log(N))
    auto computeBasisVector = [=] __device__ (int energyIndex) { // 19 seconds for L=13 N=6, 570 seconds for L=15 N=7, 197 s for L=17, N=6

        auto gaudinDet = allGaudinDetsPtr[energyIndex];
        auto betheProduct = gpuProduct(&allBetheRootsPtr[energyIndex*nUp], nUp);
        // auto denom = gpuDenominator(&allBetheRootsPtr[energyIndex*nUp], nUp, delta);

        for (unsigned long permIndex = 0; permIndex < nPerm; permIndex++) { // O(N!) // Next step is to parallelize across 
            // auto amplitude = gpuNumerator(&allBetheRootsPtr[energyIndex*nUp], &sigmaPtr[permIndex*nUp], nUp, delta) / denom;
            auto amplitude = gpuAmplitude(&allBetheRootsPtr[energyIndex*nUp], &sigmaPtr[permIndex*nUp], nUp, delta); // O(N * (N-1))
            // std::cout << "EnergyIndex: " << energyIndex << " permIndex: " << permIndex << " amplitude= " << amplitude << std::endl;

            for (unsigned long positionIndex = 0; positionIndex < nBasis; positionIndex++) { // O(L choose N)
                GPUComplexType accum(1);
                for (unsigned long i = 0; i < nUp; i++) { // O(N)
                    // An alternative is to use log and exp but those operations are much slower (~10x slowdown for L=13, N=6)
                    accum *= intPower(allBetheRootsPtr[energyIndex*nUp + sigmaPtr[permIndex*nUp + i]], allConfigsPtr[positionIndex*nUp + i]); // O(Log(N))
                }

                e2cPtr[energyIndex*nBasis + positionIndex] += amplitude * accum;
                c2ePtr[positionIndex*nBasis + energyIndex] += GPUComplexType(1) / ( gaudinDet * betheProduct * amplitude * accum );
            }
        }
    };

    // Call the lambda function using a parallel for all
    thrust::for_each(r, r + nBasis, computeBasisVector);

	thrust::host_vector<GPUComplexType> h_e2c(d_e2c);
    thrust::host_vector<GPUComplexType> h_c2e(d_c2e);

    MyComplexVector e2c(h_e2c.begin(), h_e2c.end());
    MyComplexVector c2e(h_c2e.begin(), h_c2e.end());
    return {e2c, c2e};
}

MyComplexVector gpuComputeBasisTransformDegen(
        const MyComplexVector allBetheRoots, // 2d vector flattened into a nIndep * nUp vector
        const MyLongVector allConfigs, // 2d vector flattened into a nBasis * nUp vector
        const MyLongVector sigma, // 2d vector flattened into a nPermutations * nUp vector
        const MyComplexType delta,
        const MyLongType nUp
) {
    auto nBasis = allConfigs.size()/nUp;
    auto nIndep = allBetheRoots.size()/nUp;
    auto nPerm = sigma.size()/nUp;

    //Store the output data here
    thrust::device_vector<GPUComplexType> d_e2c(nIndep*nBasis, MyComplexType(0));

    //Turn everything into device vectors
    thrust::device_vector<GPUComplexType> d_allBetheRoots(allBetheRoots.begin(), allBetheRoots.end());
    thrust::device_vector<MyLongType> d_allConfigs(allConfigs.begin(), allConfigs.end());
    thrust::device_vector<MyLongType> d_sigma(sigma.begin(), sigma.end());

    // Cast everything to raw pointers
    GPUComplexType* e2cPtr = thrust::raw_pointer_cast(&d_e2c[0]);

    const GPUComplexType* allBetheRootsPtr = thrust::raw_pointer_cast(&d_allBetheRoots[0]);
    const MyLongType* allConfigsPtr = thrust::raw_pointer_cast(&d_allConfigs[0]);
    const MyLongType* sigmaPtr = thrust::raw_pointer_cast(&d_sigma[0]);

    // Create the lambda function to do the actual calculation

    auto r = thrust::counting_iterator<long>(0);

    // Computational complexity O( N! * (L choose N) * N * Log(N))
    auto computeBasisVector = [=] __device__ (int energyIndex) { // 19 seconds for L=13 N=6, 570 seconds for L=15 N=7, 197 s for L=17, N=6

        auto betheProduct = gpuProduct(&allBetheRootsPtr[energyIndex*nUp], nUp);

        for (unsigned long permIndex = 0; permIndex < nPerm; permIndex++) { // O(N!) // Next step is to parallelize across 
            auto amplitude = gpuAmplitude(&allBetheRootsPtr[energyIndex*nUp], &sigmaPtr[permIndex*nUp], nUp, delta); // O(N * (N-1))
            // std::cout << "EnergyIndex: " << energyIndex << " permIndex: " << permIndex << " amplitude= " << amplitude << std::endl;

            for (unsigned long positionIndex = 0; positionIndex < nBasis; positionIndex++) { // O(L choose N)
                GPUComplexType accum(1);
                for (unsigned long i = 0; i < nUp; i++) { // O(N)
                    // An alternative is to use log and exp but those operations are much slower (~10x slowdown for L=13, N=6)
                    accum *= intPower(allBetheRootsPtr[energyIndex*nUp + sigmaPtr[permIndex*nUp + i]], allConfigsPtr[positionIndex*nUp + i]); // O(Log(N))
                }

                e2cPtr[energyIndex*nBasis + positionIndex] += amplitude * accum;
                // e2cPtr[energyIndex*nBasis + positionIndex] = energyIndex*nBasis + positionIndex;
            }
        }
    };

    // Call the lambda function using a parallel for all
    thrust::for_each(r, r + nIndep, computeBasisVector);

	thrust::host_vector<GPUComplexType> h_e2c(d_e2c);

    MyComplexVector e2c(h_e2c.begin(), h_e2c.end());
    return e2c;
}

std::pair<MyComplexVector, MyComplexVector> gpuComputeBasisTransformGranular(
        const MyComplexVector allBetheRoots, // 2d vector flattened into a nBasis * nUp vector
        const MyLongVector allConfigs, // 2d vector flattened into a nBasis * nUp vector
        const MyComplexVector allGaudinDets, // 1d vector nBasis
        const MyLongVector sigma, // 2d vector flattened into a nPermutations * nUp vector
        const MyComplexType delta
) {

    // Algorithm:
    // Kernel1
    // Each core owns a pair of [energyIndex, permIndex]
    // Compute the amplitude A[energyIndex, permIndex]
    // Kernel2
    // Each core owns a pair of [energyIndex, positionIndex]
    // Compute e2c, c2e

    auto nBasis = allGaudinDets.size();
    auto nUp = allBetheRoots.size()/nBasis;
    auto nPerm = sigma.size()/nUp;

    //Store the output data here
    thrust::device_vector<GPUComplexType> d_e2c(nBasis*nBasis, MyComplexType(0)), d_c2e(nBasis*nBasis, MyComplexType(0));

    //Turn everything into device vectors
    thrust::device_vector<GPUComplexType> d_allBetheRoots(allBetheRoots.begin(), allBetheRoots.end());
    thrust::device_vector<MyLongType> d_allConfigs(allConfigs.begin(), allConfigs.end());
    thrust::device_vector<GPUComplexType> d_allGaudinDets(allGaudinDets.begin(), allGaudinDets.end());
    thrust::device_vector<MyLongType> d_sigma(sigma.begin(), sigma.end());

    // Cast everything to raw pointers
    GPUComplexType* e2cPtr = thrust::raw_pointer_cast(&d_e2c[0]);
    GPUComplexType* c2ePtr = thrust::raw_pointer_cast(&d_c2e[0]);

    const GPUComplexType* allBetheRootsPtr = thrust::raw_pointer_cast(&d_allBetheRoots[0]);
    const MyLongType* allConfigsPtr = thrust::raw_pointer_cast(&d_allConfigs[0]);
    const GPUComplexType* allGaudinDetsPtr = thrust::raw_pointer_cast(&d_allGaudinDets[0]);
    const MyLongType* sigmaPtr = thrust::raw_pointer_cast(&d_sigma[0]);

    // Create the lambda function to do the actual calculation

    auto r = thrust::counting_iterator<long>(0);

    // Computational complexity O( N! * (L choose N) * N * Log(N))
    auto computeBasisVector = [=] __device__ (int energyIndex) { // 19 seconds for L=13 N=6, 570 seconds for L=15 N=7, 197 s for L=17, N=6

        auto gaudinDet = allGaudinDetsPtr[energyIndex];
        auto betheProduct = gpuProduct(&allBetheRootsPtr[energyIndex*nUp], nUp);
        // auto denom = gpuDenominator(&allBetheRootsPtr[energyIndex*nUp], nUp, delta);

        for (unsigned long permIndex = 0; permIndex < nPerm; permIndex++) { // O(N!) // Next step is to parallelize across 
            // auto amplitude = gpuNumerator(&allBetheRootsPtr[energyIndex*nUp], &sigmaPtr[permIndex*nUp], nUp, delta) / denom;
            auto amplitude = gpuAmplitude(&allBetheRootsPtr[energyIndex*nUp], &sigmaPtr[permIndex*nUp], nUp, delta); // O(N * (N-1))
            // std::cout << "EnergyIndex: " << energyIndex << " permIndex: " << permIndex << " amplitude= " << amplitude << std::endl;

            for (unsigned long positionIndex = 0; positionIndex < nBasis; positionIndex++) { // O(L choose N)
                GPUComplexType accum(1);
                for (unsigned long i = 0; i < nUp; i++) { // O(N)
                    // An alternative is to use log and exp but those operations are much slower (~10x slowdown for L=13, N=6)
                    accum *= intPower(allBetheRootsPtr[energyIndex*nUp + sigmaPtr[permIndex*nUp + i]], allConfigsPtr[positionIndex*nUp + i]); // O(Log(N))
                }

                e2cPtr[energyIndex*nBasis + positionIndex] += amplitude * accum;
                c2ePtr[positionIndex*nBasis + energyIndex] += GPUComplexType(1) / ( gaudinDet * betheProduct * amplitude * accum );
            }
        }
    };

    // Call the lambda function using a parallel for all
    thrust::for_each(r, r + nBasis, computeBasisVector);

	thrust::host_vector<GPUComplexType> h_e2c(d_e2c);
    thrust::host_vector<GPUComplexType> h_c2e(d_c2e);

    MyComplexVector e2c(h_e2c.begin(), h_e2c.end());
    MyComplexVector c2e(h_c2e.begin(), h_c2e.end());
    return {e2c, c2e};
}
