/**
 * @Author: Eric Corwin <ecorwin>
 * @Email:  eric.corwin@gmail.com
 * @Filename: spinChain.cpp
 */

// #include "gpuControl.h"
#include "goldCode.h"
#include "gpuCode.cuh"
#include <functional>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


PYBIND11_MODULE(libspinChain, m) {
    m.doc() = "libspinChain python interface to CUDA code for computing spin chain information.";
    
    //Select the gpu to use, must be done before the class is instantiated
	// m.def("setDeviceNumber", &setDeviceNumber, "Select the active gpu device, must happen before calculations are performed");
	// m.def("getDeviceNumber", &getDeviceNumber, "Return the active gpu device");

    //Function to compute the Bethe roots
    // m.def("computeBetheRoots", &computeBetheRoots, "Compute all of the bethe roots");
    // m.def("computeBasisTransformMatrices", &computeBasisTransformMatrices, "Compute the forward and backwards basis transform matrices");
    m.def("sum", &sum);
    m.def("vectorSum", &vectorSum);
    m.def("inspectArray", &inspectArray);
    m.def("arraySumSq", &arraySumSq);
    m.def("intPower", &intPower);
    m.def("product", &product);
    m.def("computeAmplitude", &computeAmplitude);
    m.def("computeBasisTransform", &computeBasisTransform);
    m.def("gpuSum", &gpuSum);
    m.def("thrustListTest", &thrustListTest);
    m.def("gpuComputeBasisTransform", &gpuComputeBasisTransform);
    m.def("gpuComputeBasisTransformDegen", &gpuComputeBasisTransformDegen);
}
