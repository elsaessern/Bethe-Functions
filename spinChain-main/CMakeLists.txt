cmake_minimum_required(VERSION 3.10)
project(libspinChain LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CUDA_ARCHITECTURES 31)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin='gcc-7'")

# set(pybind11_DIR "$(python3 -m pybind11 --cmakedir)")
set(pybind11_DIR "/home/ecorwin/Documents/Code/spinChain/sp-env/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
set(pybind11_INCLUDE_DIR "/home/ecorwin/Documents/Code/spinChain/sp-env/lib/python3.12/site-packages/pybind11/include")

# Find pybind11 and CUDA
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)
# find_package(CUDAToolkit)
# find_package(Thrust REQUIRED CONFIG)
# thrust_create_target(ThrustCUDA)
# target_link_libraries(sort-with-cuda ThrustCUDA)
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G -lineinfo --extended-lambda")

# enable device lambda support
# list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")

# Include directories for pybind11 and CUDA
include_directories(${PYBIND11_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

# Define the Python extension module
pybind11_add_module(libspinChain src/pybindWrapper.cpp src/goldCode.cpp src/gpuCode.cu)
target_link_libraries(libspinChain PUBLIC ${CUDA_LIBRARIES} ${pybind11_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})

target_compile_options(libspinChain PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                    #    --relocatable-device-code=true
                       --extended-lambda
                       >)

# add_library(librt STATIC IMPORTED)
# set_target_properties(librt PROPERTIES
#                         IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/temp/librt.a"
# )

# Ensure the CUDA architecture is set
set_target_properties(libspinChain PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
