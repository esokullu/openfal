cmake_minimum_required(VERSION 3.22)
project(openfal_strata_dma LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(OPENFAL_ENABLE_CUDA "Build CUDA async ring-buffer DMA and FP8/INT4 dequant kernels" ON)

if(OPENFAL_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(strata_dma SHARED
        csrc/strata_ops.cpp
        csrc/strata_dma.cu
    )
    target_include_directories(strata_dma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(strata_dma PRIVATE CUDA::cudart)
    set_target_properties(strata_dma PROPERTIES
        CUDA_ARCHITECTURES "86;89;90;120" # Ampere, Ada (4090), Hopper, Blackwell (5060 Ti / 5090)
    )
else()
    add_library(strata_dma SHARED
        csrc/strata_ops.cpp
    )
    target_compile_definitions(strata_dma PRIVATE OPENFAL_CPU_ONLY=1)
    target_include_directories(strata_dma PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
endif()
