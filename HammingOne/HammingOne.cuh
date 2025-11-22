#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include "thrust/sort.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"


#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)calloc(xDim * yDim,  sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)


struct KeyValueTuple {
    long long hash;
    int index;

    __device__ __host__ bool operator<(const KeyValueTuple& b) const
    {
        return hash < b.hash;
    }
} ;

inline __global__ void computeHashesKernel(
    long long* device_matrix,
    int stringNumber,
    int longsPerRow,
    long long* deviceHashes,
    KeyValueTuple * deviceKeyValueTuples
);

inline __global__ void findPair(
    long long* device_matrix,
    int stringNumber,
    int longsPerRow,
    long long* deviceHashes,
    KeyValueTuple* deviceKeyValueTuples);

inline void cudaComputeRowHashes(
    long long** matrix,
    int rows,
    int longsPerRow
);

inline __device__ int hashBinSearch(long long valueToLookFor,
    KeyValueTuple* deviceKeyValueTuple,
    int numStrings);