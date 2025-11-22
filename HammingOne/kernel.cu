#include "file_io.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include "HammingOne.cu"
#include "CpuHammingOne.h"

#define NUM_STR 100000
#define STR_LEN 35

int main()
{
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::microseconds duration;
    int longsPerRow = 0;
    //generateBinaryString("sample.txt", NUM_STR, STR_LEN);
    long long** matrix = file_read("sample.txt", &longsPerRow, NUM_STR, STR_LEN);
    start = std::chrono::high_resolution_clock::now();
    cpu_hamming_one(matrix, NUM_STR, longsPerRow);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("CPU total time: %lf s\n", (double)duration.count() / 1000000);
    std::cout << "---GPU:---" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    cudaComputeRowHashes(matrix, NUM_STR, longsPerRow);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("GPU total time: %lf s\n", (double)duration.count() / 1000000);
}