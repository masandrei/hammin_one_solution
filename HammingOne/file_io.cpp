#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <random>

#include "file_io.h"

long long** file_read(
    const char* filename,
    int* outLongsPerRow,
    int maxStrings,
    int stringLength
) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file");
    }

    int longsPerRow = (stringLength + 63) / 64;
    long long** matrix = new long long* [maxStrings];
    for (int i = 0; i < maxStrings; ++i) {
        matrix[i] = new long long[longsPerRow]();
        std::memset(matrix[i], 0, longsPerRow * sizeof(long long));
    }

    std::string line;
    int rows = 0;

    while (file && rows < maxStrings && std::getline(file, line)) {

        for (int i = 0; i < stringLength; ++i) {
            int longLongIndex = i / 64;
            int bitPosition = i % 64;

            if (line[i] == '1') {
                matrix[rows][longLongIndex] |= (1LL << (63 - bitPosition));
            }
        }
        rows++;
    }

    *outLongsPerRow = longsPerRow;
    return matrix;
}

// Function to generate a random binary string of given length
std::string generateRandomBinaryString(int length, std::mt19937& rng) {
    std::string binaryString;
    std::uniform_int_distribution<int> dist(0, 1);

    for (int i = 0; i < length; ++i) {
        binaryString += std::to_string(dist(rng));
    }
    return binaryString;
}

// Main function to generate binary strings and write to a file
void generateBinaryString(const char* filename, int numStrings, int stringLength) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    // Seed for randomness
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int i = 0; i < numStrings; ++i) {
        std::string binaryString = generateRandomBinaryString(stringLength, rng);
        outputFile << binaryString << "\n";
    }

    outputFile.close();
    std::cout << "Successfully written " << numStrings << " binary strings to " << filename << ".\n";
}