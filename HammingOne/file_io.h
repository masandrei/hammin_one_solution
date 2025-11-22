#pragma once
#include <assert.h>



long long** file_read(const char* filename, int* outLongsPerRow, int maxStrings, int stringLength);

void generateBinaryString(const char* filename, int numStrings, int stringLength);