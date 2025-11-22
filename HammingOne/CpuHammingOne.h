#pragma once
#include <stdlib.h>
#include <stdio.h>

void cpu_hamming_one(long long** matrix,
					   int numStrings,
					   int numLongsPerRow)
{
	for (int i = 0; i < numStrings; i++)
	{
		for (int j = i + 1; j < numStrings; j++)
		{
			int counter = 0, dist = 0;
			long long diff;
			while (dist <= 1 && counter < numLongsPerRow)
			{
				diff = matrix[i][counter] ^ matrix[j][counter];
				if (((diff - 1) & diff) == 0)
				{
					dist++;
				}
				else
				{
					dist = 2;
					//break;
				}
				counter++;
			}
			if (dist <= 1 && counter == numLongsPerRow)
			{
				printf("%d %d\n", i, j);
			}
			//printf("%d %d %d\n", i, j, dist);
		}
	}
}