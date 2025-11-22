#include "HammingOne.cuh"

#define CUDA_ERR(message) {if(message != cudaSuccess) { fprintf(stderr, "Error has occured: %ll", message); goto Error;} }
#define PRIME_BASE 2
#define PRIME_MOD 1000000000000037 // 1e15 + 37

__global__ void computeHashesKernel(
    long long* device_matrix,
    int stringNumber,
    int longsPerRow,
    long long* deviceHashes,
    KeyValueTuple* deviceKeyValueTuples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < stringNumber) {
        
        long long subHash = 0;
        long long incrementalBase = PRIME_BASE;
        int change;
        for (int j = 0; j < longsPerRow; j++) {
            long long value = device_matrix[j * stringNumber + idx];
            for(int i = 0; i < 64; i++) {
                change = (value & (1LL << 63)) == (1LL << 63);
                subHash = (subHash + change  * incrementalBase) % PRIME_MOD;
                incrementalBase = (incrementalBase * PRIME_BASE) % PRIME_MOD;
                value <<= 1;
            }
            
        }
        deviceHashes[idx] = subHash;
        KeyValueTuple temp;
        temp.hash = subHash;
        temp.index = idx;
        deviceKeyValueTuples[idx] = temp;
        //printf("%d. %lld\n", idx, subHash);
    }
}

__device__ int hashBinSearch(long long valueToLookFor,
    KeyValueTuple* deviceKeyValueTuple, 
    int numStrings)
{
    int l = 0, r = numStrings - 1;
    int mid;
    while (l <= r)
    {
        mid = l + ((r - l) >> 1);
        if (deviceKeyValueTuple[mid].hash == valueToLookFor)
        {
            return deviceKeyValueTuple[mid].index;
        }
        if (deviceKeyValueTuple[mid].hash < valueToLookFor)
        {
            l = mid + 1;
        }
        else
        {
            r = mid - 1;
        }
    }
    return -1;
};

__global__ void findPair(
    long long* device_matrix,
    int stringNumber,
    int longsPerRow,
    long long* deviceHashes,
    KeyValueTuple* deviceKeyValueTuples)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < stringNumber)
    {
        long long newHash = 0;
        long long int idxHash = deviceHashes[idx];
        long long incrementalBase = PRIME_BASE;
        long long value;
        int bit_change, pair;

        int iteration = 0;
        for (int i = 0; i < longsPerRow; i++)
        {
            value = device_matrix[i * stringNumber + idx];
            for(int j = 0; j < 64; j++)
            {
                if ((1LL << 63) & value)
                {
                    bit_change = -1;
                }
                else
                {
                    bit_change = 1;
                }
                
                newHash = (idxHash + bit_change * incrementalBase + PRIME_MOD) % PRIME_MOD;
                if ((pair = hashBinSearch(newHash, deviceKeyValueTuples, stringNumber)) != -1)
                {
                    printf("%d %d\n", pair, idx);
                }
                
                value <<= 1;
                incrementalBase = (incrementalBase * PRIME_BASE) % PRIME_MOD;
                iteration++;
            }
        }
    }
}

void cudaComputeRowHashes(
    long long** matrix,
    int rows,
    int longsPerRow
) {
    long long* transposed_matrix;
    long long* device_transposed_matrix;
    thrust::device_vector<long long> device_hashes(rows);
    thrust::device_vector<KeyValueTuple> device_key_value_tuples(rows);
    cudaError_t error;

    long transposed_matrix_size = rows * longsPerRow;
    transposed_matrix = (long long*)malloc(transposed_matrix_size * sizeof(long long));
    for (int i = 0; i < longsPerRow; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            transposed_matrix[i * rows + j] = matrix[j][i];
        }
    }
    
    error = cudaMalloc(&device_transposed_matrix, transposed_matrix_size * sizeof(long long));
    CUDA_ERR(error);
    error = cudaMemcpy(device_transposed_matrix, transposed_matrix, transposed_matrix_size * sizeof(long long), cudaMemcpyHostToDevice);
    CUDA_ERR(error);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    computeHashesKernel <<<blocksPerGrid, threadsPerBlock >>> (device_transposed_matrix, rows, longsPerRow, thrust::raw_pointer_cast(device_hashes.data()), thrust::raw_pointer_cast(device_key_value_tuples.data()));
    error = cudaDeviceSynchronize();


    thrust::sort(device_key_value_tuples.begin(), device_key_value_tuples.end());

    findPair << <blocksPerGrid, threadsPerBlock >> > (device_transposed_matrix, rows, longsPerRow, thrust::raw_pointer_cast(device_hashes.data()), thrust::raw_pointer_cast(device_key_value_tuples.data()));
    error = cudaDeviceSynchronize();
    CUDA_ERR(error);

Error:
    free(transposed_matrix);
    cudaFree(device_transposed_matrix);
    return;
}