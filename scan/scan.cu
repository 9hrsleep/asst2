#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"


extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2.
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void exclusive_scan_upsweep_kernel(int* device_data, int twod, int twod1, int treeLength) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int start_index = thread_index * twod1;
    int end_index = start_index + twod1 - 1;
    if (end_index < treeLength) // check end bounds of block
       device_data[start_index+twod1-1] += device_data[start_index+twod-1];
}

__global__ void exclusive_scan_downsweep_kernel(int* device_data, int twod, int twod1, int treeLength) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    int start_index = thread_index * twod1;
    int end_index = start_index + twod1 - 1;
    if (end_index < treeLength) {
        int t = device_data[start_index+twod-1];
        device_data[start_index+twod-1] = device_data[start_index+twod1-1];
        device_data[start_index+twod1-1] += t;
    }
}

void exclusive_scan(int* device_data, int length)
{
    /* TODO
     * Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the data in device memory
     * The data are initialized to the inputs.  Your code should
     * do an in-place scan, generating the results in the same array.
     * This is host code -- you will need to declare one or more CUDA
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the data array is sized to accommodate the next
     * power of 2 larger than the input.
     */
    
    // our implementation with cuda
    int threadsPerBlock = 512;
    int treeLength = nextPow2(length);

    // upsweep phase
    for (int twod = 1; twod < treeLength; twod *= 2){
        int twod1 = twod * 2;
        int numThreads = treeLength / twod1; // the chunks of the tree
        int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
        // launch the parallel kernels
        exclusive_scan_upsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, twod, twod1, treeLength);
    }

    int zero = 0;
    // write last elem of the array to be 0 
    cudaMemcpy(device_data + treeLength - 1, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    // downsweep phase
    for (int twod = treeLength / 2; twod >= 1; twod /= 2){
        int twod1 = twod * 2;
        int numThreads = treeLength / twod1; // the chunks of the tree
        int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;
        // launch the parallel kernels
        exclusive_scan_downsweep_kernel<<<blocks, threadsPerBlock>>>(device_data, twod, twod1, treeLength);
    }
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_data;
    // We round the array size up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness.
    // You may have an easier time in your implementation if you assume the
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_data, sizeof(int) * rounded_length);

    cudaMemcpy(device_data, inarray, (end - inarray) * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_data, end - inarray);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;

    cudaMemcpy(resultarray, device_data, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}


__global__ void flag_peaks_kernel(int* device_input, int *flags_of_peaks, int length, int treeLength) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    // since more blocks than needed treeLength spawn, don't do anything with extra threads
    if (thread_index >= treeLength) {
        return;
    }
    if (thread_index == 0 || thread_index >= length - 1){
        flags_of_peaks[thread_index] = 0;
        return;
    }
    flags_of_peaks[thread_index] = (device_input[thread_index] > device_input[thread_index-1] && 
                        device_input[thread_index] > device_input[thread_index+1]) ? 1 : 0;
}

__global__ void return_peak_indexes_kernel(int* device_output, int *flags_of_peaks, int length) {
    int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index == 0 || thread_index >= length - 1){
        return;
    }
    if(flags_of_peaks[thread_index+1] != flags_of_peaks[thread_index]){
        int peak_index = flags_of_peaks[thread_index];
        device_output[peak_index] = thread_index;
    }

}

int find_peaks(int *device_input, int length, int *device_output) {
    /* TODO:
     * Finds all elements in the list that are greater than the elements before and after,
     * storing the index of the element into device_result.
     * Returns the number of peak elements found.
     * By definition, neither element 0 nor element length-1 is a peak.
     *
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if
     * it requires that. However, you must ensure that the results of
     * find_peaks are correct given the original length.
     */

    int threadsPerBlock = 512;
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    int treeLength = nextPow2(length);

    // flag the peaks in new array with peak indexs with value 1     
    int* flags_of_peaks; 
    cudaMalloc(&flags_of_peaks, length*sizeof(int));
    // assign chunks of device_input to different threads to find peaks in parallel
    int flagBlocks = (treeLength + threadsPerBlock - 1) / threadsPerBlock;
    flag_peaks_kernel<<<flagBlocks, threadsPerBlock>>>(device_input, flags_of_peaks, length, treeLength);
    // then do exclusive scan on the flag array to get indices of peaks
    exclusive_scan(flags_of_peaks, length);

    // collect peaks into the output array
    return_peak_indexes_kernel<<<blocks, threadsPerBlock>>>(device_output, flags_of_peaks, length);
    int numPeaksFound = 0;
    // the highest peak found is put into the flags_of_peaks last index
    cudaMemcpy(&numPeaksFound, flags_of_peaks+length-1, sizeof(int), cudaMemcpyDeviceToHost);
    // wait for all threads to complete
    cudaDeviceSynchronize();
    cudaFree(flags_of_peaks);
    return numPeaksFound;
}



/* Timing wrapper around find_peaks. You should not modify this function.
 */
double cudaFindPeaks(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int),
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    int result = find_peaks(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}


void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
