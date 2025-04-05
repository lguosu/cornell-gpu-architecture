#include <stdio.h>

// Helper function to convert SM version to number of cores
int _ConvertSMVer2Cores(int major, int minor)
{
    // Defpth of each generation is only the major version # and each arch is the same or better than its predecessors
    // Only items which have been officially announced are included in this list
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    { { 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
      { 0x32, 192 }, // Kepler Generation (SM 3.2) GK11x class
      { 0x35, 192 }, // Kepler Generation (SM 3.5) GK21x class
      { 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
      { 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
      { 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
      { 0x53, 128 }, // Maxwell Generation (SM 5.3) GM20x class
      { 0x60,  64 }, // Pascal Generation (SM 6.0) GP100 class
      { 0x61, 128 }, // Pascal Generation (SM 6.1) GP10x class
      { 0x62, 128 }, // Pascal Generation (SM 6.2) GP10x class
      { 0x70,  64 }, // Volta Generation (SM 7.0) GV100 class
      { 0x72,  64 }, // Turing Generation (SM 7.2) TU102 class
      { 0x75,  64 }, // Turing Generation (SM 7.5) TU116 class
      { 0x80,  64 }, // Ampere Generation (SM 8.0) GA100 class
      { 0x86, 128 }, // Ampere Generation (SM 8.6) GA104 class
      { 0x89, 128 }, // Ada Lovelace Generation (SM 8.9) AD102 class
      {   -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}

// Print device properties
void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Maximum number of threads per SM: %d\n", devProp.maxThreadsPerMultiProcessor);

    // Add detailed core information
    printf("\nCore Information per SM:\n");
    printf("FP32 (CUDA) cores:            %d\n", _ConvertSMVer2Cores(devProp.major, devProp.minor));
    printf("FP16 cores:                   %d\n", _ConvertSMVer2Cores(devProp.major, devProp.minor) * 2);  // FP16 is typically 2x FP32
    printf("INT32 cores:                  %d\n", _ConvertSMVer2Cores(devProp.major, devProp.minor));      // Usually same as FP32
    
    // Tensor cores vary by architecture
    int tensorCoresPerSM = 0;
    if (devProp.major >= 7) {  // Volta and newer
        if (devProp.major == 7 && devProp.minor == 0) {  // Volta
            tensorCoresPerSM = 8;
        } else if (devProp.major == 7 && devProp.minor >= 2) {  // Turing
            tensorCoresPerSM = 8;
        } else if (devProp.major == 8) {  // Ampere
            tensorCoresPerSM = 4;
        } else if (devProp.major == 9) {  // Hopper
            tensorCoresPerSM = 4;
        }
    }
    printf("Tensor cores:                 %d\n", tensorCoresPerSM);
    
    printf("\nTotal Cores across all SMs:\n");
    printf("Total FP32 cores:             %d\n", _ConvertSMVer2Cores(devProp.major, devProp.minor) * devProp.multiProcessorCount);
    printf("Total FP16 cores:             %d\n", _ConvertSMVer2Cores(devProp.major, devProp.minor) * 2 * devProp.multiProcessorCount);
    printf("Total INT32 cores:            %d\n", _ConvertSMVer2Cores(devProp.major, devProp.minor) * devProp.multiProcessorCount);
    printf("Total Tensor cores:           %d\n", tensorCoresPerSM * devProp.multiProcessorCount);
    
    printf("\nKernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    return;
}
 
int main()
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);
 
    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
    return 0;
}
