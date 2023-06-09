#include "device_launch_parameters.h"
#include <helper_cuda.h>

struct Points
{
	float x, y, z;
	float r, g, b;
};

__global__ void processCUDA(Points* cudaData, size_t numElements, int frame_number)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < numElements){
		frame_number = frame_number % 100;
		cudaData[idx].x = cudaData[idx].x + frame_number * 0.000002;
	}
}

extern "C" void launch_cudaProcess(int grid, int block, Points * cudaData, size_t numBytes, int frame_number){
	processCUDA <<<grid, block >>> (cudaData, numBytes, frame_number);
	cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("153::CUDA Error: %s\n", cudaGetErrorString(err));
            // exit(1);
        } 
}
