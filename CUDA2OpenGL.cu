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
	
	// if (idx == 1) {
	// 	printf("sizeof(cudaData) = %lu, numE = %lu, frame = %d\n", sizeof(cudaData), numElements, frame_number);
	// 	// printf(" p = %.8f %.8f %.8f\n", cudaData[idx].x, cudaData[idx].y, cudaData[idx].z);
	// 	// printf("CUDA hello\n");
	// 	// printf("\t %d:cudaData p = %.8f %.8f %.8f c = %.2f %.2f %.2f\n", frame_number, cudaData[idx].x, cudaData[idx].y, cudaData[idx].z, cudaData[idx].r, cudaData[idx].g, cudaData[idx].b);
	// }

	if(idx == 0){
		printf("idx = %d, numElements = %lu\n",idx, numElements);
	}

	if (idx < numElements)
	{
		cudaData[idx].x = cudaData[idx].x + frame_number * 0.0002;
		if(idx == 1)
			printf("changeData\n");
			// printf("cudaData[0] = %d\n", cudaData[idx].x);
	}
}

extern "C" void launch_cudaProcess(int grid, int block, Points * cudaData, size_t numBytes, int frame_number){
	printf("hello, frame_number = %d numBytes = %lu\n", frame_number, numBytes);
	processCUDA <<<grid, block >>> (cudaData, numBytes, frame_number);
}
