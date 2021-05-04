

#ifndef CUDA_LIB_H
#define CUDA_LIB_H

#include "cpuLib.h"

#include <cuda.h>
#include <curand_kernel.h>


void AlexNet(int batchSize);

float * makeVector ( uint64_t size);
float * makeTensorBatched (TensorShape shape, int batchSize);
inline float * mallocTensorBatched (TensorShape shape, int batchSize);

__global__ void poolLayer_gpu (float * input, TensorShape inShape, float * output, TensorShape outShape, PoolLayerArgs args, float min_float);

__global__ void convLayer_gpu ( float * in, TensorShape iShape, float * filter, TensorShape fShape, float * bias, float * out, TensorShape oShape, ConvLayerArgs args, uint32_t batchSize);

__global__ void medianFilter_gpu (uint8_t * inPixels, ImageDim imgDim, uint8_t * outPixels, MedianFilterArgs args);

__global__ void gemmLayer_gpu(float* A1, float* B1, float* C1, TensorShape aShape, TensorShape bShape, TensorShape cShape) ;

#endif
