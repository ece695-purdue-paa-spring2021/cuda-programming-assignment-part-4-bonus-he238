
#include "cudaLib.cuh"
#include <cstdlib>

__global__ void convLayer_gpu ( float * in, TensorShape iShape, float * filter, TensorShape fShape, float * bias, float * out, TensorShape oShape, ConvLayerArgs args, uint32_t batchSize){
	uint32_t n = blockIdx.z / oShape.channels;
	uint32_t m = blockIdx.z % oShape.channels;
	
	uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	

	if (x >=  oShape.height || y >= oShape.width){
		return;
	}
	
	float result = bias[m];
	for (uint32_t i = 0; i < fShape.height; ++ i) {
		for (uint32_t j = 0; j < fShape.width; ++ j) {
			for (uint32_t k = 0; k < fShape.channels; ++ k) {
				if (args.strideH*x >= iShape.height || args.strideW*y >= iShape.width){
					
				} else {
				 	result += filter[((m*fShape.channels+k)*fShape.height+i)*fShape.width+j] *in[((n*iShape.channels+k)*iShape.height+args.strideH*x)*iShape.width+args.strideW*y];				
				}	
			}
		}
	}
	out[((n*oShape.channels+m)*oShape.height+x)*oShape.width+y] = result;
	if (args.activation) {
		result = (result > 0)? result:0;
	}
}

__global__ void gemmLayer_gpu(float* A1, float* B1, float* C1, TensorShape aShape, TensorShape bShape, TensorShape cShape) {

    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int n = blockIdx.z;
    float tmpSum = 0;
	float * A = &A1[n * aShape.height * aShape.width];
	float * B = &B1[n * bShape.height * bShape.width];
	float * C = &C1[n * cShape.height * cShape.width];
    if (y < cShape.width && x < cShape.height) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < aShape.width; i++) {
            tmpSum += A[x * aShape.width + i] * B[i * bShape.width + y];
        }
        C[x * cShape.width + y] = tmpSum;
    }	
}

__global__ void poolLayer_gpu (float * input, TensorShape inShape, float * output, TensorShape outShape, PoolLayerArgs args, float min_float) {
	float poolPick;
	uint32_t outRow = blockIdx.x*blockDim.x + threadIdx.x;
	uint32_t outCol = blockIdx.y*blockDim.y + threadIdx.y;
	uint32_t channel = blockIdx.z;
	if (outRow >= outShape.height || outCol >= outShape.width){
		return;
	}
	int inRow, inCol;
	poolPick = min_float;
	for (uint32_t poolRow = 0; poolRow < args.poolH; ++ poolRow) {
		for (uint32_t poolCol = 0; poolCol < args.poolW; ++ poolCol) {
			inRow = outRow *  args.strideH - (args.poolH - 1) / 2 + poolRow;
			inCol = outCol * args.strideW - (args.poolW - 1) / 2 + poolCol;
			if (inRow < 0 || inCol < 0 || inRow >= inShape.height|| inCol >= inShape.width){
			} else {
				//poolPick = (input[(inRow * inShape.width + inCol) * inShape.channels +  channel]<poolPick)?poolPick:input[(inRow * inShape.width + inCol) * inShape.channels +  channel];
				poolPick = (input[(channel*inShape.height + inCol)*inShape.width+inRow]<poolPick)?poolPick:input[(channel*inShape.height + inCol)*inShape.width+inRow];	
						
			}
		}
	}
	output[(channel*outShape.height + outCol)*outShape.width+outRow] = poolPick;
	return;
}

float * mallocTensorBatched (TensorShape shape, int batchSize) {
	if (shape.count == 0) {
		std::cout << " Shape has invalid count (4th dim) - setting to 1 \n";
		shape.count = 1;
	}
	float * d_t;
	cudaMalloc((void**) &d_t, sizeof(float) * tensorSize(shape) * batchSize);
	return d_t;
}
float * makeTensorBatched (TensorShape shape, int batchSize) {
	float * t = (float *) malloc (tensorSize(shape) * sizeof(float) * batchSize);
	if (shape.count == 0) {
		std::cout << " Shape has invalid count (4th dim) - setting to 1 \n";
		shape.count = 1;
	}

	if (t == nullptr) {
		std::cout << "Malloc failed ! \n";
		return nullptr;
	}

	float * m = t;
	uint64_t offset;

	std::random_device random_device;
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	//	Implement NCHW layout
	for (uint32_t count = 0; count < shape.count * batchSize; ++ count) {
		for (uint32_t chIdx = 0; chIdx < shape.channels; ++ chIdx ) {
			for (uint32_t rowIdx = 0; rowIdx < shape.height; ++ rowIdx) {
				for (uint32_t colIdx = 0; colIdx < shape.width; ++ colIdx) {
					offset = chIdx * shape.height * shape.width + rowIdx * shape.width + colIdx;
					m[offset] = dist(random_device);
				}
			}
		}
	}
	float * d_t;
	cudaMalloc((void**) &d_t, sizeof(float) * tensorSize(shape) * batchSize);
	cudaMemcpy(d_t, t, sizeof(float) * tensorSize(shape) * batchSize, cudaMemcpyHostToDevice);
	free(t);
	return d_t;
}

float * makeVector ( uint64_t size) {
	float * v;
	v = (float *) malloc (size * sizeof(float));
	float * m = v;
	float * d_v;
	std::random_device random_device;
	std::uniform_real_distribution<float> dist(0.0, 1.0);

	//	Implement NCHW layout
	for (uint64_t idx = 0; idx < size; ++ idx) {
		m[idx] = dist(random_device);
	}
	cudaMalloc((void**) &d_v, sizeof(float) * size);
	cudaMemcpy(d_v, v, sizeof(float) * size, cudaMemcpyHostToDevice);
	free(v);
	return d_v;
}
float * mallocVector ( uint64_t size, int batchSize) {
	float * d_v;
	cudaMalloc((void**) &d_v, sizeof(float) * size * batchSize);
	return d_v;
}
void convLayer (float * d_in, TensorShape iShape, float * d_filter, TensorShape fShape, float * d_bias, float * d_out, TensorShape oShape, ConvLayerArgs args, uint32_t batchSize){
 	dim3 blockSize(16, 16);
	dim3 gridSize(oShape.height / 16 + 1, oShape.width / 16 + 1, oShape.channels *  batchSize);		
	convLayer_gpu<<<gridSize, blockSize>>>(d_in, iShape, d_filter, fShape, d_bias, d_out, oShape, args, batchSize);
	return;
}


void poolLayer (float * d_in, TensorShape iShape, float * d_out, TensorShape oShape, PoolLayerArgs_t args, uint32_t batchSize){
 	dim3 blockSize(16, 16);
	dim3 gridSize(oShape.height / 16 + 1, oShape.width / 16 + 1, oShape.channels *  batchSize);	
	TensorShape outShape = {1, iShape.channels, iShape.height / args.strideH, iShape.width / args.strideH};	
	if (outShape.height != oShape.height || outShape.width != oShape.width || outShape.channels != oShape.channels){
		std::cout << oShape << " oShape does not match! " << outShape <<std::endl;
		return;
	}
	poolLayer_gpu<<<gridSize, blockSize>>>(d_in, iShape, d_out, oShape, args, std::numeric_limits<float>::min());
	return;
}

void FCLayer (float* d_a, float* d_b, float* d_c, TensorShape aShape, TensorShape bShape, TensorShape cShape, int batchSize){
	const dim3 blockSize(16, 16);
	const dim3 gridSize(cShape.height / 16 + 1, cShape.width / 16 + 1, 1);
	aShape.height = batchSize * aShape.height;
	cShape.height = batchSize * cShape.height;
	gemmLayer_gpu<<<gridSize, blockSize>>>(d_a, d_b, d_c, aShape, bShape, cShape);

}
void AlexNet(int batchSize) {
	float ** d_filter = new float*[11];
	float ** d_in = new float*[11];
	float ** d_bias = new float*[11];
	for (int i = 0; i < 11; i++){
		d_filter[i] = nullptr;
		d_in[i] = nullptr;
		d_bias[i] = nullptr;
	}
	//init buffers
	//d_in[0] = makeTensorBatched(AlexL1_InShape, batchSize);
	d_in[1] = mallocTensorBatched(AlexL2_InShape, batchSize);
	d_in[2] = mallocTensorBatched(AlexL3_InShape, batchSize);
	d_in[3] = mallocTensorBatched(AlexL4_InShape, batchSize);
	d_in[4] = mallocTensorBatched(AlexL5_InShape, batchSize);
	d_in[5] = mallocTensorBatched(AlexL6_InShape, batchSize);
	d_in[6] = mallocTensorBatched(AlexL7_InShape, batchSize);
	d_in[7] = mallocTensorBatched(AlexL8_InShape, batchSize);
	d_in[8] = mallocTensorBatched(AlexL9_InShape, batchSize);
	d_in[9] = mallocTensorBatched(AlexL10_InShape, batchSize);
	d_in[10] = mallocTensorBatched(AlexL10_InShape, batchSize);

	
	d_filter[0] = makeTensorBatched(AlexL1_FilterShape, 1);
	d_filter[2] = makeTensorBatched(AlexL3_FilterShape, 1);
	d_filter[4] = makeTensorBatched(AlexL5_FilterShape, 1);
	d_filter[5] = makeTensorBatched(AlexL6_FilterShape, 1);
	d_filter[6] = makeTensorBatched(AlexL7_FilterShape, 1);
	d_filter[8] = makeTensorBatched(AlexL9_bShape, 1);
	d_filter[9] = makeTensorBatched(AlexL10_bShape, 1);

	d_bias[0] = makeVector(AlexL1_FilterShape.count);
	d_bias[2] = makeVector(AlexL3_FilterShape.count);
	d_bias[4] = makeVector(AlexL5_FilterShape.count);
	d_bias[5] = makeVector(AlexL6_FilterShape.count);
	d_bias[6] = makeVector(AlexL7_FilterShape.count);
	//input 


	//layers
	//layer1
	for (int i = 0; i < 20;i++){
		std::cout << "batch" << i << std::endl;
		d_in[0] = makeTensorBatched(AlexL1_InShape, batchSize);
		convLayer(d_in[0], AlexL1_InShape, d_filter[0], AlexL1_FilterShape, d_bias[0], d_in[1], AlexL2_InShape, AlexL1_ConvArgs, batchSize);
		poolLayer(d_in[1], AlexL2_InShape, d_in[2], AlexL3_InShape, AlexL2_PoolArgs, batchSize);
		convLayer(d_in[2], AlexL3_InShape, d_filter[2], AlexL3_FilterShape, d_bias[2], d_in[3], AlexL4_InShape, AlexL3_ConvArgs, batchSize);
		poolLayer(d_in[3], AlexL4_InShape, d_in[4], AlexL5_InShape, AlexL4_PoolArgs, batchSize);
		convLayer(d_in[4], AlexL5_InShape, d_filter[4], AlexL5_FilterShape, d_bias[4], d_in[5], AlexL6_InShape, AlexL5_ConvArgs, batchSize);
		convLayer(d_in[5], AlexL6_InShape, d_filter[5], AlexL6_FilterShape, d_bias[5], d_in[6], AlexL7_InShape, AlexL6_ConvArgs, batchSize);
		convLayer(d_in[6], AlexL7_InShape, d_filter[6], AlexL7_FilterShape, d_bias[6], d_in[7], AlexL8_InShape, AlexL7_ConvArgs, batchSize);
		poolLayer(d_in[7], AlexL8_InShape, d_in[8], AlexL9_InShape, AlexL8_PoolArgs, batchSize);

		FCLayer(d_in[8], d_filter[8], d_in[9], AlexL9_aShape, AlexL9_bShape, AlexL9_cShape, batchSize);
		FCLayer(d_in[9], d_filter[9], d_in[10], AlexL10_aShape, AlexL10_bShape, AlexL10_cShape, batchSize);
	}
	//free Memory
	for (int i = 0; i < 11; i++){
		if (d_filter[i] != nullptr){		
			cudaFree(d_filter[i]); 
		}
		if (d_in[i] != nullptr){
			cudaFree(d_in[i]); 
		}
		if (d_bias[i] != nullptr){
			cudaFree(d_bias[i]); 
		}
	}
	return;
}

