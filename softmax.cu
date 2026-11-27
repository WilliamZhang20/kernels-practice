#include <cuda_runtime.h>
#include <math.h>

// Input:  X [N, D], contiguous FP32
// Output: Y [N, D], contiguous FP32

__device__ __forceinline__ float warp_reduce_sum(float v) {
    constexpr unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(mask, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    constexpr unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

// Softmax along the last dimension.
__global__ void online_softmax(
    const float* __restrict__ X,
    float* __restrict__ Y,
    int N,
    int D
) {
    int row = blockIdx.x;
    if (row >= N) return;

    const float* x = X + row * D;
    float* y = Y + row * D;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x >> 5;
    __shared__ float warpData[32];

    float max_val = -INFINITY;
    for (int i = tid; i < D; i += blockDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }
    max_val = warp_reduce_max(max_val);
    if(lane == 0) warpData[warp] = max_val;
    __syncthreads();

    if(warp == 0) {
        float block_max = lane < num_warps ? warpData[lane] : -INFINITY;
        block_max = warp_reduce_max(block_max);
        if(lane == 0) warpData[0] = block_max;
    }
    __syncthreads();
    max_val = warpData[0];
    
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        sum += expf(x[i] - max_val);
    }
    
    sum = warp_reduce_sum(sum);
    if(lane == 0) warpData[warp] = sum;
    __syncthreads();

    if(warp == 0) {
        float block_sum = lane < num_warps ? warpData[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if(lane == 0) warpData[0] = block_sum;
    }
    __syncthreads();
    sum = warpData[0];

    for (int i = tid; i < D; i += blockDim.x) {
        y[i] = expf(x[i] - max_val) / sum;
    }
}

void launch_softmax(
    const float* X,
    float* Y,
    int N,
    int D,
    cudaStream_t stream
) {
	online_softmax<<<N, 256, 0, stream>>>(X, Y, N, D);
}
