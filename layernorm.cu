#include <cuda_runtime.h>

#include <math.h>

// Input and output: [N, D], contiguous FP32. Normalize along the last dimension.
// gamma and beta are FP32 vectors of length D.
__global__ void naive_layernorm(
    const float* __restrict__ X,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ Y,
    int N,
    int D,
    float eps) {
    const int row = blockIdx.x;
    if (row >= N) return;

    // Intentionally naive: one thread handles one row.
    if (threadIdx.x == 0) {
        const float* x = X + row * D;
        float* y = Y + row * D;

        float mean = 0.0f;
        for (int i = 0; i < D; ++i) {
            mean += x[i];
        }
        mean /= D;

        float variance = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float diff = x[i] - mean;
            variance += diff * diff;
        }
        variance /= D;

        const float inv_std = rsqrtf(variance + eps);
        for (int i = 0; i < D; ++i) {
            y[i] = (x[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

void launch_layernorm(
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    int N,
    int D,
    float eps = 1e-5f) {
    naive_layernorm<<<N, 256>>>(X, gamma, beta, Y, N, D, eps);
}
