#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <vector>
__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v)  { return __bfloat162float(v); }

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ float         from_f<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half        from_f<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

template <typename T, int N>
struct alignas(sizeof(T) * N) Vec { T v[N]; };

__device__ __forceinline__ float warp_reduce_sum(float v) {
	#pragma unroll
	for(int o=16; o>0; o >>= 1) {
		v += __shfl_xor_sync(0xffffffff, v, o);
	}
	return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
	__shared__ float smem[32];
	const int lane = threadIdx.x & 31;
	const int wid = threadIdx.x >> 5;
	const int nw = (blockDim.x + 31) >> 5;

	v = warp_reduce_sum(v);
	if(lane == 0) smem[wid] = v;
	__syncthreads();
	if (wid == 0) {
		v = (lane < nw) ? smem[lane] : 0.f;
		v = warp_reduce_sum(v);
	}
	return v;
}

template <typename T, int VEC, int BLOCK, int ITERS>
__global__ void __launch_bounds__(BLOCK) rmsnorm_kernel(
	const T* __restrict__ x, // [N, D]
	T* __restrict__ out, // [N, d]
	const T* __restrict__ weight, // scale of size d
	int N,
	float eps
) {
	constexpr int D = BLOCK * VEC * ITERS;
	using V = Vec<T, VEC>;
	
	const V* xr = reinterpret_cast<const V*>(x + (size_t)blockIdx.x * D);
	const V* wv = reinterpret_cast<const V*>(weight);
	V*       yr = reinterpret_cast<V*>(out + (size_t)blockIdx.x * D);
	float xf[ITERS][VEC];
	float ss = 0.0f;
	
	#pragma unroll
	for(int it=0; it<ITERS; it++) {
		V t = xr[it * BLOCK + threadIdx.x]; // coalesced load instead of skipped in slower version
		#pragma unroll
		for(int j=0; j<VEC; ++j) {
			xf[it][j] = to_f(t.v[j]);
			ss += xf[it][j] * xf[it][j];
		}
	}

	ss = block_reduce_sum(ss);
	const float inv = rsqrtf(ss / D + eps);
	
	#pragma unroll
	for(int it=0; it<ITERS; it++) {
  		V wt = wv[it * BLOCK + threadIdx.x];
		V o;
		#pragma unroll
		for(int j=0; j<VEC; ++j) {
			o.v[j] = from_f<T>(xf[it][j] * inv * to_f(wt.v[j]));	
		}
		yr[it * BLOCK + threadIdx.x] = o;
	}
}

template <typename T, int VEC, int BLOCK, int ITERS>
static void launch_rmsnorm(const T* x, T* out, const T* weight, int rows,
                           float eps, cudaStream_t stream) {
  rmsnorm_kernel<T, VEC, BLOCK, ITERS>
      <<<rows, BLOCK, 0, stream>>>(x, out, weight, rows, eps);
}

// This signature must match the declaration used by benchmark.py's PyTorch binding.
void rmsnorm_cuda(const float* x, float* out, const float* weight,
                  int N, int D, float eps, cudaStream_t stream) {
  constexpr int VEC = 16 / sizeof(float);
  constexpr int BLOCK = 256;
  constexpr int ELEMENTS_PER_ITER = BLOCK * VEC;

  if (D == 128) {
    // A smaller block covers the benchmark's 128-element row without overreading.
    launch_rmsnorm<float, VEC, 32, 1>(x, out, weight, N, eps, stream);
    return;
  }

  if (D % ELEMENTS_PER_ITER != 0) return;
  switch (D / ELEMENTS_PER_ITER) {
    case 1: launch_rmsnorm<float, VEC, BLOCK, 1>(x, out, weight, N, eps, stream); return;
    case 2: launch_rmsnorm<float, VEC, BLOCK, 2>(x, out, weight, N, eps, stream); return;
    case 4: launch_rmsnorm<float, VEC, BLOCK, 4>(x, out, weight, N, eps, stream); return;
    case 8: launch_rmsnorm<float, VEC, BLOCK, 8>(x, out, weight, N, eps, stream); return;
    default: return;
  }
}

int main() {
  const int rows = 4096, D = 4096; const float eps = 1e-6f;
  std::vector<float> hx(rows * D), hw(D), hy(rows * D);
  for (auto& v : hx) v = (rand() / (float)RAND_MAX) * 2.f - 1.f;
  for (auto& v : hw) v = 0.5f + rand() / (float)RAND_MAX;

  float *dx, *dw, *dy;
  cudaMalloc(&dx, hx.size() * 4); cudaMalloc(&dw, D * 4); cudaMalloc(&dy, hy.size() * 4);
  cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, hw.data(), D * 4, cudaMemcpyHostToDevice);

  rmsnorm_cuda(dx, dy, dw, rows, D, eps, 0);
  cudaMemcpy(hy.data(), dy, hy.size() * 4, cudaMemcpyDeviceToHost);

  double max_err = 0;
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    for (int i = 0; i < D; ++i) ss += (double)hx[r*D+i] * hx[r*D+i];
    double inv = 1.0 / std::sqrt(ss / D + eps);
    for (int i = 0; i < D; ++i)
      max_err = std::fmax(max_err, std::fabs(hx[r*D+i] * inv * hw[i] - hy[r*D+i]));
  }
  printf("max abs err: %g\n", max_err);   // expect ~1e-6 for fp32
}
