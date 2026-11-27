#include <cuda_runtime.h>
#include <mma.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __forceinline__ 
void compute_scores(
	const float* Q_shared,
	const float* K_shared,
	float* S_shared,
	int D,
	int B_M,
	int B_N
) {
	// Distribute B_M * B_N across threads
	for(int i=threadIdx.x; i<B_M*B_N; i+= blockDim.x) {
		int q_row = i / B_N;
		int k_row = i % B_N;
		float score = 0.0f;
		
		for(int d=0; d < D; d++) {
			score += Q_shared[q_row * D + d] * K_shared[k_row * D + d];
		}
		S_shared[q_row * B_N + k_row] = score * rsqrtf(float(D));
	}
}

// High-accuracy TF32 Tensor Core path for B_M=16, B_N=32, D=64.
// The two-term TF32 decomposition uses three products (hi*hi, lo*hi,
// hi*lo) and accumulates them in float, omitting only the tiny lo*lo term.
__device__ __forceinline__
void compute_scores_tc(
	const float* Q_shared,
	const float* K_shared,
	float* S_shared,
	int D,
	int B_M,
	int B_N
) {
	using namespace nvcuda;
	const int warp_id = threadIdx.x / 32;

	// Two warps cover the 16x32 score tile, one 16x16 output each.
	if (warp_id < 2 && B_M == 16 && B_N == 32 && D == 64) {
		const int key_col = warp_id * 16;
		using tf32 = wmma::precision::tf32;
		wmma::fragment<wmma::matrix_a, 16, 16, 8, tf32, wmma::row_major> a_raw, a_hi, a_lo;
		wmma::fragment<wmma::matrix_b, 16, 16, 8, tf32, wmma::col_major> b_raw, b_hi, b_lo;
		wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc_frag;
		wmma::fill_fragment(acc_frag, 0.0f);

		for (int d = 0; d < D; d += 8) {
			wmma::load_matrix_sync(a_raw, Q_shared + d, D);
			// K is [B_N,D] row-major; this views its selected 16 rows as
			// a [D,16] column-major matrix, which is the required K transpose.
			wmma::load_matrix_sync(b_raw, K_shared + key_col * D + d, D);
			for (int i = 0; i < a_raw.num_elements; ++i) {
				a_hi.x[i] = wmma::__float_to_tf32(a_raw.x[i]);
				a_lo.x[i] = wmma::__float_to_tf32(a_raw.x[i] - a_hi.x[i]);
			}
			for (int i = 0; i < b_raw.num_elements; ++i) {
				b_hi.x[i] = wmma::__float_to_tf32(b_raw.x[i]);
				b_lo.x[i] = wmma::__float_to_tf32(b_raw.x[i] - b_hi.x[i]);
			}
			wmma::mma_sync(acc_frag, a_hi, b_hi, acc_frag);
			wmma::mma_sync(acc_frag, a_lo, b_hi, acc_frag);
			wmma::mma_sync(acc_frag, a_hi, b_lo, acc_frag);
		}

		wmma::store_matrix_sync(S_shared + key_col, acc_frag, B_N,
		                        wmma::mem_row_major);
	}

	// All threads wait for the two warps to finish writing scores, then
	// distribute scaling across the full score tile.
	__syncthreads();
	for (int i = threadIdx.x; i < B_M * B_N; i += blockDim.x) {
		S_shared[i] *= rsqrtf(static_cast<float>(D));
	}
}

// GPU kernel signature
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v, float* out,
					       int batch, int heads, int sequence, int head_dim) {
	// given it's a dim3 we have, we can query gridDim.x, gridDim.y, gridDim.z, blockIdx will also be 3D
	// however blockDim.x 
	int q_tile = blockIdx.x;
	int h = blockIdx.y;
	int b = blockIdx.z;

	// need to schedule each head
	int q_start = q_tile * 16;
	int head_offset = (b * heads + h) * sequence * head_dim;
	const float* q_tile_ptr = q + head_offset + q_start*head_dim;

	// process the current CTA's q_tile
	constexpr int B_M = 16;
	constexpr int D = 64;

	__shared__ __align__(32) float Q_shared[B_M][D];
	int tid = threadIdx.x;
	int lane = tid % warpSize;
	int warp_id = tid / warpSize;

	for(int i = tid; i < B_M * D; i+=blockDim.x) {
		int row = i / D;
		int col = i % D;
		const float q_value = (q_start + row < sequence)
			? q_tile_ptr[i]
			: 0.0f;
		Q_shared[row][col] = q_value;
	}

	__syncthreads();

	// pull tiles for K^T and V
	constexpr int B_N = 32;
	__shared__ __align__(32) float K_shared[B_N][D];
	__shared__ float V_shared[B_N][D];
	__shared__ __align__(32) float S_shared[B_M][B_N];
	__shared__ float m_running[B_M];
	__shared__ float l_running[B_M];
	__shared__ float O_accum[B_M][D];
	__shared__ float m_new[B_M];
	__shared__ float alpha[B_M];
	__shared__ float l_new[B_M];
	__shared__ float P[B_M][B_N];
	
	// Initialize the online softmax state once per query tile. It persists
	// across all K/V tiles in the loop below.
	for (int i = tid; i < B_M; i += blockDim.x) {
		m_running[i] = -FLT_MAX;
		l_running[i] = 0.0f;
	}
	for (int i = tid; i < B_M * D; i += blockDim.x) {
		O_accum[i / D][i % D] = 0.0f;
	}
	__syncthreads();

	for(int k_start = 0; k_start < sequence; k_start += B_N) {
		
		const int kv_head_offset = head_offset + k_start * head_dim;
		for (int i = tid; i < B_N * D; i += blockDim.x) {
			const int row = i / D;
			const int col = i % D;
			const int kv_row = k_start + row;
			if (kv_row < sequence) {
				const int offset = kv_head_offset + row * head_dim + col;
				const float k_value = k[offset];
				K_shared[row][col] = k_value;
				V_shared[row][col] = v[offset];
			} else {
				// Keep unused K/V data finite. Mask the corresponding attention
				// logits to -infinity after the QK dot product is computed.
				K_shared[row][col] = 0.0f;
				V_shared[row][col] = 0.0f;
			}
		}
		__syncthreads();

		// obtain score tile S = Q @ K^T
		compute_scores_tc(
		    &Q_shared[0][0],
		    &K_shared[0][0],
		    &S_shared[0][0],
		    D, B_M, B_N
		);
		__syncthreads();

		// First online-softmax step: find this tile's maximum score per query row.
		for (int row = tid; row < B_M; row += blockDim.x) {
			float tile_max = -FLT_MAX;
			for (int col = 0; col < B_N; ++col) {
				if (k_start + col < sequence) {
					tile_max = fmaxf(tile_max, S_shared[row][col]);
				}
			}
			m_new[row] = fmaxf(m_running[row], tile_max);
			alpha[row] = expf(m_running[row] - m_new[row]);
		}
		__syncthreads();

		// Form this tile's unnormalized probabilities. A key outside the
		// sequence contributes zero, equivalent to a -infinity logit.
		for (int i = tid; i < B_M * B_N; i += blockDim.x) {
			const int row = i / B_N;
			const int col = i % B_N;
			P[row][col] = (k_start + col < sequence)
				? expf(S_shared[row][col] - m_new[row])
				: 0.0f;
		}
		__syncthreads();

		// Sum P across each row with one warp per row. With eight warps in
		// this block, each warp handles two rows over the loop iterations.
		for (int row = warp_id; row < B_M; row += blockDim.x / warpSize) {
			float row_sum = P[row][lane];
			for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
				row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
			}
			if (lane == 0) {
				l_new[row] = alpha[row] * l_running[row] + row_sum;
			}
		}
		__syncthreads();
		
		// Warp-reduce P[row,:] * V[:,d] for each output element.
		for (int i = warp_id; i < B_M * D; i += blockDim.x / warpSize) {
			const int row = i / D;
			const int d = i % D;
			float pv_sum = P[row][lane] * V_shared[lane][d];
			for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
				pv_sum += __shfl_down_sync(0xffffffff, pv_sum, offset);
			}
			if (lane == 0) {
				O_accum[row][d] = alpha[row] * O_accum[row][d] + pv_sum;
			}
		}

		for (int row = tid; row < B_M; row += blockDim.x) {
			m_running[row] = m_new[row];
			l_running[row] = l_new[row];
		}

		__syncthreads();
		
	}
	
	// normalize and done!
	for (int i = tid; i < B_M*D; i += blockDim.x) {
		const int row = i / D;
                const int d = i % D;
               	if (q_start + row < sequence) {
               		out[head_offset + (q_start + row) * head_dim + d] = O_accum[row][d] / l_running[row];
                }
        }	
}

// Host entry point
void attention_cuda(const float* q, const float* k, const float* v, float* out,
		   int batch, int heads, int sequence, int head_dim,
		   cudaStream_t stream) {
	if (head_dim != 64) {
		std::fprintf(stderr, "attention_cuda currently requires head_dim == 64 (got %d)\n", head_dim);
		return;
	}
	constexpr int BLOCK_SIZE = 256;
	const int query_tiles = (sequence + 15) / 16;
	flash_attention_kernel<<<dim3(query_tiles, heads, batch), BLOCK_SIZE, 0, stream>>>(
		q, k, v, out, batch, heads, sequence, head_dim);
}

int main() {
	const int batch = 1;
	const int heads = 8;
	const int sequence = 1024;
	const int head_dim = 64;
	const size_t elements = static_cast<size_t>(batch) * heads * sequence * head_dim;
	const size_t bytes = elements * sizeof(float);

	std::vector<float> hq(elements), hk(elements), hv(elements), hout(elements);
	for (size_t i = 0; i < elements; ++i) {
		hq[i] = (std::rand() / static_cast<float>(RAND_MAX)) * 2.0f - 1.0f;
		hk[i] = (std::rand() / static_cast<float>(RAND_MAX)) * 2.0f - 1.0f;
		hv[i] = (std::rand() / static_cast<float>(RAND_MAX)) * 2.0f - 1.0f;
	}

	float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
	cudaMalloc(&dq, bytes);
	cudaMalloc(&dk, bytes);
	cudaMalloc(&dv, bytes);
	cudaMalloc(&dout, bytes);
	cudaMemcpy(dq, hq.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dk, hk.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dv, hv.data(), bytes, cudaMemcpyHostToDevice);

	attention_cuda(dq, dk, dv, dout, batch, heads, sequence, head_dim, 0);
	cudaDeviceSynchronize();
	cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost);

	std::printf("Ran attention for batch=%d, heads=%d, sequence=%d, head_dim=%d\n",
	            batch, heads, sequence, head_dim);

	cudaFree(dq);
	cudaFree(dk);
	cudaFree(dv);
	cudaFree(dout);
}
