#!/usr/bin/env python3
"""Correctness-check and benchmark the CUDA kernels in this directory."""

import argparse
import contextlib
import statistics
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parent

RMSNORM_BINDING = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void rmsnorm_cuda(const float* x, float* out, const float* weight,
                  int N, int D, float eps, cudaStream_t stream);

torch::Tensor rmsnorm(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "x must have a nonempty final dimension");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(-1),
                "weight must have shape [x.size(-1)]");
    TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same device");

    const int D = static_cast<int>(x.size(-1));
    const int N = static_cast<int>(x.numel() / D);
    auto out = torch::empty_like(x);
    rmsnorm_cuda(x.data_ptr<float>(), out.data_ptr<float>(), weight.data_ptr<float>(),
                 N, D, static_cast<float>(eps), at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

ATTENTION_BINDING = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void attention_cuda(const float* q, const float* k, const float* v, float* out,
                   int batch, int heads, int sequence, int head_dim,
                   cudaStream_t stream);

torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q, k, and v must be CUDA tensors");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 &&
                k.scalar_type() == torch::kFloat32 &&
                v.scalar_type() == torch::kFloat32, "q, k, and v must be float32");
    TORCH_CHECK(q.dim() == 4, "q must have shape [batch, heads, sequence, head_dim]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q, k, and v must have the same shape");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q, k, and v must be contiguous");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, and v must be on the same device");

    const int batch = static_cast<int>(q.size(0));
    const int heads = static_cast<int>(q.size(1));
    const int sequence = static_cast<int>(q.size(2));
    const int head_dim = static_cast<int>(q.size(3));
    auto out = torch::empty_like(q);
    attention_cuda(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
                   out.data_ptr<float>(), batch, heads, sequence, head_dim,
                   at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

SOFTMAX_BINDING = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void launch_softmax(const float* X, float* Y, int N, int D, cudaStream_t stream);

torch::Tensor softmax(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must have shape [N, D]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "x dimensions must be nonempty");

    const int N = static_cast<int>(x.size(0));
    const int D = static_cast<int>(x.size(1));
    auto out = torch::empty_like(x);
    launch_softmax(x.data_ptr<float>(), out.data_ptr<float>(), N, D,
                   at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

LAYERNORM_BINDING = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void launch_layernorm(const float* X, const float* gamma, const float* beta,
                      float* Y, int N, int D, float eps);

torch::Tensor layernorm(torch::Tensor x, torch::Tensor gamma,
                        torch::Tensor beta, double eps) {
    TORCH_CHECK(x.is_cuda() && gamma.is_cuda() && beta.is_cuda(),
                "x, gamma, and beta must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                gamma.scalar_type() == torch::kFloat32 &&
                beta.scalar_type() == torch::kFloat32,
                "x, gamma, and beta must be float32");
    TORCH_CHECK(x.dim() == 2, "x must have shape [N, D]");
    TORCH_CHECK(x.is_contiguous() && gamma.is_contiguous() && beta.is_contiguous(),
                "x, gamma, and beta must be contiguous");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "x dimensions must be nonempty");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == x.size(1),
                "gamma must have shape [D]");
    TORCH_CHECK(beta.dim() == 1 && beta.size(0) == x.size(1),
                "beta must have shape [D]");
    TORCH_CHECK(x.device() == gamma.device() && x.device() == beta.device(),
                "x, gamma, and beta must be on the same device");

    const int N = static_cast<int>(x.size(0));
    const int D = static_cast<int>(x.size(1));
    auto out = torch::empty_like(x);
    launch_layernorm(x.data_ptr<float>(), gamma.data_ptr<float>(), beta.data_ptr<float>(),
                     out.data_ptr<float>(), N, D, static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""

CONV2D_BINDING = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

void launch_conv2d(const float* X, const float* K, float* Y, int N, int H,
                   int W, int Ci, int Co, int R, int S, cudaStream_t stream);

torch::Tensor conv2d(torch::Tensor x, torch::Tensor kernel) {
    TORCH_CHECK(x.is_cuda() && kernel.is_cuda(), "x and kernel must be CUDA tensors");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                kernel.scalar_type() == torch::kFloat32,
                "x and kernel must be float32");
    TORCH_CHECK(x.dim() == 4 && kernel.dim() == 4,
                "x must be NHWC and kernel must be RSCK");
    TORCH_CHECK(x.is_contiguous() && kernel.is_contiguous(),
                "x and kernel must be contiguous");
    TORCH_CHECK(x.device() == kernel.device(),
                "x and kernel must be on the same device");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 &&
                x.size(3) > 0 && kernel.size(0) > 0 && kernel.size(1) > 0 &&
                kernel.size(2) > 0 && kernel.size(3) > 0,
                "all dimensions must be positive");
    TORCH_CHECK(x.size(3) == kernel.size(2),
                "input channels must match kernel channels");
    TORCH_CHECK(x.size(1) >= kernel.size(0) && x.size(2) >= kernel.size(1),
                "kernel must fit within the input spatial dimensions");

    const int N = static_cast<int>(x.size(0));
    const int H = static_cast<int>(x.size(1));
    const int W = static_cast<int>(x.size(2));
    const int Ci = static_cast<int>(x.size(3));
    const int R = static_cast<int>(kernel.size(0));
    const int S = static_cast<int>(kernel.size(1));
    const int Co = static_cast<int>(kernel.size(3));
    auto out = torch::empty({N, H - R + 1, W - S + 1, Co}, x.options());
    launch_conv2d(x.data_ptr<float>(), kernel.data_ptr<float>(),
                  out.data_ptr<float>(), N, H, W, Ci, Co, R, S,
                  at::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""


def load_rmsnorm():
    source = (ROOT / "rmsnorm.cu").read_text()
    return load_inline(
        name="kernels_practice_rmsnorm",
        cpp_sources=RMSNORM_BINDING,
        cuda_sources=source,
        functions=["rmsnorm"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


def load_attention():
    source = (ROOT / "attention.cu").read_text()
    return load_inline(
        name="kernels_practice_attention",
        cpp_sources=ATTENTION_BINDING,
        cuda_sources=source,
        functions=["attention"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


def load_softmax():
    source = (ROOT / "softmax.cu").read_text()
    return load_inline(
        name="kernels_practice_softmax",
        cpp_sources=SOFTMAX_BINDING,
        cuda_sources=source,
        functions=["softmax"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


def load_layernorm():
    source = (ROOT / "layernorm.cu").read_text()
    return load_inline(
        name="kernels_practice_layernorm",
        cpp_sources=LAYERNORM_BINDING,
        cuda_sources=source,
        functions=["layernorm"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


def load_conv2d():
    source = (ROOT / "conv2d.cu").read_text()
    return load_inline(
        name="kernels_practice_conv2d",
        cpp_sources=CONV2D_BINDING,
        cuda_sources=source,
        functions=["conv2d"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )


KERNELS = {
    "rmsnorm": load_rmsnorm,
    "attn": load_attention,
    "softmax": load_softmax,
    "layernorm": load_layernorm,
    "conv2d": load_conv2d,
}


@contextlib.contextmanager
def strict_fp32():
    """Disable TF32 so correctness references are true fp32; timing keeps torch defaults."""
    conv, mm = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = conv, mm


def reference_rmsnorm(x, weight, eps):
    inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    return x * inv_rms * weight


def reference_attention(q, k, v):
    scale = q.size(-1) ** -0.5
    probs = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
    return torch.matmul(probs, v)


def reference_conv2d(x, kernel):
    # The custom kernel uses NHWC input and RSCK weights.
    out = torch.nn.functional.conv2d(
        x.permute(0, 3, 1, 2), kernel.permute(3, 2, 0, 1)
    )
    return out.permute(0, 2, 3, 1)


def benchmark_conv2d(args, extension):
    shapes = args.conv2d_shape or [(1, 32, 32, 8, 16, 3, 3), (4, 64, 64, 16, 32, 3, 3)]
    print("Kernel: conv2d; dtype: float32; layout: NHWC/RSCK")
    for shape in shapes:
        n, h, w, ci, co, r, s = shape
        x = torch.randn((n, h, w, ci), device="cuda", dtype=torch.float32)
        kernel = torch.randn((r, s, ci, co), device="cuda", dtype=torch.float32)
        actual = extension.conv2d(x, kernel)
        with strict_fp32():
            expected = reference_conv2d(x, kernel)
        max_abs_error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-4)

        custom_ms = cuda_time_ms(
            lambda: extension.conv2d(x, kernel), args.warmup, args.iterations
        )
        torch_ms = cuda_time_ms(
            lambda: reference_conv2d(x, kernel), args.warmup, args.iterations
        )
        print(
            f"shape={shape}: max_abs_error={max_abs_error:.3e}, "
            f"custom={custom_ms:.4f} ms, torch_ref={torch_ms:.4f} ms, "
            f"speedup={torch_ms / custom_ms:.2f}x"
        )


def benchmark_softmax(args, extension):
    shapes = args.softmax_shape or [(4096, 128), (2048, 1024), (512, 4096)]
    print("Kernel: softmax; dtype: float32")
    for shape in shapes:
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        actual = extension.softmax(x)
        expected = torch.softmax(x, dim=-1)
        max_abs_error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-4)

        kernel_ms = cuda_time_ms(
            lambda: extension.softmax(x), args.warmup, args.iterations
        )
        torch_ms = cuda_time_ms(
            lambda: torch.softmax(x, dim=-1), args.warmup, args.iterations
        )
        print(
            f"shape={shape}: max_abs_error={max_abs_error:.3e}, "
            f"custom={kernel_ms:.4f} ms, torch_ref={torch_ms:.4f} ms, "
            f"speedup={torch_ms / kernel_ms:.2f}x"
        )


def benchmark_layernorm(args, extension):
    shapes = args.layernorm_shape or [(4096, 128), (2048, 1024), (512, 4096)]
    print(f"Kernel: layernorm; dtype: float32; eps: {args.eps:g}")
    for shape in shapes:
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        gamma = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
        beta = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
        actual = extension.layernorm(x, gamma, beta, args.eps)
        expected = torch.nn.functional.layer_norm(x, (shape[-1],), gamma, beta, args.eps)
        max_abs_error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-4)

        kernel_ms = cuda_time_ms(
            lambda: extension.layernorm(x, gamma, beta, args.eps),
            args.warmup, args.iterations
        )
        torch_ms = cuda_time_ms(
            lambda: torch.nn.functional.layer_norm(x, (shape[-1],), gamma, beta, args.eps),
            args.warmup, args.iterations
        )
        print(
            f"shape={shape}: max_abs_error={max_abs_error:.3e}, "
            f"custom={kernel_ms:.4f} ms, torch_ref={torch_ms:.4f} ms, "
            f"speedup={torch_ms / kernel_ms:.2f}x"
        )


def cuda_time_ms(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    return statistics.median(samples)


def parse_shape(value):
    try:
        shape = tuple(int(dim) for dim in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be comma-separated integers, e.g. 2,3,128") from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("all shape dimensions must be positive")
    return shape


def parse_attention_shape(value):
    shape = parse_shape(value)
    if len(shape) != 4:
        raise argparse.ArgumentTypeError("attention shape must be B,H,S,D, e.g. 2,8,128,64")
    return shape


def parse_softmax_shape(value):
    shape = parse_shape(value)
    if len(shape) != 2:
        raise argparse.ArgumentTypeError("softmax shape must be N,D, e.g. 4096,128")
    return shape


def parse_conv2d_shape(value):
    shape = parse_shape(value)
    if len(shape) != 7:
        raise argparse.ArgumentTypeError(
            "conv2d shape must be N,H,W,Ci,Co,R,S, e.g. 1,32,32,8,16,3,3"
        )
    n, h, w, ci, co, r, s = shape
    if h < r or w < s:
        raise argparse.ArgumentTypeError("kernel dimensions R,S must fit within H,W")
    return shape


def benchmark_attention(args, extension):
    shapes = args.attn_shape or [(2, 8, 128, 64), (2, 8, 512, 64)]
    print("Kernel: attention; dtype: float32")
    for batch, heads, seq_len, head_dim in shapes:
        q, k, v = [
            torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float32)
            for _ in range(3)
        ]
        actual = extension.attention(q, k, v)
        with strict_fp32():
            expected = reference_attention(q, k, v)
        max_abs_error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-4)

        sdpa_ms = cuda_time_ms(
            lambda: extension.attention(q, k, v), args.warmup, args.iterations
        )
        reference_ms = cuda_time_ms(
            lambda: reference_attention(q, k, v), args.warmup, args.iterations
        )
        print(
            f"shape={(batch, heads, seq_len, head_dim)}: max_abs_error={max_abs_error:.3e}, "
            f"custom={sdpa_ms:.4f} ms, torch_ref={reference_ms:.4f} ms, "
            f"speedup={reference_ms / sdpa_ms:.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    task = parser.add_mutually_exclusive_group()
    task.add_argument("--rmsnorm", action="store_true", help="run the RMSNorm CUDA benchmark (default)")
    task.add_argument("--attn", action="store_true", help="run the attention benchmark")
    task.add_argument("--softmax", action="store_true", help="run the softmax CUDA benchmark")
    task.add_argument("--layernorm", action="store_true", help="run the LayerNorm CUDA benchmark")
    task.add_argument("--conv2d", action="store_true", help="run the conv2d CUDA benchmark")
    parser.add_argument(
        "--shape", type=parse_shape, action="append",
        help="input shape; can be repeated (default: 2,3,128 and 16,512,4096)",
    )
    parser.add_argument(
        "--attn-shape", type=parse_attention_shape, action="append",
        help="attention shape B,H,S,D; can be repeated (default: 2,8,128,64 and 2,8,512,64)",
    )
    parser.add_argument(
        "--softmax-shape", type=parse_softmax_shape, action="append",
        help="softmax shape N,D; can be repeated (default: 4096,128; 2048,1024; 512,4096)",
    )
    parser.add_argument(
        "--layernorm-shape", type=parse_softmax_shape, action="append",
        help="LayerNorm shape N,D; can be repeated (default: 4096,128; 2048,1024; 512,4096)",
    )
    parser.add_argument(
        "--conv2d-shape", type=parse_conv2d_shape, action="append",
        help="conv2d shape N,H,W,Ci,Co,R,S; can be repeated",
    )
    parser.add_argument("--eps", type=float, default=1.0e-5)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run this on a GPU node with PyTorch installed.")
    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit("--iterations must be positive and --warmup cannot be negative")

    torch.manual_seed(args.seed)
    print(f"GPU: {torch.cuda.get_device_name()}")

    if args.attn:
        extension = KERNELS["attn"]()
        benchmark_attention(args, extension)
        return

    if args.softmax:
        extension = KERNELS["softmax"]()
        benchmark_softmax(args, extension)
        return

    if args.layernorm:
        extension = KERNELS["layernorm"]()
        benchmark_layernorm(args, extension)
        return

    if args.conv2d:
        extension = KERNELS["conv2d"]()
        benchmark_conv2d(args, extension)
        return

    extension = KERNELS["rmsnorm"]()
    shapes = args.shape or [(2, 3, 128), (16, 512, 4096)]
    print(f"Kernel: rmsnorm; dtype: float32; eps: {args.eps:g}")

    for shape in shapes:
        x = torch.randn(shape, device="cuda", dtype=torch.float32)
        weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
        actual = extension.rmsnorm(x, weight, args.eps)
        expected = reference_rmsnorm(x, weight, args.eps)
        max_abs_error = (actual - expected).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=1.0e-4, atol=1.0e-4)

        kernel_ms = cuda_time_ms(
            lambda: extension.rmsnorm(x, weight, args.eps), args.warmup, args.iterations
        )
        torch_ms = cuda_time_ms(
            lambda: reference_rmsnorm(x, weight, args.eps), args.warmup, args.iterations
        )
        print(
            f"shape={shape}: max_abs_error={max_abs_error:.3e}, "
            f"custom={kernel_ms:.4f} ms, torch_ref={torch_ms:.4f} ms, "
            f"speedup={torch_ms / kernel_ms:.2f}x"
        )


if __name__ == "__main__":
    main()
