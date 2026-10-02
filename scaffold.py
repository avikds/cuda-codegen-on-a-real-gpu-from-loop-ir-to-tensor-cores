"""
CUDA Codegen on a Real GPU: From Loop IR to Tensor Cores scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""CUDA codegen on a real GPU: from a loop IR to tensor cores.

Story: a small loop IR with thread indices and shared memory renders to CUDA,
nvcc compiles it, ctypes runs it on torch tensors. The GEMM ladder then climbs
from one thread per element, through coalescing, shared-memory tiles and
register micro-tiles, to tensor cores, each rung a change in how loops map onto
the hardware, measured against cuBLAS on the same T4.
"""
import torch


def main() -> None:
    print("device:", torch.cuda.get_device_name(0))
    n = 1024
    print(f"\n1. The GEMM ladder at {n}x{n}x{n} (fp32, tensor cores fp16 in / fp32 out)")
    rows = benchmark_ladder(n, reps=10)
    base = rows[1][1]
    for name, g, err in rows:
        print(f"   {name:24s} {g:8.0f} GFLOP/s   {g / base:6.1f}x the uncoalesced kernel   max err {err:.1e}")
    print("   coalescing, shared memory, register tiles and tensor cores each remove one bottleneck; cuBLAS adds staging and tuning.")

    print("\n2. What the compiler emitted")
    k = gemm_regtile(n, n, n)
    src = render_kernel(k)
    print(f"   register-tiled kernel: {len(src.splitlines())} lines, {src.count('float acc')} accumulators, {src.count('__syncthreads();')} barriers per tile, block {k.block}, grid {k.grid}")
    naive = render_kernel(gemm_naive(64, 64, 64))
    print("   the naive kernel in full:")
    for line in naive.splitlines()[:16]:
        print("     " + line)

    print("\n3. Fusion as an epilogue")
    bias = param("bias", dtypes.float32, 3)
    fused = with_epilogue(gemm_regtile(n, n, n), bias_relu(bias, n), [bias])
    torch.manual_seed(1)
    A = torch.randn(n, n, device="cuda"); B = torch.randn(n, n, device="cuda"); b = torch.randn(n, device="cuda"); C = torch.zeros(n, n, device="cuda")
    lib = compile_cuda(render_kernel(fused))
    run(lib, fused.name, [C, A, B, b]); torch.cuda.synchronize()
    err = float((C - torch.relu(A @ B + b)).abs().max())
    g = bench(lib, fused.name, [C, A, B, b], 2 * n ** 3, reps=10)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): torch.relu(A @ B + b)
    torch.cuda.synchronize(); tt = (time.perf_counter() - t0) / 10
    print(f"   GEMM + bias + ReLU in one kernel: {g:.0f} GFLOP/s, max err {err:.1e}; torch's three ops take {1000 * tt:.2f} ms per call")
    print("   The epilogue is applied to the IR's store expressions; the loop structure never changes.")


if __name__ == "__main__":
    main()

