# CUDA Codegen on a Real GPU: From Loop IR to Tensor Cores

Weeks seven and eight of the 'Compilers for Machine Learning' syllabus, on a real Tesla T4. Build a small loop IR with thread indices, shared memory and accumulators, render it to CUDA with a C-style expression renderer, compile the text with nvcc and run it through ctypes on PyTorch tensors. Then climb the GEMM ladder the way a compiler would: one thread per output element, the coalescing experiment that changes which axis maps to x, shared-memory tiles with barriers, register micro-tiles, and finally a tensor-core kernel with wmma fragments, with a fused bias-and-ReLU epilogue generated from the IR. Every kernel is checked against torch.matmul and timed against cuBLAS; the ladder runs from tens of GFLOP/s to over a TFLOP/s on the same hardware.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** UOp
- [x] **2.** Renderer
- [x] **3.** Kernel
- [x] **4.** render_kernel
- [x] **5.** compile_cuda
- [x] **6.** gemm_naive
- [x] **7.** gemm_smem
- [x] **8.** gemm_regtile
- [x] **9.** gemm_wmma
- [x] **10.** with_epilogue
- [x] **11.** benchmark_ladder

## Results

```
device: Tesla T4

1. The GEMM ladder at 1024x1024x1024 (fp32, tensor cores fp16 in / fp32 out)
   cublas                       3991 GFLOP/s     84.3x the uncoalesced kernel   max err 0.0e+00
   gemm_naive_swap                47 GFLOP/s      1.0x the uncoalesced kernel   max err 2.6e-04
   gemm_naive                    312 GFLOP/s      6.6x the uncoalesced kernel   max err 2.6e-04
   gemm_smem16                   368 GFLOP/s      7.8x the uncoalesced kernel   max err 2.6e-04
   gemm_smem32                   404 GFLOP/s      8.5x the uncoalesced kernel   max err 2.6e-04
   gemm_reg64x64x8_4x4          1337 GFLOP/s     28.2x the uncoalesced kernel   max err 2.6e-04
   gemm_reg128x128x8_8x8        1301 GFLOP/s     27.5x the uncoalesced kernel   max err 2.6e-04
   gemm_wmma                    1351 GFLOP/s     28.5x the uncoalesced kernel   max err 4.4e-04
   coalescing, shared memory, register tiles and tensor cores each remove one bottleneck; cuBLAS adds staging and tuning.

2. What the compiler emitted
   register-tiled kernel: 220 lines, 16 accumulators, 2 barriers per tile, block (256, 1), grid (16, 16)
   the naive kernel in full:
     __global__ void gemm_naive(float* data0, const float* data1, const float* data2) {
         int gidx0 = blockIdx.x * blockDim.x + threadIdx.x;
         int gidx1 = blockIdx.y * blockDim.y + threadIdx.y;
         float acc0 = 0.0f;
         for (int r0 = 0; r0 < 64; r0++) {
           int v0 = (gidx1 * 64);
           int v1 = (v0 + r0);
           float v2 = data1[v1];
           int v3 = (r0 * 64);
           int v4 = (v3 + gidx0);
           float v5 = data2[v4];
           float v6 = (v2 * v5);
           float v7 = (acc0 + v6);
           acc0 = v7;
         }
         int v8 = (gidx1 * 64);

3. Fusion as an epilogue
   GEMM + bias + ReLU in one kernel: 1338 GFLOP/s, max err 1.8e-04; torch's three ops take 0.92 ms per call
   The epilogue is applied to the IR's store expressions; the loop structure never changes.
```
