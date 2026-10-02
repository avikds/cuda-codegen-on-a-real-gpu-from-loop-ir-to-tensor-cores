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

---

Built on Deep-ML.
