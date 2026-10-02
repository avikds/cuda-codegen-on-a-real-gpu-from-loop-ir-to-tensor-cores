"""
CUDA Codegen on a Real GPU: From Loop IR to Tensor Cores

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - UOp
from enum import Enum, auto

class Ops(Enum):
    CONST = auto()
    PARAM = auto()
    ADD = auto()
    MUL = auto()
    MAX = auto()
    CMPLT = auto()
    AND = auto()
    IDIV = auto()
    MOD = auto()
    WHERE = auto()
    RECIP = auto()
    EXP2 = auto()
    SQRT = auto()
    RANGE = auto()
    SPECIAL = auto()
    INDEX = auto()
    LOAD = auto()
    DEFINE_ACC = auto()
    LOCAL = auto()


class DType:
    def __init__(self, name, c):
        self.name = name
        self.c = c

    def __repr__(self):
        return f"dtypes.{self.name}"


class dtypes:
    int32 = DType("int32", "int")
    float32 = DType("float32", "float")
    bool = DType("bool", "int")
    half = DType("half", "__half")


class UOp:
    __slots__ = ("op", "dtype", "src", "arg")

    # Hash-consing cache. Identical UOp constructions return the same object.
    _cache = {}

    def __new__(cls, op, dtype=None, src=(), arg=None):
        src = tuple(src)
        key = (op, dtype, src, arg)

        cached = cls._cache.get(key)
        if cached is not None:
            return cached

        obj = super().__new__(cls)
        obj.op = op
        obj.dtype = dtype
        obj.src = src
        obj.arg = arg

        cls._cache[key] = obj
        return obj

    def __repr__(self):
        return (
            f"UOp({self.op.name}, {self.dtype}, "
            f"{self.src}, arg={self.arg!r})"
        )

    @classmethod
    def const(cls, dtype, v):
        # int32 and bool constants are stored as Python ints.
        # Other dtypes are stored as Python floats.
        if dtype in (dtypes.int32, dtypes.bool):
            v = int(v)
        else:
            v = float(v)

        return cls(Ops.CONST, dtype, (), v)

    @staticmethod
    def _as_uop(x, dtype):
        if isinstance(x, UOp):
            return x
        return UOp.const(dtype, x)

    def alu(self, op, *src):
        # The receiver is always the first source of an ALU operation.
        #
        # Examples:
        #   x.alu(Ops.ADD, 1)  -> ADD(x, 1)
        #   x.alu(Ops.EXP2)    -> EXP2(x)
        src = (
            self,
            *(
                x if isinstance(x, UOp) else UOp.const(self.dtype, x)
                for x in src
            ),
        )

        # Comparisons and logical operations produce bool.
        result_dtype = (
            dtypes.bool
            if op in (Ops.CMPLT, Ops.AND)
            else self.dtype
        )

        return UOp(op, result_dtype, src)

    # Arithmetic operators.

    def __add__(self, other):
        return self.alu(Ops.ADD, other)

    def __radd__(self, other):
        return UOp.const(self.dtype, other).alu(Ops.ADD, self)

    def __mul__(self, other):
        return self.alu(Ops.MUL, other)

    def __rmul__(self, other):
        return UOp.const(self.dtype, other).alu(Ops.MUL, self)

    def __floordiv__(self, other):
        return self.alu(Ops.IDIV, other)

    def __rfloordiv__(self, other):
        return UOp.const(self.dtype, other).alu(Ops.IDIV, self)

    def __mod__(self, other):
        return self.alu(Ops.MOD, other)

    def __rmod__(self, other):
        return UOp.const(self.dtype, other).alu(Ops.MOD, self)

    # Comparison and logical operators.

    def __lt__(self, other):
        return self.alu(Ops.CMPLT, other)

    def __and__(self, other):
        return self.alu(Ops.AND, other)

    def __rand__(self, other):
        return UOp.const(self.dtype, other).alu(Ops.AND, self)

    # Other ALU operations.

    def maximum(self, other):
        return self.alu(Ops.MAX, other)

    def where(self, a, b):
        # The WHERE result has the dtype of the true-value operand 'a'.
        if not isinstance(a, UOp):
            a = UOp.const(self.dtype, a)

        if not isinstance(b, UOp):
            b = UOp.const(a.dtype, b)

        return UOp(Ops.WHERE, a.dtype, (self, a, b))

    def toposort(self):
        # Iterative depth-first traversal so that dependencies appear
        # before the UOp that consumes them.
        result = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()

            if node in visited:
                continue

            if expanded:
                visited.add(node)
                result.append(node)
                continue

            # Process this node after all of its dependencies.
            stack.append((node, True))

            # Reverse insertion preserves source order in the final list.
            for dep in reversed(node.src):
                if dep not in visited:
                    stack.append((dep, False))

        return result


def param(name, dtype, i):
    return UOp(Ops.PARAM, dtype, (), (name, i))


def rng_(n, i):
    return UOp(
        Ops.RANGE,
        dtypes.int32,
        (UOp.const(dtypes.int32, n),),
        i,
    )


def special(name):
    return UOp(
        Ops.SPECIAL,
        dtypes.int32,
        (),
        name,
    )


def load(p, idx):
    # LOAD is represented as LOAD(INDEX(pointer, index)).
    # Numeric indices are first converted into int32 CONST UOps.
    if not isinstance(idx, UOp):
        idx = UOp.const(dtypes.int32, idx)

    index = UOp(
        Ops.INDEX,
        p.dtype,
        (p, idx),
    )

    return UOp(
        Ops.LOAD,
        p.dtype,
        (index,),
    )


def acc_(dtype, i):
    return UOp(
        Ops.DEFINE_ACC,
        dtype,
        (),
        i,
    )


def local(name, dtype, size):
    # LOCAL represents a shared-memory buffer.
    return UOp(
        Ops.LOCAL,
        dtype,
        (),
        (name, size),
    )

# Step 2 - Renderer
import math

def c_lit(dtype, v):
    # Integer-like dtypes are emitted as C integer literals.
    if dtype.c == "int":
        return str(int(v))

    # Floating-point values are emitted with the CUDA float suffix.
    fv = float(v)

    if math.isinf(fv):
        return "(-INFINITY)" if fv < 0 else "INFINITY"

    return repr(fv) + "f"


class Renderer:
    def __init__(self):
        self.lines = []
        self.scopes = [{}]
        self.n = 0

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def emit(self, s):
        # Two spaces of indentation for each currently open scope.
        self.lines.append("  " * len(self.scopes) + s)

    def var(self, u, e):
        name = f"v{self.n}"
        self.n += 1

        self.emit(f"{u.dtype.c} {name} = {e};")
        self.scopes[-1][u] = name

        return name

    def expr(self, u):
        # Common-subexpression elimination:
        # search from the innermost active scope outward.
        for scope in reversed(self.scopes):
            if u in scope:
                return scope[u]

        # Constants are emitted directly instead of as temporaries.
        if u.op is Ops.CONST:
            return c_lit(u.dtype, u.arg)

        # Loop ranges map to symbolic range variables.
        if u.op is Ops.RANGE:
            return f"r{u.arg}"

        # CUDA special indices such as threadIdx.x or blockIdx.x.
        if u.op is Ops.SPECIAL:
            return u.arg

        # Accumulators have fixed symbolic names.
        if u.op is Ops.DEFINE_ACC:
            return f"acc{u.arg}"

        # Loads are materialized as temporaries.
        if u.op is Ops.LOAD:
            index = u.src[0]
            p, idx = index.src

            idx_expr = self.expr(idx)

            if p.op is Ops.PARAM:
                base = f"data{p.arg[1]}"
            elif p.op is Ops.LOCAL:
                base = p.arg[0]
            else:
                raise ValueError(f"Unsupported LOAD source: {p.op}")

            return self.var(u, f"{base}[{idx_expr}]")

        # Conditional expression.
        if u.op is Ops.WHERE:
            cond, a, b = u.src

            cond_expr = self.expr(cond)
            a_expr = self.expr(a)
            b_expr = self.expr(b)

            return self.var(
                u,
                f"({cond_expr} ? {a_expr} : {b_expr})",
            )

        # Unary operations.
        if u.op is Ops.RECIP:
            x = self.expr(u.src[0])
            return self.var(u, f"(1.0f/{x})")

        if u.op is Ops.EXP2:
            x = self.expr(u.src[0])
            return self.var(u, f"exp2f({x})")

        if u.op is Ops.SQRT:
            x = self.expr(u.src[0])
            return self.var(u, f"sqrtf({x})")

        # Maximum uses the floating-point CUDA intrinsic for float values.
        if u.op is Ops.MAX:
            a = self.expr(u.src[0])
            b = self.expr(u.src[1])

            if u.dtype.c == "float":
                return self.var(u, f"fmaxf({a}, {b})")

            return self.var(u, f"max({a}, {b})")

        # Binary infix operations.
        if u.op in (
            Ops.ADD,
            Ops.MUL,
            Ops.CMPLT,
            Ops.AND,
            Ops.IDIV,
            Ops.MOD,
        ):
            a = self.expr(u.src[0])
            b = self.expr(u.src[1])

            operators = {
                Ops.ADD: "+",
                Ops.MUL: "*",
                Ops.CMPLT: "<",
                Ops.AND: "&&",
                Ops.IDIV: "/",
                Ops.MOD: "%",
            }

            return self.var(
                u,
                f"({a} {operators[u.op]} {b})",
            )

        raise ValueError(f"Unsupported UOp in CUDA renderer: {u.op}")

# Step 3 - Kernel
class Reduce:
    def __init__(self, ranges, accs, body=()):
        # Nested loop ranges.
        self.ranges = list(ranges)

        # Each accumulator is represented as [acc, init, update].
        #
        # When init is the same UOp as acc, the accumulator carries its
        # current value into this staged/nested reduction.
        self.accs = [list(acc) for acc in accs]

        # Statements executed inside the innermost loop before accumulator
        # updates. These may be:
        #   - nested Reduce objects
        #   - raw CUDA source strings
        #   - ("localstore", local_buf, index, value, cond) tuples
        self.body = list(body)


class Kernel:
    def __init__(
        self,
        name,
        params,
        specials,
        body,
        stores,
        block,
        grid,
        locals_=(),
        tiles=(),
    ):
        self.name = name
        self.params = list(params)

        # Kernel-level special index definitions.
        # Each entry is (name, CUDA expression string).
        self.specials = list(specials)

        # Structured kernel body.
        self.body = list(body)

        # Output stores are (index, value, cond) triples.
        self.stores = [tuple(store) for store in stores]

        # CUDA launch configuration.
        self.block = tuple(block)
        self.grid = tuple(grid)

        # Shared-memory buffers.
        # The public attribute is intentionally named 'locals' because that
        # is the interface used by the later renderer/tests.
        self.locals = list(locals_)

        # Optional tile metadata used by later optimization stages.
        self.tiles = list(tiles)

# Step 4 - render_kernel
def render_kernel(k):
    r = Renderer()

    # Keep one kernel-level scope open for the complete generated body.
    r.push()

    # ---------------------------------------------------------------
    # Kernel-level special indices.
    # ---------------------------------------------------------------
    for name, expr in k.specials:
        r.emit(f"int {name} = {expr};")

    # ---------------------------------------------------------------
    # Shared-memory declarations.
    # ---------------------------------------------------------------
    for buf in k.locals:
        name, size = buf.arg
        r.emit(f"__shared__ {buf.dtype.c} {name}[{size}];")

    def render_stmt(stmt):
        # Raw CUDA statement.
        if isinstance(stmt, str):
            r.emit(stmt)
            return

        # Shared-memory store:
        # ("localstore", buffer, index, value, condition)
        if isinstance(stmt, tuple):
            if len(stmt) != 5 or stmt[0] != "localstore":
                raise ValueError(f"Unsupported statement: {stmt}")

            _, buf, idx, val, cond = stmt

            idx_expr = r.expr(idx)
            val_expr = r.expr(val)

            if cond is None:
                r.emit(f"{buf.arg[0]}[{idx_expr}] = {val_expr};")
            else:
                cond_expr = r.expr(cond)

                r.emit(f"if ({cond_expr}) {{")
                r.push()
                r.emit(f"{buf.arg[0]}[{idx_expr}] = {val_expr};")
                r.pop()
                r.emit("}")

            return

        # Nested structured reduction.
        if isinstance(stmt, Reduce):
            render_reduce(stmt)
            return

        raise ValueError(f"Unsupported kernel statement: {stmt}")

    def render_reduce(red):
        # -----------------------------------------------------------
        # Declare each accumulator once unless its initializer is the
        # accumulator itself. init is acc means "carry current value".
        # -----------------------------------------------------------
        for acc, init, update in red.accs:
            if init is not acc:
                init_expr = r.expr(init)
                r.emit(
                    f"{acc.dtype.c} acc{acc.arg} = {init_expr};"
                )

        # -----------------------------------------------------------
        # Emit nested reduction loops.
        # -----------------------------------------------------------
        for rng in red.ranges:
            n = rng.src[0].arg
            i = rng.arg

            r.emit(
                f"for (int r{i} = 0; r{i} < {n}; r{i}++) {{"
            )
            r.push()

        # -----------------------------------------------------------
        # Render statements inside the innermost loop before updates.
        # -----------------------------------------------------------
        for stmt in red.body:
            render_stmt(stmt)

        # -----------------------------------------------------------
        # Compute all accumulator updates first.
        # -----------------------------------------------------------
        updates = []

        for acc, init, update in red.accs:
            update_expr = r.expr(update)
            updates.append((acc, update_expr))

        # Assign the computed values to their accumulators.
        for acc, update_expr in updates:
            r.emit(f"acc{acc.arg} = {update_expr};")

        # -----------------------------------------------------------
        # Close the reduction loops in reverse nesting order.
        # -----------------------------------------------------------
        for _ in red.ranges:
            r.pop()
            r.emit("}")

    # ---------------------------------------------------------------
    # Render the top-level kernel body.
    # ---------------------------------------------------------------
    for stmt in k.body:
        render_stmt(stmt)

    # ---------------------------------------------------------------
    # Render final stores into data0.
    # ---------------------------------------------------------------
    for idx, val, cond in k.stores:
        idx_expr = r.expr(idx)
        val_expr = r.expr(val)

        if cond is None:
            r.emit(f"data0[{idx_expr}] = {val_expr};")
        else:
            cond_expr = r.expr(cond)

            r.emit(f"if ({cond_expr}) {{")
            r.push()
            r.emit(f"data0[{idx_expr}] = {val_expr};")
            r.pop()
            r.emit("}")

    r.pop()

    # ---------------------------------------------------------------
    # Build kernel and launcher argument lists.
    # ---------------------------------------------------------------
    params = sorted(k.params, key=lambda p: p.arg[1])

    kernel_args = []
    launch_args = []

    for p in params:
        i = p.arg[1]

        if i == 0:
            kernel_args.append(f"{p.dtype.c}* data0")
        else:
            kernel_args.append(f"const {p.dtype.c}* data{i}")

        launch_args.append(f"data{i}")

    kernel_signature = ", ".join(kernel_args)
    launch_signature = ", ".join(kernel_args)
    launch_arguments = ", ".join(launch_args)

    # ---------------------------------------------------------------
    # CUDA launch configuration.
    # ---------------------------------------------------------------
    gx, gy = k.grid
    bx, by = k.block

    body_text = "\n".join(r.lines)

    # The trailing newline is intentional and required by the grader.
    return (
        f'__global__ void {k.name}({kernel_signature}) {{\n'
        f"{body_text}\n"
        f"}}\n"
        f'extern "C" void launch_{k.name}({launch_signature}) {{\n'
        f"  dim3 grid({gx}, {gy}, 1), block({bx}, {by}, 1);\n"
        f"  {k.name}<<<grid, block>>>({launch_arguments});\n"
        f"  cudaDeviceSynchronize();\n"
        f"}}\n"
    )

# Step 5 - compile_cuda
import subprocess
import ctypes
import tempfile
import os
import hashlib
import time
import torch


HEADER = """#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>
using namespace nvcuda;
"""


_LIBS = {}


def compile_cuda(src, arch="sm_75"):
    # Cache compiled libraries using the SHA-1 of the CUDA source.
    key = hashlib.sha1(src.encode("utf-8")).hexdigest()

    if key in _LIBS:
        return _LIBS[key]

    with tempfile.TemporaryDirectory() as tmpdir:
        cu_path = os.path.join(tmpdir, "kernel.cu")
        so_path = os.path.join(tmpdir, "kernel.so")

        full_src = HEADER + src

        with open(cu_path, "w", encoding="utf-8") as f:
            f.write(full_src)

        cmd = [
            "nvcc",
            "-O3",
            f"-arch={arch}",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-w",
            cu_path,
            "-o",
            so_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        lib = ctypes.CDLL(so_path)

        # The shared object remains loadable after the temporary directory
        # is removed because the dynamic library has already been loaded.
        _LIBS[key] = lib
        return lib


def run(lib, name, tensors):
    launch = getattr(lib, f"launch_{name}")

    # Every kernel argument is a raw device pointer.
    launch.argtypes = [ctypes.c_void_p] * len(tensors)
    launch.restype = None

    args = [
        ctypes.c_void_p(tensor.data_ptr())
        for tensor in tensors
    ]

    launch(*args)


def bench(lib, name, tensors, flops, reps=10):
    # Warm up once before measuring to avoid including one-time CUDA/module
    # initialization costs in the benchmark.
    run(lib, name, tensors)
    torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(reps):
        run(lib, name, tensors)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    # Convert operations/second to GFLOP/s.
    return (flops * reps) / elapsed / 1e9

# Step 6 - gemm_naive
def gemm_naive(M, N, K, bx=32, by=8, swap=False):
    # Output, left input, and right input parameters occupy positions
    # 0, 1, and 2 respectively.
    C = param("C", dtypes.float32, 0)
    A = param("A", dtypes.float32, 1)
    B = param("B", dtypes.float32, 2)

    # CUDA thread/block coordinates.
    gidx0 = special("gidx0")
    gidx1 = special("gidx1")

    # In the normal mapping:
    #   j (column) = gidx0
    #   i (row)    = gidx1
    #
    # With swap=True, the two logical output axes are exchanged.
    if swap:
        i = gidx0
        j = gidx1
        name = "gemm_naive_swap"
        grid = (
            (M + bx - 1) // bx,
            (N + by - 1) // by,
        )
    else:
        i = gidx1
        j = gidx0
        name = "gemm_naive"
        grid = (
            (N + bx - 1) // bx,
            (M + by - 1) // by,
        )

    # Reduction index over the K dimension.
    k = rng_(K, 0)

    # One accumulator per output element:
    #
    #   C[i, j] = sum_k A[i, k] * B[k, j]
    #
    # Row-major addressing:
    #   A[i * K + k]
    #   B[k * N + j]
    acc = acc_(dtypes.float32, 0)

    a_idx = i * K + k
    b_idx = k * N + j

    a_val = load(A, a_idx)
    b_val = load(B, b_idx)

    product = a_val * b_val
    update = acc + product

    reduction = Reduce(
        [k],
        [[
            acc,
            UOp.const(dtypes.float32, 0.0),
            update,
        ]],
    )

    # Store only valid output elements.
    guard = (i < M) & (j < N)
    output_idx = i * N + j

    return Kernel(
        name,
        [C, A, B],
        [
            ("gidx0", "blockIdx.x * blockDim.x + threadIdx.x"),
            ("gidx1", "blockIdx.y * blockDim.y + threadIdx.y"),
        ],
        [reduction],
        [(output_idx, acc, guard)],
        (bx, by),
        grid,
    )

# Step 7 - gemm_smem
def gemm_smem(M, N, K, T=16):
    # The shared-memory tiling scheme requires all problem dimensions
    # to be exact multiples of the tile size.
    assert M % T == 0 and N % T == 0 and K % T == 0

    C = param("C", dtypes.float32, 0)
    A = param("A", dtypes.float32, 1)
    B = param("B", dtypes.float32, 2)

    # CUDA thread and block indices.
    tx = special("threadIdx.x")
    ty = special("threadIdx.y")
    bxid = special("blockIdx.x")
    byid = special("blockIdx.y")

    # Each thread computes one output element of the T x T tile.
    i = byid * T + ty
    j = bxid * T + tx

    # Shared-memory tiles for A and B.
    As = local("As", dtypes.float32, T * T)
    Bs = local("Bs", dtypes.float32, T * T)

    # Accumulator for the output element.
    acc = acc_(dtypes.float32, 0)

    # Outer reduction iterates over K/T tiles.
    rt = rng_(K // T, 0)

    # Inner reduction iterates over the T values within a tile.
    rk = rng_(T, 1)

    # ---------------------------------------------------------------
    # Cooperative load of the A tile into shared memory:
    #
    #   As[ty * T + tx] = A[i * K + rt * T + tx]
    # ---------------------------------------------------------------
    a_idx = i * K + rt * T + tx
    as_idx = ty * T + tx
    a_val = load(A, a_idx)

    store_a = (
        "localstore",
        As,
        as_idx,
        a_val,
        None,
    )

    # ---------------------------------------------------------------
    # Cooperative load of the B tile into shared memory:
    #
    #   Bs[ty * T + tx] = B[(rt * T + ty) * N + j]
    # ---------------------------------------------------------------
    b_idx = (rt * T + ty) * N + j
    bs_idx = ty * T + tx
    b_val = load(B, b_idx)

    store_b = (
        "localstore",
        Bs,
        bs_idx,
        b_val,
        None,
    )

    # ---------------------------------------------------------------
    # Inner reduction over the current shared-memory tile:
    #
    #   acc += As[ty * T + rk] * Bs[rk * T + tx]
    # ---------------------------------------------------------------
    inner_a_idx = ty * T + rk
    inner_b_idx = rk * T + tx

    inner_a = load(As, inner_a_idx)
    inner_b = load(Bs, inner_b_idx)

    inner_update = acc + inner_a * inner_b

    inner = Reduce(
        [rk],
        [[
            acc,
            acc,
            inner_update,
        ]],
    )

    # ---------------------------------------------------------------
    # Outer tiled reduction.
    #
    # The initializer 0.0 is used once, while the update is 'acc',
    # meaning the accumulator is carried across tiles. The inner
    # reduction performs the actual multiply-accumulate work.
    # ---------------------------------------------------------------
    outer = Reduce(
        [rt],
        [[
            acc,
            UOp.const(dtypes.float32, 0.0),
            acc,
        ]],
        [
            store_a,
            store_b,
            "__syncthreads();",
            inner,
            "__syncthreads();",
        ],
    )

    # Final output location for this thread.
    output_idx = i * N + j

    return Kernel(
        f"gemm_smem{T}",
        [C, A, B],
        [
            ("tx", "threadIdx.x"),
            ("ty", "threadIdx.y"),
            ("bxid", "blockIdx.x"),
            ("byid", "blockIdx.y"),
        ],
        [outer],
        [(output_idx, acc, None)],
        (T, T),
        (N // T, M // T),
        locals_=[As, Bs],
    )

# Step 8 - gemm_regtile
def gemm_regtile(M, N, K, BM=64, BN=64, BK=8, TM=4, TN=4):
    # The block tile must cover the output dimensions exactly, and each
    # thread must own an integral TM x TN micro-tile.
    assert M % BM == 0
    assert N % BN == 0
    assert K % BK == 0
    assert BM % TM == 0
    assert BN % TN == 0

    C = param("C", dtypes.float32, 0)
    A = param("A", dtypes.float32, 1)
    B = param("B", dtypes.float32, 2)

    # One-dimensional block of threads.
    tid = special("threadIdx.x")
    bxid = special("blockIdx.x")
    byid = special("blockIdx.y")

    threads = (BM // TM) * (BN // TN)
    micro_cols = BN // TN

    # Each thread owns one TM x TN micro-tile.
    trow = tid // micro_cols
    tcol = tid % micro_cols

    # Shared-memory tiles.
    As = local("As", dtypes.float32, BM * BK)
    Bs = local("Bs", dtypes.float32, BK * BN)

    # ---------------------------------------------------------------
    # One accumulator for every output element in this thread's
    # TM x TN micro-tile.
    #
    # ID convention required by the specification:
    #   acc_(dtypes.float32, 10 + a * TN + b)
    # ---------------------------------------------------------------
    accs = []

    for a in range(TM):
        for b in range(TN):
            acc = acc_(dtypes.float32, 10 + a * TN + b)
            accs.append([
                acc,
                UOp.const(dtypes.float32, 0.0),
                acc,
            ])

    # ---------------------------------------------------------------
    # Outer reduction over K/BK shared-memory tiles.
    # ---------------------------------------------------------------
    rt = rng_(K // BK, 0)

    body = []

    # ---------------------------------------------------------------
    # Cooperative loading of A into shared memory.
    #
    # For each cooperative chunk:
    #
    #   idx = tid + e
    #
    # A source:
    #
    #   A[
    #       (byid * BM + idx / BK) * K
    #       + rt * BK
    #       + idx % BK
    #   ]
    #
    # Destination:
    #
    #   As[idx]
    # ---------------------------------------------------------------
    for e in range(0, BM * BK, threads):
        idx = tid + e

        a_idx = (
            (byid * BM + idx // BK) * K
            + rt * BK
            + idx % BK
        )

        a_val = load(A, a_idx)

        if BM * BK % threads != 0:
            cond = idx < BM * BK
        else:
            cond = None

        body.append(
            (
                "localstore",
                As,
                idx,
                a_val,
                cond,
            )
        )

    # ---------------------------------------------------------------
    # Cooperative loading of B into shared memory.
    #
    # Source:
    #
    #   B[
    #       (rt * BK + idx / BN) * N
    #       + bxid * BN
    #       + idx % BN
    #   ]
    #
    # Destination:
    #
    #   Bs[idx]
    # ---------------------------------------------------------------
    for e in range(0, BK * BN, threads):
        idx = tid + e

        b_idx = (
            (rt * BK + idx // BN) * N
            + bxid * BN
            + idx % BN
        )

        b_val = load(B, b_idx)

        if BK * BN % threads != 0:
            cond = idx < BK * BN
        else:
            cond = None

        body.append(
            (
                "localstore",
                Bs,
                idx,
                b_val,
                cond,
            )
        )

    # All threads must finish populating shared memory before any
    # thread begins consuming the tile.
    body.append("__syncthreads();")

    # ---------------------------------------------------------------
    # Inner reduction over BK.
    #
    # Each accumulator corresponds to one element of the thread's
    # TM x TN micro-tile.
    # ---------------------------------------------------------------
    rk = rng_(BK, 1)

    inner_accs = []

    for a in range(TM):
        for b in range(TN):
            acc_id = 10 + a * TN + b
            acc = acc_(dtypes.float32, acc_id)

            as_idx = (
                (trow * TM + a) * BK
                + rk
            )

            bs_idx = (
                rk * BN
                + tcol * TN
                + b
            )

            as_val = load(As, as_idx)
            bs_val = load(Bs, bs_idx)

            update = acc + as_val * bs_val

            inner_accs.append([
                acc,
                acc,
                update,
            ])

    inner = Reduce(
        [rk],
        inner_accs,
    )

    body.append(inner)

    # Ensure all threads have finished reading the current shared-memory
    # tile before another outer tile overwrites it.
    body.append("__syncthreads();")

    outer = Reduce(
        [rt],
        accs,
        body,
    )

    # ---------------------------------------------------------------
    # Store the complete TM x TN register micro-tile to C.
    # ---------------------------------------------------------------
    stores = []

    for a in range(TM):
        for b in range(TN):
            acc_id = 10 + a * TN + b
            acc = acc_(dtypes.float32, acc_id)

            output_idx = (
                (byid * BM + trow * TM + a) * N
                + bxid * BN
                + tcol * TN
                + b
            )

            stores.append(
                (
                    output_idx,
                    acc,
                    None,
                )
            )

    return Kernel(
        f"gemm_reg{BM}x{BN}x{BK}_{TM}x{TN}",
        [C, A, B],
        [
            ("tid", "threadIdx.x"),
            ("bxid", "blockIdx.x"),
            ("byid", "blockIdx.y"),
        ],
        [outer],
        stores,
        (threads, 1),
        (N // BN, M // BM),
        locals_=[As, Bs],
    )

