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
    if dtype.c == "int":
        return str(int(v))

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
        return self.scopes.pop()

    def emit(self, s):
        self.lines.append("  " * len(self.scopes) + s)

    def var(self, u, e):
        name = f"v{self.n}"
        self.n += 1
        self.emit(f"{u.dtype.c} {name} = {e};")
        self.scopes[-1][u] = name
        return name

    def expr(self, u):
        # Reuse an already-rendered expression from the nearest scope.
        for scope in reversed(self.scopes):
            if u in scope:
                return scope[u]

        # Constants are emitted inline.
        if u.op is Ops.CONST:
            return c_lit(u.dtype, u.arg)

        # RANGE and SPECIAL correspond directly to CUDA/index expressions.
        if u.op is Ops.RANGE:
            return f"r{u.arg}"

        if u.op is Ops.SPECIAL:
            return u.arg

        # Accumulators are represented by their fixed symbolic names.
        if u.op is Ops.DEFINE_ACC:
            return f"acc{u.arg}"

        # LOAD(INDEX(base, idx)).
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
            return self.var(
                u,
                f"({self.expr(cond)} ? {self.expr(a)} : {self.expr(b)})",
            )

        # Unary floating-point operations.
        if u.op is Ops.RECIP:
            x = self.expr(u.src[0])
            return self.var(u, f"(1.0f/{x})")

        if u.op is Ops.EXP2:
            x = self.expr(u.src[0])
            return self.var(u, f"exp2f({x})")

        if u.op is Ops.SQRT:
            x = self.expr(u.src[0])
            return self.var(u, f"sqrtf({x})")

        # MAX uses fmaxf for floating-point expressions and max otherwise.
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

