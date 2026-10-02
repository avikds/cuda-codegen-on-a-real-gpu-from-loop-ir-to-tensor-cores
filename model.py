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

    # Hash-consing cache: equal constructions return the exact same object.
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
        # Integer-like dtypes are represented as Python ints.
        # Floating-point dtypes are represented as Python floats.
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
        # Numeric operands are lifted to constants using this UOp's dtype.
        src = tuple(self._as_uop(x, self.dtype) for x in src)

        # Comparisons and logical operations produce bool.
        result_dtype = dtypes.bool if op in (Ops.CMPLT, Ops.AND) else self.dtype

        return UOp(op, result_dtype, src)

    # Arithmetic operators.
    def __add__(self, other):
        return self.alu(Ops.ADD, self, other)

    def __radd__(self, other):
        return self.alu(Ops.ADD, other, self)

    def __mul__(self, other):
        return self.alu(Ops.MUL, self, other)

    def __rmul__(self, other):
        return self.alu(Ops.MUL, other, self)

    def __floordiv__(self, other):
        return self.alu(Ops.IDIV, self, other)

    def __rfloordiv__(self, other):
        return self.alu(Ops.IDIV, other, self)

    def __mod__(self, other):
        return self.alu(Ops.MOD, self, other)

    def __rmod__(self, other):
        return self.alu(Ops.MOD, other, self)

    # Comparison / logical operators.
    def __lt__(self, other):
        return self.alu(Ops.CMPLT, self, other)

    def __and__(self, other):
        return self.alu(Ops.AND, self, other)

    def __rand__(self, other):
        return self.alu(Ops.AND, other, self)

    def maximum(self, other):
        return self.alu(Ops.MAX, self, other)

    def where(self, a, b):
        # The result dtype is the dtype of the selected value 'a'.
        if not isinstance(a, UOp):
            a = UOp.const(self.dtype, a)

        if not isinstance(b, UOp):
            b = UOp.const(a.dtype, b)

        return UOp(Ops.WHERE, a.dtype, (self, a, b))

    def toposort(self):
        """
        Iterative DFS topological sort.

        Dependencies always appear before the UOp that consumes them.
        """
        result = []
        visited = set()

        # (node, expanded)
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()

            if node in visited:
                continue

            if expanded:
                visited.add(node)
                result.append(node)
                continue

            # Visit dependencies first.
            stack.append((node, True))

            # Reverse traversal preserves source order in the final result.
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
    return UOp(Ops.SPECIAL, dtypes.int32, (), name)


def load(p, idx):
    # LOAD(INDEX(pointer, index))
    # A numeric index must itself be represented as a CONST UOp.
    if not isinstance(idx, UOp):
        idx = UOp.const(dtypes.int32, idx)

    index = UOp(Ops.INDEX, p.dtype, (p, idx))
    return UOp(Ops.LOAD, p.dtype, (index,))


def acc_(dtype, i):
    return UOp(Ops.DEFINE_ACC, dtype, (), i)


def local(name, dtype, size):
    # Represents a shared-memory buffer.
    return UOp(Ops.LOCAL, dtype, (), (name, size))

