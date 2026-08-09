"""TVM Relax frontend: materialize a nnsmith GIR directly into a TVM Relax IRModule.

This is the "TVM frontend + TVM backend" execution mode. The GIR is translated
directly to a `tvm.IRModule` (Relax) — no ONNX / PyTorch intermediate — and the
TVM backend executes that IRModule directly.

Reference: AIFuzzer's TvmRelaxTranslator generates Relax Python code from its
own IR; here we build the IRModule in-process with relax.BlockBuilder instead of
emitting Python source.
"""

import os
import pickle
from os import PathLike
from typing import Dict, List, Optional, Type

import numpy as np

import tvm
from tvm import relax

from nnsmith.abstract.op import AbsOpBase, AbsTensor, Constant, Input
from nnsmith.abstract.tensor import AbsTensor as _AbsTensor
from nnsmith.gir import GraphIR
from nnsmith.logging import TORCH_LOG
from nnsmith.materialize import Model, Oracle
from nnsmith.materialize.tvm.forward import _dtype_str, relax_forward

# set of ops TVM frontend supports (for auto_opset filtering if needed)
# Excluding Reshape due to nnsmith core Z3 constraint issue with -1 dimensions
TVM_REALIZABLE_OPS = [op for op in relax_forward.__globals__["RELAX_FORWARD"].keys()
                      if op.__name__ != 'Reshape']


def _as_int(v):
    """Concretize a possibly-symbolic integer to a Python int."""
    if hasattr(v, "as_long"):
        return v.as_long()
    return int(v)


class TVMModel(Model):
    """A model whose `native_model` is a TVM Relax IRModule built directly from GIR."""

    def __init__(self) -> None:
        super().__init__()
        self.ir: Optional[GraphIR] = None
        self.mod: Optional[tvm.IRModule] = None
        self.input_map: Dict[str, AbsTensor] = {}
        self.output_map: Dict[str, AbsTensor] = {}
        self._weight_map: Dict[str, np.ndarray] = {}  # retval name -> numpy

    @property
    def version(self) -> str:
        return f"tvm{tvm.__version__}"

    # ── GIR -> Relax IRModule ──
    @classmethod
    def from_gir(cls, ir: GraphIR, **kwargs) -> "TVMModel":
        ret = cls()
        ret.ir = ir
        ret.input_map = {name: ir.vars[name] for name in ir.input_var()}
        ret.output_map = {name: ir.vars[name] for name in ir.leaf_var()}
        try:
            ret.mod = ret._build_relax_module(ir)
        except Exception as e:
            # TVM build failure (e.g., invalid reshape shape) — store as a broken model
            # The dtype test will catch this via factory.make_testcase()
            ret.mod = None
            ret._build_error = str(e)
        return ret

    def _build_relax_module(self, ir: GraphIR) -> tvm.IRModule:
        bb = relax.BlockBuilder()
        value_map: Dict[str, relax.Expr] = {}

        # 1. Inputs -> relax.Var (function parameters)
        input_vars = []
        for iname in ir.input_var():
            aten = ir.vars[iname]
            shape = tuple(_as_int(s) for s in aten.shape)
            var = relax.Var(
                str(iname),
                relax.TensorStructInfo(shape=relax.ShapeExpr(list(shape)),
                                       dtype=_dtype_str(aten.dtype)),
            )
            value_map[iname] = var
            input_vars.append(var)

        # 2. Build the single function
        with bb.function("main", input_vars):
            for inst in ir.insts:
                op = inst.iexpr.op
                if isinstance(op, Input):
                    continue  # already handled
                if isinstance(op, Constant):
                    # Constant op: generate a random tensor of the declared shape
                    cname = inst.retval()
                    aten = ir.vars[cname]
                    shape = tuple(_as_int(s) for s in aten.shape)
                    data = self._gen_const(shape, aten.dtype)
                    self._weight_map[cname] = data
                    cexpr = bb.emit(relax.const(data, _dtype_str(aten.dtype)))
                    value_map[cname] = cexpr
                    continue

                # compute op
                inputs = [value_map[arg] for arg in inst.iexpr.args]
                try:
                    out_expr = relax_forward(op, inputs, bb, self._weight_map)
                except NotImplementedError as e:
                    raise NotImplementedError(
                        f"{e} (inst: {inst})"
                    )
                out_expr = bb.emit(out_expr)
                # handle multi-output tuple
                if inst.n_output() == 1:
                    value_map[inst.retval()] = out_expr
                else:
                    for ridx in range(inst.n_output()):
                        value_map[inst.retval(ridx)] = relax.TupleGetItem(
                            out_expr, ridx
                        )

            # outputs
            outputs = [value_map[oname] for oname in ir.leaf_var()]
            if len(outputs) == 1:
                bb.emit_func_output(outputs[0])
            else:
                bb.emit_func_output(relax.Tuple(outputs))

        return bb.get()

    @staticmethod
    def _gen_const(shape: tuple, dtype) -> np.ndarray:
        dt = _dtype_str(dtype)
        if dtype.is_float():
            return np.random.uniform(0.5, 1.5, size=shape).astype(dt)
        elif dtype == DType.bool:
            return np.random.randint(0, 2, size=shape).astype(dt)
        else:
            return np.random.randint(0, 5, size=shape).astype(dt)

    # ── Model interface ──
    @property
    def input_like(self) -> Dict[str, AbsTensor]:
        return self.input_map

    @property
    def output_like(self) -> Dict[str, AbsTensor]:
        return self.output_map

    @property
    def native_model(self) -> tvm.IRModule:
        return self.mod

    def refine_weights(self) -> None:
        # Weights are baked in as relax.const during from_gir; nothing to refine.
        pass

    def make_oracle(self) -> Oracle:
        # Run the IRModule with TVM to produce reference outputs.
        # This is TVM-vs-TVM; used for crash detection / self-consistency.
        # For differential testing, use `cmp.with` to add a reference backend.
        if self.mod is None:
            err = getattr(self, '_build_error', 'Unknown build error')
            return Oracle({}, None, provider=f"tvm[relax] build_error: {err}")
        inputs = {}
        for name, aten in self.input_map.items():
            dt = _dtype_str(aten.dtype)
            shape = tuple(_as_int(s) for s in aten.shape)
            if aten.dtype.is_float():
                inputs[name] = np.random.uniform(0.5, 1.5, size=shape).astype(dt)
            else:
                inputs[name] = np.random.randint(0, 5, size=shape).astype(dt)

        try:
            ex = relax.build(self.mod, target="llvm")
            vm = relax.VirtualMachine(ex, tvm.cpu())
            args = [inputs[name] for name in self.input_map.keys()]
            out = vm["main"](*args)
            outs = self._cvt_output(out)
            output = dict(zip(self.output_map.keys(), outs))
        except Exception:
            # If TVM fails to build/run, fall back to oracle with no output
            # (only crash detection; the backend will surface the real error).
            output = None
        return Oracle(inputs, output, provider="tvm[relax] eager")

    @staticmethod
    def _cvt_output(output):
        if output is None:
            return []
        if isinstance(output, (list, tuple)):
            return [r.numpy() if hasattr(r, "numpy") else r for r in output]
        elif hasattr(output, "numpy"):
            return [output.numpy()]
        elif hasattr(output, "__iter__") and not isinstance(output, (str, bytes)):
            return [r.numpy() if hasattr(r, "numpy") else r for r in output]
        return [output]

    def dump(self, path: PathLike) -> None:
        # Save the source GIR (pickle) so the model can be reloaded.
        gir_path = path.replace(
            self.name_prefix() + self.name_suffix(),
            self.gir_name(),
        )
        with open(gir_path, "wb") as f:
            pickle.dump(self.ir, f)

    @classmethod
    def load(cls, path: PathLike) -> "TVMModel":
        gir_path = path.replace(
            cls.name_prefix() + cls.name_suffix(),
            cls.gir_name(),
        )
        with open(gir_path, "rb") as f:
            ir = pickle.load(f)
        return cls.from_gir(ir)

    @staticmethod
    def name_suffix() -> str:
        return ".tvm"

    @staticmethod
    def gir_name() -> str:
        return "gir.pkl"

    @staticmethod
    def operators() -> List[Type[AbsOpBase]]:
        return TVM_REALIZABLE_OPS

    @staticmethod
    def skip_dtypes():
        # TVM Relax does not support complex64/complex128
        from nnsmith.abstract.dtype import DTYPE_GEN_COMPLEX
        return DTYPE_GEN_COMPLEX

    @property
    def import_libs(self) -> List[str]:
        return ["import tvm"]

    @staticmethod
    def name_prefix() -> str:
        return "model"


from nnsmith.abstract.dtype import DType  # noqa: E402  (used in _gen_const)