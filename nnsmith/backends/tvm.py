import logging
from typing import List

import tvm
from multipledispatch import dispatch

from nnsmith.backends import BackendFactory
from nnsmith.materialize.onnx import ONNXModel

# ── TVM 0.25 compat: tvm.relay removed, use relax instead ──
from tvm import relax as _relax

class _RelayCompat:
    class frontend:
        @staticmethod
        def from_onnx(model, shape_dict, freeze_params=True, **kwargs):
            from tvm.relax.frontend import onnx as _onnx_frontend
            mod = _onnx_frontend.from_onnx(
                model, shape_dict=shape_dict,
                keep_params_in_input=not freeze_params,
            )
            return mod, {}

    class transform:
        @staticmethod
        def InferType():
            return tvm.transform.Sequential([
                _relax.transform.CanonicalizeBindings(),
            ])

    class build_module:
        @staticmethod
        def create_executor(executor_mode, mod, device, target, params, **kwargs):
            exe = _relax.build(mod, target=target, params=params)
            vm = _relax.VirtualMachine(exe, device)
            class _Executor:
                def evaluate(self):
                    def executor(**inputs):
                        import numpy as np
                        cuda_inputs = []
                        for name, val in inputs.items():
                            if isinstance(val, np.ndarray):
                                tvm_tensor = tvm.runtime.empty(
                                    val.shape, val.dtype, device=device
                                )
                                tvm_tensor.copyfrom(val)
                                val = tvm_tensor
                            cuda_inputs.append(val)
                        return vm['main'](*cuda_inputs)
                    return executor
            return _Executor()

relay = _RelayCompat()

logging.getLogger("te_compiler").disabled = True
logging.getLogger("autotvm").disabled = True


def list_eq(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


class TVM(BackendFactory):
    def __init__(
        self,
        target: str = "cpu",
        optmax: bool = True,
        executor: str = "graph",
        **kwargs,
    ) -> None:
        super().__init__(target, optmax, **kwargs)
        # WARNING: setting opt_level 4 sometimes causes false alarms
        # as in this level fast_math is enabled where slight numerical
        # inconsistency is allowed and outputs for UB-input may differ.
        self.opt_level = 4 if optmax else 0
        if target == "cpu":
            self.tvm_target = tvm.target.Target("llvm")
        else:
            tvm_possible_targets = tvm.target.Target.list_kinds()
            assert (
                target in tvm_possible_targets
            ), f"Unknown target {target}. Possible targets are {tvm_possible_targets}"
            self.tvm_target = tvm.target.Target(target)

        self.executor_mode = executor

    def get_device(self):
        dev_cand = self.tvm_target.export()["keys"]
        assert len(dev_cand) > 0, f"No viable device found for {self.tvm_target}"
        if "cuda" in dev_cand:
            return tvm.cuda()
        if "rocm" in dev_cand:
            return tvm.rocm()
        if "cpu" in dev_cand:
            return tvm.cpu()
        return tvm.device(dev_cand[0])

    @property
    def system_name(self) -> str:
        return "tvm"

    @staticmethod
    def cvt_result(output):
        """Pack output tensor(s) into a list"""
        # TVM 0.25: multi-output Relax VM returns tvm_ffi.container.Array
        # (iterable, NOT list/tuple, no .numpy()). Handle like list.
        if output is None:
            return []
        if isinstance(output, (list, tuple)):
            output = [r.numpy() if hasattr(r, 'numpy') else r for r in output]
        elif hasattr(output, 'numpy'):
            output = [output.numpy()]
        elif hasattr(output, '__iter__') and not isinstance(output, (str, bytes)):
            # e.g. tvm_ffi.container.Array
            output = [r.numpy() if hasattr(r, 'numpy') else r for r in output]
        elif hasattr(output, 'shape'):
            output = [output.numpy()]
        return output

    @dispatch(ONNXModel)
    def make_backend(self, model: ONNXModel):
        onnx_model = model.native_model
        shape_dict = {name: aten.shape for name, aten in model.input_like.items()}
        mod, params = relay.frontend.from_onnx(
            onnx_model, shape_dict, freeze_params=True
        )
        mod = relay.transform.InferType()(mod)

        with tvm.transform.PassContext(opt_level=self.opt_level):
            executor = relay.build_module.create_executor(
                self.executor_mode, mod, self.get_device(), self.tvm_target, params
            ).evaluate()

        def closure(inputs):
            output = executor(**inputs)
            output = self.cvt_result(output)
            return dict(zip(model.output_like.keys(), output))

        return closure

    @property
    def import_libs(self) -> List[str]:
        return ["import tvm"]

    @property
    def version(self) -> str:
        return tvm.__version__
