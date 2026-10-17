import unittest
from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import Placeholder, ReLU, Abs, Add, Transpose, BatchNorm2d, Input
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.gir import GraphIR, InstExpr
from nnsmith.materialize import Model
from nnsmith.reducer import DDMin, GraphIRReducer, _remap_state_dict, _slice_to_outputs


class TestReducer(unittest.TestCase):
    def test_ddmin_list(self):
        # Target: list must contain 3 and 7
        def prop(items):
            return 3 in items and 7 in items

        ddmin = DDMin(prop)
        reduced = ddmin.execute([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertEqual(sorted(reduced), [3, 7])

    def test_gir_output_slice(self):
        # Branch 1: ph1 -> ReLU (v1_0)
        # Branch 2: ph2 -> Abs (v3_0)
        ir = GraphIR()
        t = AbsTensor(shape=[2, 2], dtype=DType.float32)
        ph1 = ir.add_inst(InstExpr(Placeholder(ttype=t), []))
        v1 = ir.add_inst(InstExpr(ReLU(), [ph1.retval()]))
        ph2 = ir.add_inst(InstExpr(Placeholder(ttype=t), []))
        v3 = ir.add_inst(InstExpr(Abs(), [ph2.retval()]))

        self.assertEqual(len(ir.leaf_var()), 2)
        sliced = _slice_to_outputs(ir, [v1.retval()])
        self.assertIsNotNone(sliced)
        assert sliced is not None
        self.assertEqual(sliced.leaf_var(), [v1.retval()])
        self.assertEqual(sliced.n_compute_inst(), 1)
        self.assertEqual(type(sliced.insts[1].iexpr.op).__name__, "ReLU")

    def test_gir_reducer_removes_unrelated_nodes(self):
        # Graph: ph -> Abs -> ReLU
        # Property: must retain ReLU
        ir = GraphIR()
        t = AbsTensor(shape=[2, 2], dtype=DType.float32)
        ph = ir.add_inst(InstExpr(Placeholder(ttype=t), []))
        v1 = ir.add_inst(InstExpr(Abs(), [ph.retval()]))
        v2 = ir.add_inst(InstExpr(ReLU(), [v1.retval()]))

        def prop(g: GraphIR) -> bool:
            return any(isinstance(inst.iexpr.op, ReLU) for inst in g.insts)

        reducer = GraphIRReducer(prop)
        res = reducer.reduce(ir)
        self.assertTrue(res.property_preserved)
        # Abs should be wired around or removed, keeping ReLU
        compute_ops = [type(i.iexpr.op).__name__ for i in res.reduced_gir.insts if not isinstance(i.iexpr.op, Placeholder)]
        self.assertEqual(compute_ops, ["ReLU"])
        self.assertEqual(res.reduced_compute_count, 1)
    def test_gir_reducer_promotes_shape_changing_output_to_input(self):
        ir = GraphIR()
        t = AbsTensor(shape=[2, 3], dtype=DType.float32)
        ph = ir.add_inst(InstExpr(Placeholder(ttype=t), []))
        transposed = ir.add_inst(InstExpr(Transpose(), [ph.retval()]))
        ir.add_inst(InstExpr(ReLU(), [transposed.retval()]))

        def prop(g: GraphIR) -> bool:
            return any(isinstance(inst.iexpr.op, ReLU) for inst in g.insts)

        result = GraphIRReducer(prop).reduce(ir)
        self.assertEqual(result.reduced_compute_count, 1)
        self.assertEqual(result.reduced_gir.n_inst(), 2)
        self.assertEqual(type(result.reduced_gir.insts[0].iexpr.op).__name__, "Input")
        self.assertEqual(type(result.reduced_gir.insts[1].iexpr.op).__name__, "ReLU")
        self.assertEqual(
            result.reduced_gir.vars[result.reduced_gir.insts[0].retval()].shape,
            [3, 2],
        )

    def test_state_remap_uses_stable_instruction_identifier(self):
        import copy
        import torch

        original = GraphIR()
        tensor = AbsTensor(shape=[2, 3, 4, 4], dtype=DType.float32)
        input_op = Input(dim=4)
        input_op.abs_tensor = tensor
        placeholder = original.add_inst(InstExpr(input_op, []))
        absolute = original.add_inst(InstExpr(Abs(), [placeholder.retval()]))
        original.add_inst(InstExpr(BatchNorm2d(nfeat=3), [absolute.retval()]))
        original.wellform_repair()

        candidate = copy.deepcopy(original)
        promoted = candidate.insts[1]
        input_op = Input(candidate.vars[promoted.retval()].ndims)
        input_op.abs_tensor = candidate.vars[promoted.retval()]
        promoted.iexpr = InstExpr(input_op, [])
        candidate.insts = candidate.insts[1:]
        candidate.wellform_repair()

        ModelType = Model.init("torch", "cpu")
        original_model = ModelType.from_gir(original)
        with torch.no_grad():
            original_model.torch_model.m2.weight.fill_(2.5)
            original_model.torch_model.m2.running_mean.fill_(1.25)
        original_state = original_model.torch_model.state_dict()

        candidate_model = ModelType.from_gir(candidate)
        expected = candidate_model.torch_model.state_dict()
        remapped = _remap_state_dict(
            original, candidate, original_state, expected.keys()
        )
        candidate_model.torch_model.load_state_dict(remapped, strict=True)
        self.assertTrue(
            torch.equal(candidate_model.torch_model.m1.weight, torch.full((3,), 2.5))
        )
        self.assertTrue(
            torch.equal(
                candidate_model.torch_model.m1.running_mean,
                torch.full((3,), 1.25),
            )
        )


if __name__ == "__main__":
    unittest.main()
