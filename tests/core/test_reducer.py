import unittest
from nnsmith.abstract.dtype import DType
from nnsmith.abstract.op import Placeholder, ReLU, Abs, Add
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.gir import GraphIR, InstExpr
from nnsmith.reducer import DDMin, GraphIRReducer, _slice_to_outputs


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


if __name__ == "__main__":
    unittest.main()
