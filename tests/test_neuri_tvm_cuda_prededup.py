import inspect
import json
import pickle
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import z3

from nnsmith.cli.fuzz import FuzzingLoop
from nnsmith.dedup import DedupMatcher
from nnsmith.dedup.constraints import (
    DtypeExact,
    ExpressionConstraint,
    ValueRangeOutside,
    parse_attr_matcher,
)
from nnsmith.dedup.matcher import PatternMatcher
from nnsmith.dedup.normalizer import _shape_to_list
from nnsmith.dedup.parser import (
    FlowConstraint,
    PatternDef,
    PatternNodeDef,
    PatternValueDef,
)
from nnsmith.dedup.rewrite import _candidates_for, rewrite_gir
from nnsmith.graph_gen import SymbolicGen

def _patch_symbolnet_debug_helpers():
    return patch.multiple(
        'nnsmith.materialize.torch.symbolnet',
        debug_numeric=lambda *_a, **_kw: None,
        check_type=lambda *_a, **_kw: None,
        debug_io=lambda *_a, **_kw: None,
    )


DATA_DIR = Path('/home/xyzboom/Data/maybeBug/2026-09-24/neuri_tvm_cuda_10k')
PATTERNS_FILE = Path('/home/xyzboom/Data/remote-2026-10-01/patterns-neu-compile/neu-compile.json')
SEEDS_FILE = Path('/home/xyzboom/Code/aiCompiler/nnsmith-worktrees/neuri-tvm-prededup/nnsmith/dedup/known_bug_seeds_tvm_cuda.json')
PYTORCH_V5_DIR = Path(
    '/home/xyzboom/Data/maybeBug/2026-10-13/'
    'neuri_pt2_5k_targeted_paired_v5/dedup'
)
PYTORCH_V6_DIR = Path(
    '/home/xyzboom/Data/maybeBug/2026-10-14/'
    'neuri_pt2_5k_targeted_paired_v6/dedup'
)
PYTORCH_V7_DIR = Path(
    '/home/xyzboom/Data/maybeBug/2026-10-14/'
    'neuri_pt2_5k_targeted_paired_v7/dedup'
)

class TestNeuriTvmCudaPrededup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = DedupMatcher(
            pattern_dir=str(PATTERNS_FILE),
            compiler='tvm',
            target='cuda',
            backend='torch',
        )

    def test_pre_dedup_rewrites_completed_gir_locally(self):
        make_testcase_source = inspect.getsource(FuzzingLoop.make_testcase)
        generator_source = inspect.getsource(FuzzingLoop.make_concrete_gir)
        self.assertIn('rewrite_gir', make_testcase_source)
        self.assertIn('self.dedup_matcher.match_occurrences(ir)', make_testcase_source)
        self.assertIn('original GIR retained', make_testcase_source)
        self.assertIn('rewritten_occurrence_keys', make_testcase_source)
        self.assertIn('reduce_input_context', make_testcase_source)
        self.assertNotIn('dedup_matcher=self.dedup_matcher', generator_source)
        self.assertNotIn('resampled without seed change', make_testcase_source)

    def test_tvm_cuda_known_seeds_exist(self):
        self.assertTrue(SEEDS_FILE.exists())
        data = json.loads(SEEDS_FILE.read_text())
        self.assertGreaterEqual(len(data['patterns']), 10)
        self.assertEqual(data['validation']['matched_reports'], 104)

    def test_tvm_cuda_pattern_matching_on_archived_bug(self):
        sample_dir = DATA_DIR / 'bug-seed-100-Symptom.EXCEPTION-Stage.COMPILATION'
        gir_path = sample_dir / 'gir.pkl'
        if not gir_path.exists():
            self.skipTest('Sample GIR file not available')
        with gir_path.open('rb') as f:
            gir = pickle.load(f)
        matched = self.matcher.matches(gir)
        matched_ids = [p.id for p in matched]
        self.assertIn('neu-bind-36', matched_ids)

    def test_tvm_cuda_pattern_rewrite_on_archived_bug(self):
        sample_dir = DATA_DIR / 'bug-seed-100-Symptom.EXCEPTION-Stage.COMPILATION'
        gir_path = sample_dir / 'gir.pkl'
        if not gir_path.exists():
            self.skipTest('Sample GIR file not available')
        with gir_path.open('rb') as f:
            gir = pickle.load(f)
        rewritten = rewrite_gir(gir, self.matcher)
        self.assertIsNotNone(rewritten)
        remaining = self.matcher.matches(rewritten)
        self.assertEqual(len(remaining), 0)

    def test_fp16_reduce_round_rewrite_changes_only_the_matched_node(self):
        gir_path = Path(
            '/home/xyzboom/Data/maybeBug/2026-10-30/'
            'bug_001_PyTorchInductorCPU_seed3916800231/reduced/gir.pkl'
        )
        if not gir_path.exists():
            self.skipTest('Reduced NeuRI GraphIR not available')
        with gir_path.open('rb') as stream:
            gir = pickle.load(stream)

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        occurrences = matcher.match_occurrences(gir)
        self.assertEqual(
            [pattern.id for pattern, _ in occurrences],
            ['pt-inductor-fp16-reducesum-round-rank1'],
        )
        matched_inst_ids = {node['inst_id'] for _, nodes in occurrences for node in nodes}
        before = {
            inst.identifier: (type(inst.iexpr.op).__name__, tuple(inst.iexpr.args), tuple(inst.retvals()))
            for inst in gir.insts
        }

        rewritten = rewrite_gir(gir, matcher)
        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        after_original = {
            inst.identifier: (type(inst.iexpr.op).__name__, tuple(inst.iexpr.args), tuple(inst.retvals()))
            for inst in gir.insts
        }
        self.assertEqual(before, after_original)
        self.assertEqual(matcher.matches(rewritten), [])
        after = {
            inst.identifier: (type(inst.iexpr.op).__name__, tuple(inst.iexpr.args), tuple(inst.retvals()))
            for inst in rewritten.insts
        }
        changed_ids = {inst_id for inst_id in before if before[inst_id] != after[inst_id]}
        self.assertEqual(len(changed_ids), 1)
        self.assertTrue(changed_ids.issubset(matched_inst_ids))
        self.assertEqual(before[next(iter(changed_ids))][0], 'Round')
        self.assertEqual(after[next(iter(changed_ids))][0], 'Floor')

    def test_rewrite_gir_accepts_last_rewrite_at_iteration_limit(self):
        import copy

        pattern = SimpleNamespace(id='single-node-round')

        class Matcher:
            def match_occurrences(self, gir):
                return [
                    (pattern, [{'inst_id': inst.identifier}])
                    for inst in gir.insts
                    if getattr(inst.iexpr.op, '_name', None) == 'Round'
                ]

        gir = _FakeGir(
            [_FakeInst(0, _FakeOp('Round'), ()), _FakeInst(1, _FakeOp('Round'), ())],
            {},
        )

        def rewrite_candidate(original, index, candidate_name):
            rewritten = copy.deepcopy(original)
            rewritten.insts[index].iexpr.op = _FakeOp(candidate_name)
            return rewritten

        with (
            patch('nnsmith.dedup.rewrite._candidates_for', return_value=['Alternative']),
            patch('nnsmith.dedup.rewrite._rewrite_inst', side_effect=rewrite_candidate),
        ):
            rewritten = rewrite_gir(gir, Matcher(), max_candidates=1)

        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        self.assertEqual(
            [inst.iexpr.op._name for inst in rewritten.insts],
            ['Alternative', 'Alternative'],
        )
        self.assertEqual([inst.iexpr.op._name for inst in gir.insts], ['Round', 'Round'])

    def test_rewrite_gir_avoids_cycles_and_uses_next_candidate(self):
        import copy

        patterns = {
            'Round': SimpleNamespace(id='round-pattern'),
            'Alternative': SimpleNamespace(id='alternative-pattern'),
        }

        class Matcher:
            def match_occurrences(self, gir):
                return [
                    (patterns[inst.iexpr.op._name], [{'inst_id': inst.identifier}])
                    for inst in gir.insts
                    if inst.iexpr.op._name in patterns
                ]

        gir = _FakeGir([_FakeInst(0, _FakeOp('Round'), ())], {})

        candidate_batches = iter([
            ['Alternative'],
            ['Round', 'SafeAlternative'],
        ])

        def candidates(_):
            return next(candidate_batches, [])

        def rewrite_candidate(original, index, candidate_name):
            rewritten = copy.deepcopy(original)
            rewritten.insts[index].iexpr.op = _FakeOp(candidate_name)
            return rewritten

        with (
            patch('nnsmith.dedup.rewrite._candidates_for', side_effect=candidates),
            patch('nnsmith.dedup.rewrite._rewrite_inst', side_effect=rewrite_candidate),
        ):
            rewritten = rewrite_gir(gir, Matcher(), max_candidates=2)

        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        self.assertEqual(rewritten.insts[0].iexpr.op._name, 'SafeAlternative')
        self.assertEqual(gir.insts[0].iexpr.op._name, 'Round')

    def test_rewrite_gir_returns_partial_copy_for_remaining_occurrences(self):
        import copy

        patterns = {
            'Round': SimpleNamespace(id='rewritable'),
            'Stubborn': SimpleNamespace(id='unrewritable'),
        }

        class Matcher:
            def match_occurrences(self, gir):
                return [
                    (patterns[inst.iexpr.op._name], [{'inst_id': inst.identifier}])
                    for inst in gir.insts
                    if inst.iexpr.op._name in patterns
                ]

        gir = _FakeGir(
            [_FakeInst(0, _FakeOp('Round'), ()), _FakeInst(1, _FakeOp('Stubborn'), ())],
            {},
        )
        candidate_batches = iter([['SafeAlternative'], []])

        def rewrite_candidate(original, index, candidate_name):
            rewritten = copy.deepcopy(original)
            rewritten.insts[index].iexpr.op = _FakeOp(candidate_name)
            return rewritten

        with (
            patch('nnsmith.dedup.rewrite._candidates_for', side_effect=lambda _: next(candidate_batches, [])),
            patch('nnsmith.dedup.rewrite._rewrite_inst', side_effect=rewrite_candidate),
        ):
            rewritten = rewrite_gir(gir, Matcher())

        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        self.assertEqual([i.iexpr.op._name for i in rewritten.insts], ['SafeAlternative', 'Stubborn'])
        self.assertEqual([pattern.id for pattern, _ in Matcher().match_occurrences(rewritten)], ['unrewritable'])
        self.assertEqual([i.iexpr.op._name for i in gir.insts], ['Round', 'Stubborn'])

    def test_dedup_matcher_exposes_match_and_occurrence_apis(self):
        self.assertTrue(callable(getattr(self.matcher, 'matches', None)))
        self.assertTrue(callable(getattr(self.matcher, 'match_occurrences', None)))
        self.assertTrue(callable(getattr(self.matcher, 'match_first', None)))


class _FakeTensor:
    def __init__(self, shape=(2, 3), dtype='float32'):
        self.shape = shape
        self.dtype = dtype


class _FakeOp:
    def __init__(self, name):
        self._name = name
        self.extra_attrs = {}

    def name(self):
        return self._name


class _FakeExpr:
    def __init__(self, op, args):
        self.op = op
        self.args = args


class _FakeInst:
    def __init__(self, identifier, op, args):
        self.identifier = identifier
        self.iexpr = _FakeExpr(op, args)

    def n_output(self):
        return 1

    def retval(self, index=0):
        return f'v{self.identifier}_{index}'


class _FakeGir:
    def __init__(self, insts, vars):
        self.insts = insts
        self.vars = vars


def _node(op, inputs=(), outputs=()):
    def ref(var_id):
        return {'var_id': var_id, 'shape': [2, 3], 'dtype': 'float32'}

    return {
        'op': op,
        'attrs': {},
        'inputs': [ref(var_id) for var_id in inputs],
        'outputs': [ref(var_id) for var_id in outputs],
        'raw_op': op,
    }


def _flow_pattern():
    return PatternDef(
        id='producer-consumer',
        compiler='tvm',
        target='cuda',
        description=None,
        severity=None,
        nodes=[
            PatternNodeDef('producer', 'PRODUCER', [], ['value'], {}),
            PatternNodeDef('consumer', 'CONSUMER', ['value'], ['result'], {}),
        ],
        values={},
        flow_constraints=[FlowConstraint('producer', 0, 'consumer', 0)],
    )


class TestCandidateParticipatingOnlineMatcher(unittest.TestCase):
    def test_v6_residual_crash_patterns_match_archived_reports(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file),
            compiler='pytorch',
            target='cpu',
            backend='torch',
        )
        samples = {
            'worker_0/run_0/bug-Symptom.EXCEPTION-Stage.EXECUTION-38':
                'pt-inductor-fp16-atan-dual-reduce-sum-duplicate-decl',
            'worker_1/run_0/bug-Symptom.EXCEPTION-Stage.EXECUTION-33':
                'pt-inductor-cpu-int32-matmul-chain-dtype-mismatch',
        }
        if not all((PYTORCH_V6_DIR / rel / 'gir.pkl').exists() for rel in samples):
            self.skipTest('Archived PyTorch v6 GIR files are not available')

        for rel, expected_pattern in samples.items():
            with (PYTORCH_V6_DIR / rel / 'gir.pkl').open('rb') as stream:
                gir = pickle.load(stream)
            self.assertIn(
                expected_pattern,
                {pattern.id for pattern in dedup.matches(gir)},
            )

    def test_byte_argreduce_pattern_matches_archived_reports(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file),
            compiler='pytorch',
            target='cpu',
            backend='torch',
        )
        samples = {
            'worker_0/run_0/bug-Symptom.INCONSISTENCY-Stage.VERIFICATION-104':
                'pt-inductor-cpu-byte-argreduce-tie',
            'worker_0/run_0/bug-Symptom.INCONSISTENCY-Stage.VERIFICATION-105':
                'pt-inductor-cpu-byte-argreduce-tie',
        }
        if not all((PYTORCH_V7_DIR / rel / 'gir.pkl').exists() for rel in samples):
            self.skipTest('Archived PyTorch v7 GIR files are not available')

        for rel, expected_pattern in samples.items():
            with (PYTORCH_V7_DIR / rel / 'gir.pkl').open('rb') as stream:
                gir = pickle.load(stream)
            self.assertIn(
                expected_pattern,
                {pattern.id for pattern in dedup.matches(gir)},
            )

    def test_byte_matmul_pattern_matches_archived_reports(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file),
            compiler='pytorch',
            target='cpu',
            backend='torch',
        )
        sample = Path(
            '/home/xyzboom/Data/maybeBug/2026-10-15/'
            'neuri_pt2_5k_targeted_paired_v9/dedup/worker_0/run_0/'
            'bug-Symptom.INCONSISTENCY-Stage.VERIFICATION-139/gir.pkl'
        )
        if not sample.exists():
            self.skipTest('Archived PyTorch v9 sample not available')
        with sample.open('rb') as stream:
            gir = pickle.load(stream)
        matched_ids = {pattern.id for pattern in dedup.matches(gir)}
        self.assertIn('pt-inductor-cpu-byte-matmul-broadcast-upcast', matched_ids)

    def test_precise_crash_patterns_match_archived_reports(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file),
            compiler='pytorch',
            target='cpu',
            backend='torch',
        )
        samples = {
            'worker_0/run_0/bug-Symptom.EXCEPTION-Stage.EXECUTION-32':
                'pt-inductor-fp16-softmax-squeeze-reduce-max-duplicate-decl',
            'worker_0/run_1/bug-Symptom.EXCEPTION-Stage.EXECUTION-58':
                'pt-inductor-fp16-cos-dual-reduce-min-duplicate-decl',
            'worker_0/run_0/bug-Symptom.EXCEPTION-Stage.EXECUTION-280':
                'pt-inductor-complex-mul-cast-softmax-ge-crash',
            'worker_0/run_0/bug-Symptom.EXCEPTION-Stage.EXECUTION-238':
                'pt-inductor-complex-mul-slice-cast-softmax-ge-crash',
            'worker_0/run_1/bug-Symptom.EXCEPTION-Stage.EXECUTION-51':
                'pt-inductor-complex-where-concat-pointwise-stride-crash',
            'worker_0/run_1/bug-Symptom.EXCEPTION-Stage.EXECUTION-78':
                'pt-inductor-complex-where-pointwise-stride-crash',
            'worker_1/run_1/bug-Symptom.EXCEPTION-Stage.EXECUTION-93':
                'pt-inductor-complex-where-squeeze-pointwise-stride-crash',
            'worker_0/run_1/bug-Symptom.EXCEPTION-Stage.EXECUTION-101':
                'pt-inductor-complex-where-pointwise-stride-input1-crash',
            'worker_0/run_1/bug-Symptom.EXCEPTION-Stage.EXECUTION-10':
                'pt-inductor-reshape-replicate-pad-symbolic-range-crash',
            'worker_1/run_1/bug-Symptom.EXCEPTION-Stage.EXECUTION-60':
                'pt-inductor-complex-transpose-reduce-transpose-mul-stride-assert',
        }
        if not all((PYTORCH_V5_DIR / rel / 'gir.pkl').exists() for rel in samples):
            self.skipTest('Archived PyTorch v5 GIR files are not available')

        self.assertNotIn(
            'pt-inductor-fp16-multi-reduction-decl',
            {pattern.id for pattern in dedup.patterns},
        )
        for relative_path, expected_id in samples.items():
            with (PYTORCH_V5_DIR / relative_path / 'gir.pkl').open('rb') as stream:
                gir = pickle.load(stream)
            self.assertIn(expected_id, {pattern.id for pattern in dedup.matches(gir)})

    def test_single_op_match_must_be_the_candidate(self):
        pattern = PatternDef(
            id='single-producer',
            compiler='tvm',
            target='cuda',
            description=None,
            severity=None,
            nodes=[PatternNodeDef('producer', 'PRODUCER', [], ['value'], {})],
            values={},
        )
        matcher = PatternMatcher([pattern], compiler='tvm', target='cuda')
        stale = _node('PRODUCER', outputs=('v0_0',))
        candidate = _node('UNRELATED', outputs=('candidate',))

        self.assertEqual(matcher.match([stale]), [pattern])
        self.assertEqual(
            matcher.match([stale, candidate], required_node=candidate), []
        )

    def test_stale_prefix_does_not_reject_unrelated_candidate(self):
        pattern = _flow_pattern()
        matcher = PatternMatcher([pattern], compiler='tvm', target='cuda')
        producer = _node('PRODUCER', outputs=('v0_0',))
        consumer = _node('CONSUMER', inputs=('v0_0',), outputs=('v1_0',))
        candidate = _node('UNRELATED', inputs=('v1_0',), outputs=('candidate',))

        self.assertEqual(matcher.match([producer, consumer]), [pattern])
        self.assertEqual(
            matcher.match([producer, consumer, candidate], required_node=candidate),
            [],
        )

    def test_flow_constraints_require_exact_var_identity(self):
        pattern = _flow_pattern()
        matcher = PatternMatcher([pattern], compiler='tvm', target='cuda')
        producer = _node('PRODUCER', outputs=('produced',))
        disconnected = _node('CONSUMER', inputs=('same_shape_other_var',))
        connected = _node('CONSUMER', inputs=('produced',))
        missing_ids = {
            **connected,
            'inputs': [{'shape': [2, 3], 'dtype': 'float32'}],
        }

        self.assertEqual(matcher.match([producer, disconnected]), [])
        self.assertEqual(matcher.match([producer, missing_ids]), [])
        self.assertEqual(matcher.match([producer, connected]), [pattern])

    def test_backward_candidate_is_inserted_before_its_consumer(self):
        pattern = _flow_pattern()
        dedup = DedupMatcher()
        dedup.patterns = [pattern]
        dedup.matcher = PatternMatcher([pattern], compiler='tvm', target='cuda')
        dedup.backend = 'torch'
        tensor = _FakeTensor()
        gir = _FakeGir(
            [_FakeInst(1, _FakeOp('Consumer'), ('v0_0',))],
            {'v0_0': tensor, 'v1_0': tensor},
        )

        matched = dedup.on_node_generated(
            gir,
            _FakeOp('Producer'),
            ['__new_placeholder'],
            [tensor],
            output_vars=['v0_0'],
            input_tensors=[tensor],
        )

        self.assertIs(matched, pattern)
        self.assertEqual(dedup.prevented_count, 1)
        self.assertEqual(dedup.prevented_by_pattern[pattern.id], 1)

    def test_online_fp16_reduce_round_materializes_symbolic_shapes(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        pattern_id = 'pt-inductor-fp16-reducesum-round-rank1'

        def try_candidate(extent):
            dim = z3.Int(f'dedup_reduce_round_extent_{extent}')
            solver = z3.Solver()
            solver.add(dim == extent)
            self.assertEqual(solver.check(), z3.sat)
            mid = _FakeTensor((dim,), 'float16')
            source = _FakeTensor((3, dim), 'float16')
            gir = _FakeGir(
                [_FakeInst(0, _FakeOp('TorchReduceSum'), ('v9_0',))],
                {'v9_0': source, 'v0_0': mid},
            )
            dedup = DedupMatcher(
                pattern_dir=str(pattern_file),
                compiler='pytorch',
                target='cpu',
                backend='torch',
            )
            return dedup.on_node_generated(
                gir,
                _FakeOp('Round'),
                ['v0_0'],
                [_FakeTensor((dim,), 'float16')],
                output_vars=['v1_0'],
                input_tensors=[mid],
                shape_model=solver.model(),
            )

        self.assertEqual(getattr(try_candidate(2), 'id', None), pattern_id)
        self.assertIsNone(try_candidate(1))

        def try_backward(extent):
            dim = z3.Int(f'dedup_reduce_round_backward_extent_{extent}')
            solver = z3.Solver()
            solver.add(dim == extent)
            self.assertEqual(solver.check(), z3.sat)
            mid = _FakeTensor((dim,), 'float16')
            source = _FakeTensor((3, dim), 'float16')
            gir = _FakeGir(
                [_FakeInst(1, _FakeOp('Round'), ('v0_0',))],
                {'v9_0': source, 'v0_0': mid, 'v1_0': mid},
            )
            dedup = DedupMatcher(
                pattern_dir=str(pattern_file),
                compiler='pytorch',
                target='cpu',
                backend='torch',
            )
            return dedup.on_node_generated(
                gir,
                _FakeOp('TorchReduceSum'),
                ['v9_0'],
                [mid],
                output_vars=['v0_0'],
                input_tensors=[source],
                shape_model=solver.model(),
            )

        self.assertEqual(getattr(try_backward(2), 'id', None), pattern_id)
        self.assertIsNone(try_backward(1))

    def test_candidate_with_unrelated_prior_match_is_accepted(self):
        pattern = _flow_pattern()
        dedup = DedupMatcher()
        dedup.patterns = [pattern]
        dedup.matcher = PatternMatcher([pattern], compiler='tvm', target='cuda')
        dedup.backend = 'torch'
        tensor = _FakeTensor()
        gir = _FakeGir(
            [
                _FakeInst(0, _FakeOp('Producer'), ()),
                _FakeInst(1, _FakeOp('Consumer'), ('v0_0',)),
            ],
            {'v0_0': tensor, 'v1_0': tensor},
        )

        matched = dedup.on_node_generated(
            gir, _FakeOp('Unrelated'), ['v1_0'], [tensor]
        )

        self.assertIsNone(matched)
        self.assertEqual(dedup.prevented_count, 0)
        self.assertEqual(dict(dedup.prevented_by_pattern), {})

    def test_value_range_constraints_fail_closed(self):
        matcher = PatternMatcher([], compiler='pytorch', target='cpu')
        value = PatternValueDef(
            id='v_mid',
            ndim=None,
            shape=[],
            dtype=DtypeExact('float16'),
            range=ValueRangeOutside(-65504, 65504),
        )
        base_ref = {'shape': [2, 3], 'dtype': 'float16'}

        self.assertFalse(matcher._check_value(value, base_ref))
        self.assertFalse(
            matcher._check_value(value, {**base_ref, 'range': (-16.0, 16.0)})
        )
        self.assertTrue(
            matcher._check_value(value, {**base_ref, 'range': (-16.0, 70000.0)})
        )

    def test_expression_constraints_fail_closed_on_symbolic_dimensions(self):
        constraint = ExpressionConstraint([-1], 'add', {1})

        self.assertTrue(constraint.matches([2, 1]))
        self.assertFalse(constraint.matches([2, object()]))  # type: ignore[list-item]

    def test_attr_gte_constraint_fails_closed(self):
        constraint = parse_attr_matcher({'$gte': 16})

        self.assertTrue(constraint.matches(16))
        self.assertTrue(constraint.matches(61))
        self.assertFalse(constraint.matches(15))
        self.assertFalse(constraint.matches(object()))

    def test_shape_materialization_handles_mixed_concrete_and_symbolic_dims(self):
        dim = z3.Int('dedup_test_dim')
        unassigned = z3.Int('dedup_test_unassigned_dim')
        solver = z3.Solver()
        solver.add(dim == 17)
        self.assertEqual(solver.check(), z3.sat)

        model = solver.model()
        self.assertEqual(_shape_to_list([1, dim, unassigned], model), [1, 17, None])
        self.assertIsNone(model.get_interp(unassigned.decl()))

    def test_online_matcher_evaluates_shapes_in_private_z3_context(self):
        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu'
        )
        dim = z3.Int('dedup_private_context_dim')
        unassigned = z3.Int('dedup_private_context_unassigned')
        solver = z3.Solver()
        solver.add(dim == 23)
        self.assertEqual(solver.check(), z3.sat)
        model = solver.model()
        original_context = model.ctx
        normalized_models = []

        def capture_normalization(ir, backend='torch', shape_model=None):
            normalized_models.append(shape_model)
            self.assertEqual(_shape_to_list([dim, unassigned], shape_model), [23, None])
            return []

        candidate = {'op': 'ADD', 'attrs': {}, 'inputs': [], 'outputs': [], 'raw_op': 'Add'}
        with (
            patch('nnsmith.dedup.normalizer.normalize_graph', side_effect=capture_normalization),
            patch('nnsmith.dedup.normalizer.normalize_node_dict', return_value=candidate),
            patch.object(dedup.matcher, 'match', return_value=[]),
        ):
            result = dedup.on_node_generated(
                SimpleNamespace(vars={}, insts=[]), object(), [], [], shape_model=model
            )

        self.assertIsNone(result)
        self.assertEqual(len(normalized_models), 1)
        self.assertIsNot(normalized_models[0].ctx, original_context)
        self.assertIsNone(model.get_interp(unassigned.decl()))

    def test_fuzz_log_records_local_rewrite_outcome(self):
        source = inspect.getsource(FuzzingLoop.make_testcase)
        self.assertIn('pattern_matches=', source)
        self.assertIn('rewritten_ops=', source)
        self.assertIn('inserted_ops=', source)
        self.assertIn('rewrite=failed', source)
        self.assertIn('original GIR retained', source)

    def test_generation_never_rejects_a_candidate_for_dedup(self):
        forward_source = inspect.getsource(SymbolicGen.try_forward_insert_at)
        backward_source = inspect.getsource(SymbolicGen.try_occupy_placeholder)
        generator_source = inspect.getsource(FuzzingLoop.make_concrete_gir)
        self.assertNotIn('on_node_generated', forward_source)
        self.assertNotIn('on_node_generated', backward_source)
        self.assertNotIn('dedup_matcher=self.dedup_matcher', generator_source)

    def test_validated_fp16_fused_arith_and_rounding_catalog(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file),
            compiler='pytorch',
            target='cpu',
            backend='torch',
        )
        def _pt_node(op, inputs=(), outputs=(), dtype='float16', shape=(2, 3), attrs=None):
            return {
                'op': op,
                'attrs': attrs or {},
                'inputs': [{'var_id': v, 'shape': list(shape), 'dtype': dtype} for v in inputs],
                'outputs': [{'var_id': v, 'shape': list(shape), 'dtype': dtype} for v in outputs],
                'raw_op': op,
            }

        # The validated historical baseline intentionally keeps this broad rule.
        n0 = _pt_node('ADD', inputs=['in0', 'in1'], outputs=['v_mid1'])
        n1 = _pt_node('MULTIPLY', inputs=['other0', 'other1'], outputs=['v_out1'])
        matched = dedup.matcher.match([n0, n1])
        fused_ids = {p.id for p in matched if 'fused-arith-chain' in p.id}
        self.assertIn('pt-inductor-fp16-fused-arith-chain', fused_ids)

        # 2. Connected arithmetic nodes: must match fused arith chain
        n2 = _pt_node('MULTIPLY', inputs=['v_mid1', 'other1'], outputs=['v_out2'])
        matched = dedup.matcher.match([n0, n2])
        fused_ids = {p.id for p in matched if 'fused-arith-chain' in p.id}
        self.assertIn('pt-inductor-fp16-fused-arith-chain', fused_ids)

        # 3. The validated historical rounding rule is likewise broad.
        n_round_dis = _pt_node('FLOOR', inputs=['other_unrelated'], outputs=['v_floor'])
        matched = dedup.matcher.match([n0, n_round_dis])
        round_ids = {p.id for p in matched if p.id == 'pt-inductor-fp16-arith-rounding'}
        self.assertIn('pt-inductor-fp16-arith-rounding', round_ids)

        # 4. Connected arithmetic and rounding: must match rounding
        n_round_con = _pt_node('FLOOR', inputs=['v_mid1'], outputs=['v_floor'])
        matched = dedup.matcher.match([n0, n_round_con])
        round_ids = {p.id for p in matched if p.id == 'pt-inductor-fp16-arith-rounding'}
        self.assertIn('pt-inductor-fp16-arith-rounding', round_ids)

    def test_validated_int_reduce_minmax_catalog(self):
        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr

        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file),
            compiler='pytorch',
            target='cpu',
            backend='torch',
        )
        pattern_ids = {
            'pt-inductor-int16-abs-reduce-minmax',
            'pt-inductor-int16-abs-squeeze-reduce-minmax',
        }

        def _make_gir(with_squeeze=False, axis_len=4):
            gir = GraphIR()
            shape = [2, 1, axis_len] if with_squeeze else [2, axis_len]
            input_op = AOP.Input(len(shape))
            input_op.abs_tensor = AbsTensor(shape, DType.int16)
            source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
            abs_op = AOP.Abs()
            abs_inputs = [gir.vars[source]]
            abs_outputs = abs_op.type_transfer(abs_inputs)
            abs_op.bind_input_like(abs_inputs)
            abs_op.bind_output_like(abs_outputs)
            abs_inst = gir.add_inst(InstExpr(abs_op, (source,)))
            source = abs_inst.retval()
            if with_squeeze:
                squeeze_op = AOP.Squeeze()
                squeeze_op.extra_attrs['reduce_dim'] = 1
                squeeze_inputs = [gir.vars[source]]
                squeeze_outputs = squeeze_op.type_transfer(squeeze_inputs)
                squeeze_op.bind_input_like(squeeze_inputs)
                squeeze_op.bind_output_like(squeeze_outputs)
                source = gir.add_inst(InstExpr(squeeze_op, (source,))).retval()
            reduce_op = AOP.ReduceMin()
            reduce_op.extra_attrs['reduce_dim'] = 1
            reduce_inputs = [gir.vars[source]]
            reduce_outputs = reduce_op.type_transfer(reduce_inputs)
            reduce_op.bind_input_like(reduce_inputs)
            reduce_op.bind_output_like(reduce_outputs)
            gir.add_inst(InstExpr(reduce_op, (source,)))
            gir.assert_wellform()
            return gir

        # Isolated reductions and singleton axes are outside the confirmed defect.
        self.assertNotIn(
            'pt-inductor-int16-abs-reduce-minmax',
            {p.id for p, _ in dedup.match_occurrences(_make_gir(axis_len=1))},
        )
        for with_squeeze, expected_id in (
            (False, 'pt-inductor-int16-abs-reduce-minmax'),
            (True, 'pt-inductor-int16-abs-squeeze-reduce-minmax'),
        ):
            matched_ids = {
                p.id for p, _ in dedup.match_occurrences(_make_gir(with_squeeze))
            }
            self.assertIn(expected_id, matched_ids)
            self.assertTrue(matched_ids <= pattern_ids)

    def test_narrow_int_reduce_minmax_widens_input_and_preserves_semantics(self):
        import torch

        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.dedup import DedupMatcher
        from nnsmith.dedup.rewrite import rewrite_gir
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.symbolnet import SymbolNet

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        values = [[-32768, -5, 5, 1], [7, -7, 0, 32767]]
        pattern_ids = {
            'pt-inductor-int16-abs-reduce-minmax',
            'pt-inductor-int16-abs-squeeze-reduce-minmax',
        }
        for with_squeeze in (False, True):
            x = torch.tensor(values).short()
            if with_squeeze:
                x = x.reshape(2, 1, 4)
            for reduce_cls in (AOP.ReduceMin, AOP.ReduceMax):
                with self.subTest(with_squeeze=with_squeeze, op=reduce_cls.__name__):
                    gir = GraphIR()
                    input_shape = [2, 1, 4] if with_squeeze else [2, 4]
                    input_op = AOP.Input(len(input_shape))
                    input_op.abs_tensor = AbsTensor(input_shape, DType.int16)
                    source = gir.add_inst(
                        InstExpr(input_op, cast(tuple[str], ()))
                    ).retval()
                    abs_op = AOP.Abs()
                    abs_inputs = [gir.vars[source]]
                    abs_outputs = abs_op.type_transfer(abs_inputs)
                    abs_op.bind_input_like(abs_inputs)
                    abs_op.bind_output_like(abs_outputs)
                    source = gir.add_inst(InstExpr(abs_op, (source,))).retval()
                    if with_squeeze:
                        squeeze_op = AOP.Squeeze()
                        squeeze_op.extra_attrs['reduce_dim'] = 1
                        squeeze_inputs = [gir.vars[source]]
                        squeeze_outputs = squeeze_op.type_transfer(squeeze_inputs)
                        squeeze_op.bind_input_like(squeeze_inputs)
                        squeeze_op.bind_output_like(squeeze_outputs)
                        source = gir.add_inst(InstExpr(squeeze_op, (source,))).retval()
                    reduce_op = reduce_cls()
                    reduce_op.extra_attrs['reduce_dim'] = 1
                    reduce_inputs = [gir.vars[source]]
                    reduce_outputs = reduce_op.type_transfer(reduce_inputs)
                    reduce_op.bind_input_like(reduce_inputs)
                    reduce_op.bind_output_like(reduce_outputs)
                    reduce_inst = gir.add_inst(InstExpr(reduce_op, (source,)))
                    gir.assert_wellform()

                    self.assertTrue(
                        pattern_ids & {p.id for p, _ in matcher.match_occurrences(gir)}
                    )
                    rewritten = rewrite_gir(gir, matcher)
                    self.assertIsNotNone(rewritten)
                    assert rewritten is not None
                    rewritten.assert_wellform()
                    rewritten_reduce = rewritten.find_inst_by_id(reduce_inst.identifier)
                    assert rewritten_reduce is not None
                    self.assertEqual(
                        rewritten.vars[rewritten_reduce.iexpr.args[0]].dtype, DType.int64
                    )
                    self.assertNotIn(
                        'pt-inductor-int16-abs-reduce-minmax',
                        {p.id for p, _ in matcher.match_occurrences(rewritten)},
                    )
                    self.assertNotIn(
                        'pt-inductor-int16-abs-squeeze-reduce-minmax',
                        {p.id for p, _ in matcher.match_occurrences(rewritten)},
                    )
                    expected = SymbolNet(gir)(x)[0]
                    self.assertEqual(expected.dtype, x.dtype)
                    self.assertTrue(torch.equal(expected, SymbolNet(rewritten)(x)[0]))
                    torch._dynamo.reset()
                    with _patch_symbolnet_debug_helpers():
                        compiled = torch.compile(
                            SymbolNet(rewritten), backend='inductor', fullgraph=True
                        )
                        compiled_output = compiled(x)[0]
                        self.assertTrue(
                            torch.equal(expected, compiled_output),
                            msg=f'expected={expected}, compiled={compiled_output}',
                        )

    def test_byte_argreduce_pattern_inserts_semantics_preserving_int64_cast(self):
        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.dedup.rewrite import rewrite_gir
        from nnsmith.gir import GraphIR, InstExpr

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        for dtype, argreduce_cls in ((DType.int8, AOP.ArgMin), (DType.uint8, AOP.ArgMax)):
            below_limit = GraphIR()
            below_input_op = AOP.Input(2)
            below_input_op.abs_tensor = AbsTensor([5, 15], dtype)
            below_source = below_limit.add_inst(
                InstExpr(below_input_op, cast(tuple[str], ()))
            ).retval()
            below_argreduce = argreduce_cls()
            below_argreduce.extra_attrs['reduce_dim'] = 1
            below_limit.add_inst(InstExpr(below_argreduce, (below_source,)))
            below_limit.assert_wellform()
            self.assertNotIn(
                'pt-inductor-cpu-byte-argreduce-tie',
                {pattern.id for pattern, _ in matcher.match_occurrences(below_limit)},
            )

            gir = GraphIR()
            input_op = AOP.Input(2)
            input_op.abs_tensor = AbsTensor([5, 16], dtype)
            source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
            argreduce = argreduce_cls()
            argreduce.extra_attrs['reduce_dim'] = 1
            argreduce_inst = gir.add_inst(InstExpr(argreduce, (source,)))
            gir.assert_wellform()

            self.assertIn(
                'pt-inductor-cpu-byte-argreduce-tie',
                {pattern.id for pattern, _ in matcher.match_occurrences(gir)},
            )
            rewritten = rewrite_gir(gir, matcher)
            self.assertIsNotNone(rewritten)
            assert rewritten is not None
            rewritten.assert_wellform()
            self.assertEqual(
                [type(inst.iexpr.op).__name__ for inst in rewritten.insts],
                ['Input', 'CastI64', type(argreduce).__name__],
            )
            rewritten_argreduce = rewritten.find_inst_by_id(argreduce_inst.identifier)
            assert rewritten_argreduce is not None
            widened_input = rewritten_argreduce.iexpr.args[0]
            self.assertEqual(list(rewritten.vars[widened_input].shape), [5, 16])
            self.assertEqual(rewritten.vars[widened_input].dtype, DType.int64)
            self.assertEqual(list(rewritten.vars[rewritten_argreduce.retval()].shape), [5])
            self.assertEqual(rewritten.vars[rewritten_argreduce.retval()].dtype, DType.int64)
            self.assertEqual(rewritten_argreduce.iexpr.op.extra_attrs['reduce_dim'], 1)
            self.assertEqual([type(inst.iexpr.op).__name__ for inst in gir.insts], ['Input', type(argreduce).__name__])
            self.assertIn('pt-inductor-cpu-byte-argreduce-tie', {
                p.id for p, _ in matcher.match_occurrences(gir)
            })
            self.assertNotIn('pt-inductor-cpu-byte-argreduce-tie', {
                p.id for p, _ in matcher.match_occurrences(rewritten)
            })

    def test_int16_abs_argreduce_rewrite_widens_and_preserves_semantics(self):
        import torch

        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.symbolnet import SymbolNet

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        values = torch.tensor(
            [
                [-32768, -5, 5, 1, 1, 0, 2, -2],
                [7, -7, 0, -32768, 32767, 2, -2, 0],
            ],
            dtype=torch.int16,
        )
        for argreduce_cls in (AOP.ArgMin, AOP.ArgMax):
            gir = GraphIR()
            input_op = AOP.Input(2)
            input_op.abs_tensor = AbsTensor([2, 8], DType.int16)
            source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
            abs_op = AOP.Abs()
            abs_inputs = [gir.vars[source]]
            abs_outputs = abs_op.type_transfer(abs_inputs)
            abs_op.bind_input_like(abs_inputs)
            abs_op.bind_output_like(abs_outputs)
            abs_inst = gir.add_inst(InstExpr(abs_op, (source,)))
            argreduce_op = argreduce_cls()
            argreduce_op.extra_attrs['reduce_dim'] = 1
            argreduce_inputs = [gir.vars[abs_inst.retval()]]
            argreduce_outputs = argreduce_op.type_transfer(argreduce_inputs)
            argreduce_op.bind_input_like(argreduce_inputs)
            argreduce_op.bind_output_like(argreduce_outputs)
            argreduce_inst = gir.add_inst(InstExpr(argreduce_op, (abs_inst.retval(),)))
            gir.assert_wellform()

            self.assertIn(
                'pt-inductor-int16-argminmax',
                {p.id for p, _ in matcher.match_occurrences(gir)},
            )
            rewritten = rewrite_gir(gir, matcher)
            self.assertIsNotNone(rewritten)
            assert rewritten is not None
            rewritten.assert_wellform()
            self.assertEqual(
                [type(inst.iexpr.op).__name__ for inst in rewritten.insts],
                ['Input', 'CastI64', 'Abs', 'Cast', 'CastI64', type(argreduce_op).__name__],
            )
            rewritten_argreduce = rewritten.find_inst_by_id(argreduce_inst.identifier)
            assert rewritten_argreduce is not None
            self.assertEqual(
                rewritten.vars[rewritten_argreduce.iexpr.args[0]].dtype, DType.int64
            )
            self.assertEqual(
                rewritten.vars[rewritten_argreduce.retval()].dtype, DType.int64
            )
            self.assertTrue(torch.equal(SymbolNet(gir)(values)[0], SymbolNet(rewritten)(values)[0]))
            if argreduce_cls is AOP.ArgMin:
                with _patch_symbolnet_debug_helpers():
                    compiled = torch.compile(SymbolNet(rewritten), backend='inductor', fullgraph=True)
                    self.assertTrue(torch.equal(SymbolNet(gir)(values)[0], compiled(values)[0]))

    def test_int16_concat_argreduce_rewrite_widens_and_preserves_semantics(self):
        import torch

        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.symbolnet import SymbolNet

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        values = torch.tensor(
            [[-32768, -5, 5, 1, 1, 0, 2, -2], [7, -7, 0, -32768, 32767, 2, -2, 0]],
            dtype=torch.int16,
        )
        for argreduce_cls in (AOP.ArgMin, AOP.ArgMax):
            gir = GraphIR()
            input_op = AOP.Input(2)
            input_op.abs_tensor = AbsTensor([2, 8], DType.int16)
            source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
            concat_op = AOP.Concat1()
            concat_op.extra_attrs['axis'] = 1
            concat_inputs = [gir.vars[source]]
            concat_outputs = concat_op.type_transfer(concat_inputs)
            concat_op.bind_input_like(concat_inputs)
            concat_op.bind_output_like(concat_outputs)
            concat_inst = gir.add_inst(InstExpr(concat_op, (source,)))
            argreduce_op = argreduce_cls()
            argreduce_op.extra_attrs['reduce_dim'] = 1
            argreduce_inputs = [gir.vars[concat_inst.retval()]]
            argreduce_outputs = argreduce_op.type_transfer(argreduce_inputs)
            argreduce_op.bind_input_like(argreduce_inputs)
            argreduce_op.bind_output_like(argreduce_outputs)
            argreduce_inst = gir.add_inst(InstExpr(argreduce_op, (concat_inst.retval(),)))
            gir.assert_wellform()

            self.assertIn(
                'pt-inductor-int16-argminmax',
                {p.id for p, _ in matcher.match_occurrences(gir)},
            )
            rewritten = rewrite_gir(gir, matcher)
            self.assertIsNotNone(rewritten)
            assert rewritten is not None
            rewritten.assert_wellform()
            self.assertEqual(
                [type(inst.iexpr.op).__name__ for inst in rewritten.insts],
                ['Input', 'Concat1', 'CastI64', type(argreduce_op).__name__],
            )
            rewritten_argreduce = rewritten.find_inst_by_id(argreduce_inst.identifier)
            assert rewritten_argreduce is not None
            self.assertEqual(
                rewritten.vars[rewritten_argreduce.iexpr.args[0]].dtype, DType.int64
            )
            self.assertTrue(torch.equal(SymbolNet(gir)(values)[0], SymbolNet(rewritten)(values)[0]))

    def test_int16_concat_after_abs_argreduce_and_clip_preserve_wrap(self):
        import torch

        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.symbolnet import SymbolNet

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        values = torch.tensor([[-32768, -5, 5, 1], [7, -7, 0, -32768]], dtype=torch.int16)
        gir = GraphIR()
        input_op = AOP.Input(2)
        input_op.abs_tensor = AbsTensor([2, 4], DType.int16)
        source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
        abs_op = AOP.Abs()
        abs_inputs = [gir.vars[source]]
        abs_outputs = abs_op.type_transfer(abs_inputs)
        abs_op.bind_input_like(abs_inputs)
        abs_op.bind_output_like(abs_outputs)
        abs_inst = gir.add_inst(InstExpr(abs_op, (source,)))
        concat_op = AOP.Concat1()
        concat_op.extra_attrs['axis'] = 1
        concat_inputs = [gir.vars[abs_inst.retval()]]
        concat_outputs = concat_op.type_transfer(concat_inputs)
        concat_op.bind_input_like(concat_inputs)
        concat_op.bind_output_like(concat_outputs)
        concat_inst = gir.add_inst(InstExpr(concat_op, (abs_inst.retval(),)))
        clip_op = AOP.Clip()
        clip_inputs = [gir.vars[concat_inst.retval()]]
        clip_outputs = clip_op.type_transfer(clip_inputs)
        clip_op.bind_input_like(clip_inputs)
        clip_op.bind_output_like(clip_outputs)
        clip_inst = gir.add_inst(InstExpr(clip_op, (concat_inst.retval(),)))
        argreduce_op = AOP.ArgMin()
        argreduce_op.extra_attrs['reduce_dim'] = 1
        argreduce_inputs = [gir.vars[concat_inst.retval()]]
        argreduce_outputs = argreduce_op.type_transfer(argreduce_inputs)
        argreduce_op.bind_input_like(argreduce_inputs)
        argreduce_op.bind_output_like(argreduce_outputs)
        argreduce_inst = gir.add_inst(InstExpr(argreduce_op, (concat_inst.retval(),)))
        gir.assert_wellform()

        rewritten = rewrite_gir(gir, matcher)
        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        rewritten.assert_wellform()
        names = [type(inst.iexpr.op).__name__ for inst in rewritten.insts]
        self.assertIn('Abs', names)
        self.assertIn('Clip', names)
        rewritten_clip = rewritten.find_inst_by_id(clip_inst.identifier)
        assert rewritten_clip is not None
        self.assertEqual(rewritten.vars[rewritten_clip.iexpr.args[0]].dtype, DType.int64)
        self.assertEqual(rewritten.vars[rewritten_clip.retval()].dtype, DType.int64)
        rewritten_argreduce = rewritten.find_inst_by_id(argreduce_inst.identifier)
        assert rewritten_argreduce is not None
        self.assertEqual(rewritten.vars[rewritten_argreduce.iexpr.args[0]].dtype, DType.int64)
        expected = SymbolNet(gir)(values)
        rewritten_eager = SymbolNet(rewritten)(values)
        self.assertEqual(len(expected), len(rewritten_eager))
        for expected_value, rewritten_value in zip(expected, rewritten_eager):
            self.assertTrue(torch.equal(expected_value, rewritten_value))
        with _patch_symbolnet_debug_helpers():
            compiled = torch.compile(SymbolNet(rewritten), backend='inductor', fullgraph=True)
            compiled_outputs = compiled(values)
        self.assertEqual(len(expected), len(compiled_outputs))
        for expected_value, compiled_value in zip(expected, compiled_outputs):
            self.assertTrue(torch.equal(expected_value, compiled_value))

    def test_int16_argreduce_rewrites_abs_chain_after_pointwise_add(self):
        import torch

        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.symbolnet import SymbolNet

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        values = torch.tensor(
            [[-16384, 10000, -12000, 2], [16384, -16384, 1, -1]], dtype=torch.int16
        )
        gir = GraphIR()
        input_op = AOP.Input(2)
        input_op.abs_tensor = AbsTensor([2, 4], DType.int16)
        source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
        add_op = AOP.Add()
        add_inputs = [gir.vars[source], gir.vars[source]]
        add_outputs = add_op.type_transfer(add_inputs)
        add_op.bind_input_like(add_inputs)
        add_op.bind_output_like(add_outputs)
        add_inst = gir.add_inst(InstExpr(add_op, (source, source)))
        abs1_op = AOP.Abs()
        abs1_inputs = [gir.vars[add_inst.retval()]]
        abs1_outputs = abs1_op.type_transfer(abs1_inputs)
        abs1_op.bind_input_like(abs1_inputs)
        abs1_op.bind_output_like(abs1_outputs)
        abs1_inst = gir.add_inst(InstExpr(abs1_op, (add_inst.retval(),)))
        abs2_op = AOP.Abs()
        abs2_inputs = [gir.vars[abs1_inst.retval()]]
        abs2_outputs = abs2_op.type_transfer(abs2_inputs)
        abs2_op.bind_input_like(abs2_inputs)
        abs2_op.bind_output_like(abs2_outputs)
        abs2_inst = gir.add_inst(InstExpr(abs2_op, (abs1_inst.retval(),)))
        concat_op = AOP.Concat1()
        concat_op.extra_attrs['axis'] = 1
        concat_inputs = [gir.vars[abs2_inst.retval()]]
        concat_outputs = concat_op.type_transfer(concat_inputs)
        concat_op.bind_input_like(concat_inputs)
        concat_op.bind_output_like(concat_outputs)
        concat_inst = gir.add_inst(InstExpr(concat_op, (abs2_inst.retval(),)))
        argreduce_op = AOP.ArgMin()
        argreduce_op.extra_attrs['reduce_dim'] = 1
        argreduce_inputs = [gir.vars[concat_inst.retval()]]
        argreduce_outputs = argreduce_op.type_transfer(argreduce_inputs)
        argreduce_op.bind_input_like(argreduce_inputs)
        argreduce_op.bind_output_like(argreduce_outputs)
        argreduce_inst = gir.add_inst(InstExpr(argreduce_op, (concat_inst.retval(),)))
        gir.assert_wellform()

        rewritten = rewrite_gir(gir, matcher)
        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        rewritten.assert_wellform()
        names = [type(inst.iexpr.op).__name__ for inst in rewritten.insts]
        self.assertEqual(names.count('Add'), 1)
        self.assertEqual(names.count('Abs'), 2)
        self.assertEqual(names.count('CastI64'), 3)
        rewritten_add = rewritten.find_inst_by_id(add_inst.identifier)
        rewritten_argreduce = rewritten.find_inst_by_id(argreduce_inst.identifier)
        assert rewritten_add is not None and rewritten_argreduce is not None
        self.assertEqual(rewritten.vars[rewritten_add.retval()].dtype, DType.int16)
        self.assertEqual(rewritten.vars[rewritten_argreduce.iexpr.args[0]].dtype, DType.int64)
        expected = SymbolNet(gir)(values)
        rewritten_eager = SymbolNet(rewritten)(values)
        self.assertTrue(torch.equal(expected[0], rewritten_eager[0]))
        with _patch_symbolnet_debug_helpers():
            compiled = torch.compile(SymbolNet(rewritten), backend='inductor', fullgraph=True)
            compiled_output = compiled(values)[0]
        self.assertTrue(torch.equal(expected[0], compiled_output))

    def test_int16_flatten_argreduce_rewrite_preserves_wrap(self):
        import torch

        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.dialect import Flatten
        from nnsmith.materialize.torch.symbolnet import SymbolNet

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        values = torch.tensor([[-32768, -5, 5, 1], [7, -7, 0, 32767]], dtype=torch.int16)
        gir = GraphIR()
        input_op = AOP.Input(2)
        input_op.abs_tensor = AbsTensor([2, 4], DType.int16)
        source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
        abs_op = AOP.Abs()
        abs_inputs = [gir.vars[source]]
        abs_outputs = abs_op.type_transfer(abs_inputs)
        abs_op.bind_input_like(abs_inputs)
        abs_op.bind_output_like(abs_outputs)
        abs_inst = gir.add_inst(InstExpr(abs_op, (source,)))
        flatten_op = Flatten()
        flatten_inputs = [gir.vars[abs_inst.retval()]]
        flatten_outputs = flatten_op.type_transfer(flatten_inputs)
        flatten_op.bind_input_like(flatten_inputs)
        flatten_op.bind_output_like(flatten_outputs)
        flatten_inst = gir.add_inst(InstExpr(flatten_op, (abs_inst.retval(),)))
        argreduce_op = AOP.ArgMin()
        argreduce_op.extra_attrs['reduce_dim'] = 0
        argreduce_inputs = [gir.vars[flatten_inst.retval()]]
        argreduce_outputs = argreduce_op.type_transfer(argreduce_inputs)
        argreduce_op.bind_input_like(argreduce_inputs)
        argreduce_op.bind_output_like(argreduce_outputs)
        argreduce_inst = gir.add_inst(InstExpr(argreduce_op, (flatten_inst.retval(),)))
        gir.assert_wellform()

        self.assertIn(
            'pt-inductor-int16-argminmax',
            {pattern.id for pattern, _ in matcher.match_occurrences(gir)},
        )
        rewritten = rewrite_gir(gir, matcher)
        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        rewritten.assert_wellform()
        rewritten_argreduce = rewritten.find_inst_by_id(argreduce_inst.identifier)
        assert rewritten_argreduce is not None
        self.assertEqual(
            rewritten.vars[rewritten_argreduce.iexpr.args[0]].dtype, DType.int64
        )
        expected = SymbolNet(gir)(values)[0]
        self.assertTrue(torch.equal(expected, SymbolNet(rewritten)(values)[0]))
        with _patch_symbolnet_debug_helpers():
            compiled = torch.compile(SymbolNet(rewritten), backend='inductor', fullgraph=True)
            self.assertTrue(torch.equal(expected, compiled(values)[0]))

    def test_int16_argreduce_rewrite_rejects_unvalidated_neg_producer(self):
        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        gir = GraphIR()
        input_op = AOP.Input(2)
        input_op.abs_tensor = AbsTensor([2, 8], DType.int16)
        source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
        neg_op = AOP.Neg()
        neg_inputs = [gir.vars[source]]
        neg_outputs = neg_op.type_transfer(neg_inputs)
        neg_op.bind_input_like(neg_inputs)
        neg_op.bind_output_like(neg_outputs)
        neg_inst = gir.add_inst(InstExpr(neg_op, (source,)))
        argreduce_op = AOP.ArgMin()
        argreduce_op.extra_attrs['reduce_dim'] = 1
        argreduce_inputs = [gir.vars[neg_inst.retval()]]
        argreduce_outputs = argreduce_op.type_transfer(argreduce_inputs)
        argreduce_op.bind_input_like(argreduce_inputs)
        argreduce_op.bind_output_like(argreduce_outputs)
        gir.add_inst(InstExpr(argreduce_op, (neg_inst.retval(),)))
        gir.assert_wellform()

        self.assertIn(
            'pt-inductor-int16-argminmax',
            {p.id for p, _ in matcher.match_occurrences(gir)},
        )
        self.assertIsNone(rewrite_gir(gir, matcher))
        self.assertEqual(
            [type(inst.iexpr.op).__name__ for inst in gir.insts],
            ['Input', 'Neg', 'ArgMin'],
        )
        gir.assert_wellform()

    def test_int_abs_clip_rewrite_preserves_narrow_wrap_and_semantics(self):
        import torch

        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.symbolnet import SymbolNet

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        for dtype, values in (
            (DType.int8, [-128, -2, -1, 0, 1, 2, 127]),
            (DType.int16, [-32768, -2, -1, 0, 1, 2, 32767]),
        ):
            gir = GraphIR()
            input_op = AOP.Input(3)
            input_op.abs_tensor = AbsTensor([20, 2, 54], dtype)
            source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
            abs_op = AOP.Abs()
            abs_inputs = [gir.vars[source]]
            abs_outputs = abs_op.type_transfer(abs_inputs)
            abs_op.bind_input_like(abs_inputs)
            abs_op.bind_output_like(abs_outputs)
            abs_inst = gir.add_inst(InstExpr(abs_op, (source,)))
            clip_op = AOP.Clip()
            clip_inputs = [gir.vars[abs_inst.retval()]]
            clip_outputs = clip_op.type_transfer(clip_inputs)
            clip_op.bind_input_like(clip_inputs)
            clip_op.bind_output_like(clip_outputs)
            clip_inst = gir.add_inst(InstExpr(clip_op, (abs_inst.retval(),)))
            gir.assert_wellform()

            self.assertIn(
                'pt-inductor-int-clip-minmax',
                {p.id for p, _ in matcher.match_occurrences(gir)},
            )
            rewritten = rewrite_gir(gir, matcher)
            self.assertIsNotNone(rewritten)
            assert rewritten is not None
            rewritten.assert_wellform()
            self.assertEqual(
                [type(inst.iexpr.op).__name__ for inst in rewritten.insts],
                ['Input', 'CastI64', 'Abs', 'Cast', 'CastI64', 'Clip', 'Cast'],
            )
            rewritten_clip = rewritten.find_inst_by_id(clip_inst.identifier)
            assert rewritten_clip is not None
            self.assertEqual(
                rewritten.vars[rewritten_clip.iexpr.args[0]].dtype, DType.int64
            )
            self.assertEqual(rewritten.vars[rewritten_clip.retval()].dtype, DType.int64)
            output = rewritten.leaf_var()
            self.assertEqual(len(output), 1)
            self.assertEqual(rewritten.vars[output[0]].dtype, dtype)
            self.assertEqual(list(rewritten.vars[output[0]].shape), [20, 2, 54])
            self.assertNotIn(
                'pt-inductor-int-clip-minmax',
                {p.id for p, _ in matcher.match_occurrences(rewritten)},
            )

            n = 20 * 2 * 54
            values_tensor = torch.tensor(values, dtype=dtype.torch())
            x = values_tensor.repeat((n + len(values) - 1) // len(values))[:n].reshape(20, 2, 54)
            original_output = SymbolNet(gir)(x)[0]
            rewritten_output = SymbolNet(rewritten)(x)[0]
            self.assertTrue(torch.equal(original_output, rewritten_output))
            with _patch_symbolnet_debug_helpers():
                compiled = torch.compile(SymbolNet(rewritten), backend='inductor', fullgraph=True)
                self.assertTrue(torch.equal(original_output, compiled(x)[0]))

    def test_int8_neg_abs_clip_rewrite_preserves_wrap_and_semantics(self):
        import torch

        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.symbolnet import SymbolNet

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        gir = GraphIR()
        input_op = AOP.Input(2)
        input_op.abs_tensor = AbsTensor([2, 4], DType.int8)
        source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()

        neg_op = AOP.Neg()
        neg_inputs = [gir.vars[source]]
        neg_outputs = neg_op.type_transfer(neg_inputs)
        neg_op.bind_input_like(neg_inputs)
        neg_op.bind_output_like(neg_outputs)
        neg_inst = gir.add_inst(InstExpr(neg_op, (source,)))

        abs_op = AOP.Abs()
        abs_inputs = [gir.vars[neg_inst.retval()]]
        abs_outputs = abs_op.type_transfer(abs_inputs)
        abs_op.bind_input_like(abs_inputs)
        abs_op.bind_output_like(abs_outputs)
        abs_inst = gir.add_inst(InstExpr(abs_op, (neg_inst.retval(),)))

        clip_op = AOP.Clip()
        clip_inputs = [gir.vars[abs_inst.retval()]]
        clip_outputs = clip_op.type_transfer(clip_inputs)
        clip_op.bind_input_like(clip_inputs)
        clip_op.bind_output_like(clip_outputs)
        gir.add_inst(InstExpr(clip_op, (abs_inst.retval(),)))
        gir.assert_wellform()

        rewritten = rewrite_gir(gir, matcher)
        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        rewritten.assert_wellform()
        self.assertEqual(
            [type(inst.iexpr.op).__name__ for inst in rewritten.insts],
            ['Input', 'CastI64', 'Neg', 'Cast', 'CastI64', 'Abs', 'Cast', 'CastI64', 'Clip', 'Cast'],
        )
        values = torch.tensor([-128, -127, -2, -1, 0, 1, 2, 127], dtype=torch.int8).reshape(2, 4)
        original = SymbolNet(gir)(values)[0]
        self.assertTrue(torch.equal(original, SymbolNet(rewritten)(values)[0]))
        with _patch_symbolnet_debug_helpers():
            compiled = torch.compile(SymbolNet(rewritten), backend='inductor', fullgraph=True)
            self.assertTrue(torch.equal(original, compiled(values)[0]))

    def test_int_squeeze_clip_without_promoting_ancestor_is_not_rewritten(self):
        import torch

        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.symbolnet import SymbolNet

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        gir = GraphIR()
        input_op = AOP.Input(3)
        input_op.abs_tensor = AbsTensor([4, 1, 7], DType.int16)
        source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
        squeeze_op = AOP.Squeeze()
        squeeze_op.extra_attrs['reduce_dim'] = 1
        squeeze_inputs = [gir.vars[source]]
        squeeze_outputs = squeeze_op.type_transfer(squeeze_inputs)
        squeeze_op.bind_input_like(squeeze_inputs)
        squeeze_op.bind_output_like(squeeze_outputs)
        squeeze_inst = gir.add_inst(InstExpr(squeeze_op, (source,)))
        clip_op = AOP.Clip()
        clip_inputs = [gir.vars[squeeze_inst.retval()]]
        clip_outputs = clip_op.type_transfer(clip_inputs)
        clip_op.bind_input_like(clip_inputs)
        clip_op.bind_output_like(clip_outputs)
        gir.add_inst(InstExpr(clip_op, (squeeze_inst.retval(),)))
        gir.assert_wellform()

        self.assertIsNone(rewrite_gir(gir, matcher))
        self.assertEqual([type(inst.iexpr.op).__name__ for inst in gir.insts], ['Input', 'Squeeze', 'Clip'])
        gir.assert_wellform()


    def test_int16_neg_clip_rewrite_widens_and_preserves_semantics(self):
        import torch

        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.symbolnet import SymbolNet

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        gir = GraphIR()
        input_op = AOP.Input(2)
        input_op.abs_tensor = AbsTensor([2, 4], DType.int16)
        source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
        neg_op = AOP.Neg()
        neg_inputs = [gir.vars[source]]
        neg_outputs = neg_op.type_transfer(neg_inputs)
        neg_op.bind_input_like(neg_inputs)
        neg_op.bind_output_like(neg_outputs)
        neg_inst = gir.add_inst(InstExpr(neg_op, (source,)))
        clip_op = AOP.Clip()
        clip_inputs = [gir.vars[neg_inst.retval()]]
        clip_outputs = clip_op.type_transfer(clip_inputs)
        clip_op.bind_input_like(clip_inputs)
        clip_op.bind_output_like(clip_outputs)
        clip_inst = gir.add_inst(InstExpr(clip_op, (neg_inst.retval(),)))
        gir.assert_wellform()

        rewritten = rewrite_gir(gir, matcher)
        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        rewritten.assert_wellform()
        self.assertEqual(
            [type(inst.iexpr.op).__name__ for inst in rewritten.insts],
            ['Input', 'CastI64', 'Neg', 'Cast', 'CastI64', 'Clip', 'Cast'],
        )
        rewritten_clip = rewritten.find_inst_by_id(clip_inst.identifier)
        assert rewritten_clip is not None
        self.assertEqual(rewritten.vars[rewritten_clip.iexpr.args[0]].dtype, DType.int64)
        self.assertEqual(rewritten.vars[rewritten_clip.retval()].dtype, DType.int64)
        x = torch.tensor(
            [-32768, -5, -1, 0, 1, 5, 32767, -32767], dtype=torch.int16
        ).reshape(2, 4)
        self.assertTrue(torch.equal(SymbolNet(gir)(x)[0], SymbolNet(rewritten)(x)[0]))
        with _patch_symbolnet_debug_helpers():
            compiled = torch.compile(SymbolNet(rewritten), backend='inductor', fullgraph=True)
            self.assertTrue(torch.equal(SymbolNet(gir)(x)[0], compiled(x)[0]))

    def test_int_clip_rewrite_rejects_unvalidated_producer(self):
        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.gir import GraphIR, InstExpr

        pattern_file = Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        matcher = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        gir = GraphIR()
        input_op = AOP.Input(2)
        input_op.abs_tensor = AbsTensor([4, 7], DType.int8)
        source = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
        neg_op = AOP.Neg()
        neg_inputs = [gir.vars[source]]
        neg_outputs = neg_op.type_transfer(neg_inputs)
        neg_op.bind_input_like(neg_inputs)
        neg_op.bind_output_like(neg_outputs)
        neg_inst = gir.add_inst(InstExpr(neg_op, (source,)))
        clip_op = AOP.Clip()
        clip_inputs = [gir.vars[neg_inst.retval()]]
        clip_outputs = clip_op.type_transfer(clip_inputs)
        clip_op.bind_input_like(clip_inputs)
        clip_op.bind_output_like(clip_outputs)
        gir.add_inst(InstExpr(clip_op, (neg_inst.retval(),)))
        gir.assert_wellform()

        self.assertIsNone(rewrite_gir(gir, matcher))
        self.assertEqual(
            [type(inst.iexpr.op).__name__ for inst in gir.insts], ['Input', 'Neg', 'Clip']
        )
        gir.assert_wellform()

    def test_int_minmax_excludes_identical_inputs(self):
        pattern = PatternDef(
            id='var-ne-capability',
            compiler='pytorch',
            target='cpu',
            description=None,
            severity=None,
            nodes=[PatternNodeDef(
                'n0',
                'MINIMUM',
                ['v_lhs', 'v_rhs'],
                ['v_out'],
                {},
                input_constraints=[{'lhs': 0, 'op': 'var_ne', 'rhs': 1}],
            )],
            values={},
        )
        matcher = PatternMatcher([pattern], compiler='pytorch', target='cpu')

        def ref(var_id):
            return {'var_id': var_id, 'shape': [2, 3], 'dtype': 'int16'}

        identical = {
            'op': 'MINIMUM',
            'attrs': {},
            'inputs': [ref('v0'), ref('v0')],
            'outputs': [ref('v1')],
            'raw_op': 'MINIMUM',
        }
        distinct = {
            **identical,
            'inputs': [ref('v0'), ref('v2')],
        }
        missing_identity = {
            **identical,
            'inputs': [
                {'shape': [2, 3], 'dtype': 'int16'},
                {'shape': [2, 3], 'dtype': 'int16'},
            ],
        }

        self.assertEqual(matcher.match([identical]), [])
        self.assertEqual(matcher.match([missing_identity]), [])
        self.assertEqual(matcher.match([distinct]), [pattern])

    def test_validated_fp16_interp_catalog(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file),
            compiler='pytorch',
            target='cpu',
            backend='torch',
        )

        def interp_node(op, shape):
            return {
                'op': op,
                'attrs': {},
                'inputs': [{'var_id': 'v0', 'shape': list(shape), 'dtype': 'float16'}],
                'outputs': [{'var_id': 'v1', 'shape': list(shape), 'dtype': 'float16'}],
                'raw_op': op,
            }

        target = 'pt-inductor-fp16-interp'
        rank5_trilinear = interp_node('TRILINEAR_INTERP', (1, 3, 4, 5, 6))
        rank4_trilinear = interp_node('TRILINEAR_INTERP', (1, 3, 5, 6))
        rank4_bilinear = interp_node('BILINEAR_INTERP', (1, 3, 5, 6))
        rank4_bicubic = interp_node('BICUBIC_INTERP', (1, 3, 5, 6))
        rank4_nearest = interp_node('NEAREST_INTERP', (1, 3, 5, 6))
        rank2_bilinear = interp_node('BILINEAR_INTERP', (5, 6))

        self.assertIn(target, {p.id for p in dedup.matcher.match([rank5_trilinear])})
        self.assertIn(target, {p.id for p in dedup.matcher.match([rank4_trilinear])})
        self.assertIn(target, {p.id for p in dedup.matcher.match([rank4_bilinear])})
        self.assertIn(target, {p.id for p in dedup.matcher.match([rank4_bicubic])})
        self.assertIn(target, {p.id for p in dedup.matcher.match([rank4_nearest])})
        self.assertEqual(_candidates_for('BicubicInterp'), [])
        self.assertEqual(_candidates_for('TrilinearInterp'), [])
        self.assertNotIn(target, {p.id for p in dedup.matcher.match([rank2_bilinear])})

    def test_torch_reduction_rewrite_candidates_are_materializable(self):
        supported = {'ReduceMean', 'ReduceMax', 'ReduceMin', 'Squeeze', 'ArgMax', 'ArgMin'}
        for source in ('ReduceSum', 'ReduceMean', 'ReduceMax', 'ReduceMin'):
            candidates = _candidates_for(source)
            self.assertTrue(set(candidates).issubset(supported), (source, candidates))
        self.assertNotIn('ReduceSum', _candidates_for('ReduceMin'))
        self.assertNotIn('ReduceProd', _candidates_for('ReduceMax'))

    def test_rewrite_binds_input_metadata_for_torch_dispatch(self):
        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.dedup.rewrite import _rewrite_inst
        from nnsmith.gir import GraphIR, InstExpr
        from nnsmith.materialize.torch.forward import forward_fn

        from typing import cast

        gir = GraphIR()
        empty_args = cast(tuple[str], ())
        inputs = []
        for _ in range(2):
            op = AOP.Input(1)
            op.abs_tensor = AbsTensor([2], DType.float32)
            inputs.append(gir.add_inst(InstExpr(op, empty_args)).retval())
        inst = gir.add_inst(InstExpr(AOP.Mul(), tuple(inputs)))

        rewritten = _rewrite_inst(gir, gir.insts.index(inst), 'Div')
        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        op = rewritten.find_inst_by_id(inst.identifier).iexpr.op
        self.assertEqual(op.input_like[0].dtype, DType.float32)
        self.assertIsNotNone(forward_fn(op))

    def test_squeeze_rewrite_preserves_explicit_reduce_dim(self):
        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.dedup.rewrite import _rewrite_inst
        from nnsmith.gir import GraphIR, InstExpr
        from typing import cast

        gir = GraphIR()
        empty_args = cast(tuple[str], ())
        input_op = AOP.Input(2)
        input_op.abs_tensor = AbsTensor([1, 3], DType.int8)
        input_var = gir.add_inst(InstExpr(input_op, empty_args)).retval()
        reduce = AOP.ReduceMin()
        reduce.extra_attrs['reduce_dim'] = 0
        inst = gir.add_inst(InstExpr(reduce, (input_var,)))

        rewritten = _rewrite_inst(gir, gir.insts.index(inst), 'Squeeze')
        self.assertIsNotNone(rewritten)
        assert rewritten is not None
        op = rewritten.find_inst_by_id(inst.identifier).iexpr.op
        self.assertIn('reduce_dim', op.extra_attrs)
        self.assertEqual(op.extra_attrs['reduce_dim'], 0)

    def test_rewrite_rejects_candidate_with_unsatisfied_operator_requirements(self):
        from typing import cast

        from nnsmith.abstract import op as AOP
        from nnsmith.abstract.dtype import DType
        from nnsmith.abstract.tensor import AbsTensor
        from nnsmith.dedup.rewrite import _rewrite_inst
        from nnsmith.gir import GraphIR, InstExpr

        gir = GraphIR()
        input_op = AOP.Input(2)
        input_op.abs_tensor = AbsTensor([2, 3], DType.int8)
        input_var = gir.add_inst(InstExpr(input_op, cast(tuple[str], ()))).retval()
        reduce = AOP.ReduceMin()
        reduce.extra_attrs['reduce_dim'] = 0
        inst = gir.add_inst(InstExpr(reduce, (input_var,)))

        self.assertIsNone(_rewrite_inst(gir, gir.insts.index(inst), 'Squeeze'))

    def test_fp16_multiply_to_int32_cast_pattern_matches_reduced_shape(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file), compiler='pytorch', target='cpu', backend='torch'
        )
        mul = {
            'op': 'MULTIPLY',
            'attrs': {},
            'inputs': [
                {'var_id': 'lhs', 'shape': [8, 5, 5], 'dtype': 'float16'},
                {'var_id': 'rhs', 'shape': [1, 5, 1], 'dtype': 'float16'},
            ],
            'outputs': [{'var_id': 'mid', 'shape': [8, 5, 5], 'dtype': 'float16'}],
            'raw_op': 'Mul',
        }
        cast = {
            'op': 'CAST',
            'attrs': {},
            'inputs': [{'var_id': 'mid', 'shape': [8, 5, 5], 'dtype': 'float16'}],
            'outputs': [{'var_id': 'out', 'shape': [8, 5, 5], 'dtype': 'int32'}],
            'raw_op': 'CastI32',
        }
        matched = {p.id for p in dedup.matcher.match([mul, cast])}
        self.assertIn('pt-inductor-fp16-multiply-to-cast-int32', matched)

    def test_validated_fp16_trans_to_trans_catalog(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file),
            compiler='pytorch',
            target='cpu',
            backend='torch',
        )

        def node(op, inputs, outputs):
            def ref(var_id):
                return {'var_id': var_id, 'shape': [2, 3], 'dtype': 'float16'}
            return {
                'op': op,
                'attrs': {},
                'inputs': [ref(v) for v in inputs],
                'outputs': [ref(v) for v in outputs],
                'raw_op': op,
            }

        all_ops = ('TAN', 'SIN', 'COS', 'ATAN')
        for producer_op in all_ops:
            for consumer_op in all_ops:
                producer = node(producer_op, ['v_in'], ['v_mid'])
                consumer = node(consumer_op, ['v_mid'], ['v_out'])
                matched = {p.id for p in dedup.matcher.match([producer, consumer])}
                matched_target = 'pt-inductor-fp16-trans-to-trans' in matched
                self.assertEqual(
                    matched_target,
                    True,
                    (producer_op, consumer_op, matched),
                )

    def test_fp16_matmul_and_arithmetic_to_trig_patterns(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file),
            compiler='pytorch',
            target='cpu',
            backend='torch',
        )

        def node(op, inputs, outputs, dtype='float16'):
            def ref(var_id):
                return {'var_id': var_id, 'shape': [2, 3], 'dtype': dtype}
            return {
                'op': op,
                'attrs': {},
                'inputs': [ref(v) for v in inputs],
                'outputs': [ref(v) for v in outputs],
                'raw_op': op,
            }

        matmul = node('MATMUL', ['lhs', 'rhs'], ['mid'])
        add = node('ADD', ['lhs', 'rhs'], ['mid'])
        sub = node('SUBTRACT', ['lhs', 'rhs'], ['mid'])
        mul = node('MULTIPLY', ['lhs', 'rhs'], ['mid'])
        for trig in ('TAN', 'SIN', 'COS'):
            matched = {
                p.id for p in dedup.matcher.match(
                    [matmul, node(trig, ['mid'], ['out'])]
                )
            }
            self.assertIn('pt-inductor-fp16-matmul-to-trig', matched)

        matched = {
            p.id for p in dedup.matcher.match([add, node('TAN', ['mid'], ['out'])])
        }
        self.assertIn('pt-inductor-fp16-add-to-tan', matched)

        matched = {
            p.id for p in dedup.matcher.match([add, node('SIN', ['mid'], ['out'])])
        }
        self.assertIn('pt-inductor-fp16-add-to-sin', matched)

        matched = {
            p.id for p in dedup.matcher.match([add, node('COS', ['mid'], ['out'])])
        }
        self.assertIn('pt-inductor-fp16-add-to-cos', matched)

        matched = {
            p.id for p in dedup.matcher.match([sub, node('TAN', ['mid'], ['out'])])
        }
        self.assertIn('pt-inductor-fp16-sub-to-tan', matched)

        matched = {
            p.id for p in dedup.matcher.match([mul, node('TAN', ['mid'], ['out'])])
        }
        self.assertIn('pt-inductor-fp16-multiply-to-tan', matched)

        gelu = node('GELU', ['in'], ['mid'])
        matched = {
            p.id for p in dedup.matcher.match([gelu, node('FLOOR', ['mid'], ['out'])])
        }
        self.assertIn('pt-inductor-fp16-gelu-to-floor', matched)

        matched = {
            p.id for p in dedup.matcher.match([gelu, node('ROUND', ['mid'], ['out'])])
        }
        self.assertIn('pt-inductor-fp16-gelu-to-round', matched)

        for producer, consumer in (
            (node('GELU', ['in'], ['mid'], dtype='float32'), node('FLOOR', ['mid'], ['out'], dtype='float32')),
            (gelu, node('CEIL', ['mid'], ['out'])),
            (add, node('FLOOR', ['mid'], ['out'])),
            (gelu, node('FLOOR', ['other'], ['out'])),
        ):
            matched = {p.id for p in dedup.matcher.match([producer, consumer])}
            self.assertNotIn('pt-inductor-fp16-gelu-to-floor', matched)

        for producer, consumer in (
            (node('GELU', ['in'], ['mid'], dtype='float32'), node('ROUND', ['mid'], ['out'], dtype='float32')),
            (gelu, node('CEIL', ['mid'], ['out'])),
            (add, node('ROUND', ['mid'], ['out'])),
            (gelu, node('ROUND', ['other'], ['out'])),
        ):
            matched = {p.id for p in dedup.matcher.match([producer, consumer])}
            self.assertNotIn('pt-inductor-fp16-gelu-to-round', matched)

        for producer, consumer in (
            (add, node('SIN', ['mid'], ['out'])),
            (node('MULTIPLY', ['lhs', 'rhs'], ['mid']), node('TAN', ['mid'], ['out'])),
            (add, node('TAN', ['other'], ['out'])),
        ):
            matched = {p.id for p in dedup.matcher.match([producer, consumer])}
            self.assertNotIn('pt-inductor-fp16-add-to-tan', matched)

        for producer, consumer in (
            (sub, node('SIN', ['mid'], ['out'])),
            (add, node('TAN', ['mid'], ['out'])),
            (sub, node('TAN', ['other'], ['out'])),
        ):
            matched = {p.id for p in dedup.matcher.match([producer, consumer])}
            self.assertNotIn('pt-inductor-fp16-sub-to-tan', matched)

        for producer, consumer in (
            (add, node('COS', ['mid'], ['out'])),
            (sub, node('SIN', ['mid'], ['out'])),
            (add, node('SIN', ['other'], ['out'])),
        ):
            matched = {p.id for p in dedup.matcher.match([producer, consumer])}
            self.assertNotIn('pt-inductor-fp16-add-to-sin', matched)

        for producer, consumer in (
            (add, node('TAN', ['mid'], ['out'])),
            (sub, node('COS', ['mid'], ['out'])),
            (add, node('COS', ['other'], ['out'])),
        ):
            matched = {p.id for p in dedup.matcher.match([producer, consumer])}
            self.assertNotIn('pt-inductor-fp16-add-to-cos', matched)

        for producer, consumer in (
            (mul, node('SIN', ['mid'], ['out'])),
            (mul, node('COS', ['mid'], ['out'])),
            (add, node('TAN', ['mid'], ['out'])),
            (mul, node('TAN', ['other'], ['out'])),
        ):
            matched = {p.id for p in dedup.matcher.match([producer, consumer])}
            self.assertNotIn('pt-inductor-fp16-multiply-to-tan', matched)

    def test_byte_matmul_requires_rhs_last_dimension_one(self):
        pattern_file = (
            Path(__file__).parents[1] / 'nnsmith' / 'dedup' / 'patterns-pytorch.json'
        )
        dedup = DedupMatcher(
            pattern_dir=str(pattern_file),
            compiler='pytorch',
            target='cpu',
            backend='torch',
        )

        def ref(var_id, shape):
            return {'var_id': var_id, 'shape': list(shape), 'dtype': 'int8'}

        def matmul(rhs_shape):
            return {
                'op': 'MATMUL',
                'attrs': {},
                'inputs': [ref('lhs', (2, 4, 5)), ref('rhs', rhs_shape)],
                'outputs': [ref('out', (2, 4, rhs_shape[-1]))],
                'raw_op': 'MATMUL',
            }

        target = 'pt-inductor-cpu-byte-matmul-broadcast-upcast'
        self.assertIn(target, {p.id for p in dedup.matcher.match([matmul((2, 5, 1))])})
        self.assertNotIn(target, {p.id for p in dedup.matcher.match([matmul((2, 5, 6))])})

if __name__ == '__main__':
    unittest.main()
