import inspect
import json
import pickle
from pathlib import Path
import unittest

from nnsmith.cli.fuzz import FuzzingLoop
from nnsmith.dedup import DedupMatcher
from nnsmith.dedup.constraints import DtypeExact, ValueRangeOutside
from nnsmith.dedup.matcher import PatternMatcher
from nnsmith.dedup.parser import (
    FlowConstraint,
    PatternDef,
    PatternNodeDef,
    PatternValueDef,
)
from nnsmith.dedup.rewrite import rewrite_gir
from nnsmith.graph_gen import SymbolicGen

DATA_DIR = Path('/home/xyzboom/Data/maybeBug/2026-09-24/neuri_tvm_cuda_10k')
PATTERNS_FILE = Path('/home/xyzboom/Data/remote-2026-10-01/patterns-neu-compile/neu-compile.json')
SEEDS_FILE = Path('/home/xyzboom/Code/aiCompiler/nnsmith-worktrees/neuri-tvm-prededup/nnsmith/dedup/known_bug_seeds_tvm_cuda.json')

class TestNeuriTvmCudaPrededup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = DedupMatcher(
            pattern_dir=str(PATTERNS_FILE),
            compiler='tvm',
            target='cuda',
            backend='torch',
        )

    def test_prededup_online_interception_without_seed_retry(self):
        """Verify online generation-time pre-dedup replicates aiFuzzer: no seed retry."""
        source = inspect.getsource(FuzzingLoop.make_testcase)
        self.assertNotIn('retrying generation', source)
        self.assertNotIn('_dedup_retry_seed', source)
        self.assertNotIn('generation_seed', source)
        self.assertNotIn('rewrite_gir', source)

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

    def test_dedup_matcher_on_node_generated_api(self):
        self.assertTrue(callable(getattr(self.matcher, 'on_node_generated', None)))
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

    def test_fuzz_log_records_pattern_ids_without_seed_retry(self):
        source = inspect.getsource(FuzzingLoop.make_testcase)
        self.assertIn('pattern_hits=', source)
        self.assertIn('prevented_by_pattern', source)
        self.assertIn('resampled without seed change', source)

    def test_backward_hook_checks_before_graph_mutation(self):
        source = inspect.getsource(SymbolicGen.try_occupy_placeholder)
        check_at = source.index('on_node_generated')
        placeholder_commit_at = source.index('self.forward_insert_node(ph, [])')
        candidate_commit_at = source.index('self.backward_insert_node')
        self.assertLess(check_at, placeholder_commit_at)
        self.assertLess(check_at, candidate_commit_at)
        self.assertIn('output_vars=phvars', source)

if __name__ == '__main__':
    unittest.main()
