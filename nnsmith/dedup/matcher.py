"""
PatternMatcher — matches normalized nnsmith GraphIR nodes against PatternDefs.
"""

from typing import Any, Dict, List, Optional, Set

from nnsmith.dedup.constraints import (
    DimAny, DimMatcher,
    DtypeAny, DtypeMatcher,
    AttrAny, AttrMatcher,
    ValueRangeAny, ValueRangeMatcher,
)
from nnsmith.dedup.parser import (
    PatternDef, PatternNodeDef, PatternValueDef,
    FlowConstraint, GraphConstraints,
)


class PatternMatcher:
    """
    Python pattern matcher for nnsmith-generated programs.
    Matches a list of normalized nodes against a list of PatternDefs.
    """

    def __init__(self, patterns: List[PatternDef], compiler: Optional[str] = None,
                 target: Optional[str] = None):
        # Filter by compiler and target
        self.patterns = []
        for p in patterns:
            if compiler is not None and p.compiler != compiler:
                continue
            if target is not None and p.target is not None and p.target != target:
                continue
            self.patterns.append(p)

        # Index patterns by type
        self.single_op_patterns: Dict[str, List[PatternDef]] = {}
        self.multi_op_patterns: List[PatternDef] = []
        for p in self.patterns:
            if len(p.nodes) == 1:
                op = p.nodes[0].op
                if op not in self.single_op_patterns:
                    self.single_op_patterns[op] = []
                self.single_op_patterns[op].append(p)
            else:
                self.multi_op_patterns.append(p)

    def match(self, nodes: List[dict]) -> List[PatternDef]:
        """Match a list of normalized nodes against all patterns. Returns matched patterns."""
        matched = []

        # Single-op patterns: straightforward
        for node in nodes:
            op = node['op']
            if op in self.single_op_patterns:
                for pattern in self.single_op_patterns[op]:
                    if pattern not in matched and self._check_single_op(pattern, node, nodes):
                        matched.append(pattern)

        # Multi-op patterns: sliding window
        for pattern in self.multi_op_patterns:
            if pattern in matched:
                continue
            if self._check_multi_op(pattern, nodes):
                matched.append(pattern)

        return matched

    def _check_single_op(self, pattern: PatternDef, node: dict,
                         all_nodes: List[dict]) -> bool:
        pnode = pattern.nodes[0]

        # 1. Check attrs
        if not self._check_attrs(node, pnode):
            return False

        # 2. Check value constraints (by position)
        for vid, pval in pattern.values.items():
            actual_ref = self._find_actual_ref(vid, pattern, [node], all_nodes)
            if actual_ref is None:
                if (all(isinstance(d, DimAny) for d in pval.shape)
                    and isinstance(pval.dtype, DtypeAny)
                    and (pval.ndim is None or isinstance(pval.ndim, DimAny))
                    and isinstance(pval.range, ValueRangeAny)
                    and not pval.expression_constraints):
                    continue
                return False

            if not self._check_value(pval, actual_ref):
                return False

        # 3. Check graph constraints
        if not self._check_graph_constraints(pattern, len(all_nodes), all_nodes):
            return False

        # 4. Check flow constraints
        if not self._check_flow_constraints(pattern, [node]):
            return False

        return True

    def _check_multi_op(self, pattern: PatternDef, all_nodes: List[dict]) -> bool:
        """Check multi-op pattern by matching pattern nodes as a SUBSEQUENCE of
        all_nodes (skipping unrelated nodes in between, e.g. dead code).

        For each starting position, greedily advance through all_nodes matching
        each pattern node in order. Flow constraints then verify dataflow.
        """
        n = len(pattern.nodes)
        if n > len(all_nodes):
            return False

        for start in range(len(all_nodes)):
            # Greedy subsequence match starting at `start`
            window = []
            ni = 0  # index into pattern.nodes
            for i in range(start, len(all_nodes)):
                if ni >= n:
                    break
                actual = all_nodes[i]
                pnode = pattern.nodes[ni]
                if actual['op'] == pnode.op and self._check_attrs(actual, pnode):
                    window.append(actual)
                    ni += 1
            if ni < n:
                continue  # not all pattern nodes matched as subsequence

            # Pattern matched structurally. Check value constraints.
            match = True
            for vid, pval in pattern.values.items():
                actual_ref = self._find_actual_ref(vid, pattern, window, all_nodes)
                if actual_ref is None:
                    if (all(isinstance(d, DimAny) for d in pval.shape)
                        and isinstance(pval.dtype, DtypeAny)
                        and (pval.ndim is None or isinstance(pval.ndim, DimAny))
                        and isinstance(pval.range, ValueRangeAny)
                        and not pval.expression_constraints):
                        continue
                    match = False
                    break
                if not self._check_value(pval, actual_ref):
                    match = False
                    break

            if not match:
                continue

            if not self._check_graph_constraints(pattern, len(all_nodes), all_nodes):
                continue
            if not self._check_flow_constraints(pattern, window):
                continue

            return True

        return False

    def _check_value(self, pval: PatternValueDef, actual_ref: dict) -> bool:
        """Check shape, ndim, dtype, expression constraints on a value reference."""
        actual_dims = actual_ref['shape']

        # Check ndim
        if pval.ndim is not None and not isinstance(pval.ndim, DimAny):
            if not pval.ndim.matches(len(actual_dims)):
                return False

        # Check shape
        if len(pval.shape) > len(actual_dims):
            return False
        for i in range(len(pval.shape)):
            if i >= len(actual_dims):
                if not isinstance(pval.shape[i], DimAny):
                    return False
                continue
            if not pval.shape[i].matches(actual_dims[i]):
                return False

        # Check dtype
        if not isinstance(pval.dtype, DtypeAny):
            if not pval.dtype.matches(actual_ref['dtype'], 0):
                return False

        # Check expression constraints
        if pval.expression_constraints:
            for ec in pval.expression_constraints:
                if not ec.matches(actual_dims):
                    return False

        return True

    def _check_attrs(self, node: dict, pnode: PatternNodeDef) -> bool:
        for key, matcher in pnode.attrs.items():
            actual = node['attrs'].get(key)
            if actual is None:
                if isinstance(matcher, AttrAny):
                    continue
                return False
            if not matcher.matches(actual):
                return False
        return True

    def _find_actual_ref(self, vid: str, pattern: PatternDef,
                         nodes_window: List[dict],
                         all_nodes: List[dict]) -> Optional[dict]:
        """Find the actual value reference by position."""
        for p_idx, pnode in enumerate(pattern.nodes):
            if p_idx >= len(nodes_window):
                continue
            actual_node = nodes_window[p_idx]

            input_pos = pnode.inputs.index(vid) if vid in pnode.inputs else -1
            if input_pos >= 0 and input_pos < len(actual_node['inputs']):
                return actual_node['inputs'][input_pos]

            output_pos = pnode.outputs.index(vid) if vid in pnode.outputs else -1
            if output_pos >= 0 and output_pos < len(actual_node['outputs']):
                return actual_node['outputs'][output_pos]

        return None

    def _check_graph_constraints(self, pattern: PatternDef, n_nodes: int,
                                  all_nodes: List[dict]) -> bool:
        gc = pattern.graph_constraints
        if gc is None:
            return True
        if gc.min_nodes is not None and n_nodes < gc.min_nodes:
            return False
        if gc.max_nodes is not None and n_nodes > gc.max_nodes:
            return False
        if gc.required_ops is not None:
            seen_ops = {n['op'] for n in all_nodes}
            if not any(op in seen_ops for op in gc.required_ops):
                return False
        return True

    def _check_flow_constraints(self, pattern: PatternDef,
                                 nodes_window: List[dict]) -> bool:
        fc = pattern.flow_constraints
        if not fc:
            return True
        node_map = {}
        for i, pnode in enumerate(pattern.nodes):
            if i < len(nodes_window):
                node_map[pnode.id] = nodes_window[i]
        for constraint in fc:
            from_node = node_map.get(constraint.from_node)
            to_node = node_map.get(constraint.to_node)
            if from_node is None or to_node is None:
                return False
            if constraint.from_output >= len(from_node['outputs']):
                return False
            if constraint.to_input >= len(to_node['inputs']):
                return False
            # Flow constraint: check shape compatibility
            from_shape = from_node['outputs'][constraint.from_output]['shape']
            to_shape = to_node['inputs'][constraint.to_input]['shape']
            if from_shape != to_shape:
                return False
        return True


class DedupMatcher:
    """
    High-level dedup matcher: loads patterns, normalizes GraphIR, checks matches.

    Usage:
        matcher = DedupMatcher(pattern_dir="/path/to/patterns/")
        if matcher.matches(gir):
            print("Matched, skip this program")
    """

    def __init__(self, pattern_dir: str = None,
                 compiler: Optional[str] = None,
                 target: Optional[str] = None,
                 backend: str = 'torch'):
        from nnsmith.dedup.parser import load_patterns

        self.compiler = compiler
        self.target = target
        self.backend = backend
        self.patterns = []

        if pattern_dir:
            import os
            if os.path.isdir(pattern_dir):
                all_patterns = load_patterns(pattern_dir)
                # Filter
                for p in all_patterns:
                    if compiler is not None and p.compiler != compiler:
                        continue
                    if target is not None and p.target is not None and p.target != target:
                        continue
                    self.patterns.append(p)
                self.matcher = PatternMatcher(self.patterns,
                                              compiler=compiler, target=target)

    def matches(self, ir) -> List[PatternDef]:
        """Check if a GraphIR program matches any pattern. Returns matched patterns."""
        if not hasattr(self, 'matcher') or not self.patterns:
            return []
        from nnsmith.dedup.normalizer import normalize_graph
        nodes = normalize_graph(ir, backend=self.backend)
        return self.matcher.match(nodes)

    def match_count(self, ir) -> int:
        """Return the number of matched patterns."""
        return len(self.matches(ir))