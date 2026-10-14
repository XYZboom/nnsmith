"""
PatternMatcher — matches normalized nnsmith GraphIR nodes against PatternDefs.
"""

from collections import Counter
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
                if isinstance(op, str):
                    if op not in self.single_op_patterns:
                        self.single_op_patterns[op] = []
                    self.single_op_patterns[op].append(p)
                elif isinstance(op, dict) and "$in" in op:
                    for o in op["$in"]:
                        if o not in self.single_op_patterns:
                            self.single_op_patterns[o] = []
                        self.single_op_patterns[o].append(p)
                else:
                    self.multi_op_patterns.append(p)
            else:
                self.multi_op_patterns.append(p)

    def match(
        self, nodes: List[dict], required_node: Optional[dict] = None
    ) -> List[PatternDef]:
        """Match patterns, optionally requiring an occurrence to contain one node."""
        matched = []

        # Single-op patterns: straightforward
        candidate_nodes = [required_node] if required_node is not None else nodes
        for node in candidate_nodes:
            op = node['op']
            if op in self.single_op_patterns:
                for pattern in self.single_op_patterns[op]:
                    if pattern not in matched and self._check_single_op(pattern, node, nodes):
                        matched.append(pattern)

        # Multi-op patterns: sliding window
        for pattern in self.multi_op_patterns:
            if pattern in matched:
                continue
            if self._check_multi_op(pattern, nodes, required_node=required_node):
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

    def _check_multi_op(
        self,
        pattern: PatternDef,
        all_nodes: List[dict],
        required_node: Optional[dict] = None,
    ) -> bool:
        """Check multi-op pattern by matching pattern nodes as a SUBSEQUENCE of
        all_nodes (skipping unrelated nodes in between, e.g. dead code).

        Enumerate structural subsequences so a required candidate can occupy any
        pattern position. Flow constraints then verify exact dataflow.
        """
        n = len(pattern.nodes)
        if n > len(all_nodes):
            return False

        def occurrence_matches(window: List[dict]) -> bool:
            if required_node is not None and all(
                node is not required_node for node in window
            ):
                return False
            # Pattern matched structurally. Check value constraints.
            for vid, pval in pattern.values.items():
                actual_ref = self._find_actual_ref(vid, pattern, window, all_nodes)
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

            if not self._check_graph_constraints(pattern, len(all_nodes), all_nodes):
                return False
            if not self._check_flow_constraints(pattern, window):
                return False

            return True

        def search(node_idx: int, pattern_idx: int, window: List[dict]) -> bool:
            if pattern_idx == n:
                return occurrence_matches(window)
            remaining = n - pattern_idx
            for idx in range(node_idx, len(all_nodes) - remaining + 1):
                actual = all_nodes[idx]
                pnode = pattern.nodes[pattern_idx]
                if not self._op_matches(actual['op'], pnode.op):
                    continue
                if not self._check_attrs(actual, pnode):
                    continue
                if search(idx + 1, pattern_idx + 1, window + [actual]):
                    return True
            return False

        return search(0, 0, [])

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

        # Check conservative value interval. Unknown intervals must not satisfy
        # a constrained range matcher.
        if not isinstance(pval.range, ValueRangeAny):
            if not pval.range.matches(actual_ref.get('range')):
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

    def _op_matches(self, actual_op: str, pattern_op: Any) -> bool:
        """Check if an actual op name matches a pattern op (string or $in list)."""
        if isinstance(pattern_op, str):
            return actual_op == pattern_op
        if isinstance(pattern_op, dict):
            if '$in' in pattern_op:
                return actual_op in pattern_op['$in']
        return actual_op == pattern_op

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
            from_ref = from_node['outputs'][constraint.from_output]
            to_ref = to_node['inputs'][constraint.to_input]
            from_var = from_ref.get('var_id')
            to_var = to_ref.get('var_id')
            # Missing identity cannot prove dataflow; fail closed.
            if from_var is None or to_var is None or from_var != to_var:
                return False
        return True


class DedupMatcher:
    """
    High-level dedup matcher: loads patterns, normalizes GraphIR, checks matches.

    Usage:
        from nnsmith.dedup.rewrite import rewrite_gir
        matcher = DedupMatcher(pattern_dir="/path/to/patterns/")
        matched = matcher.matches(gir)
        if matched:
            # Locally rewrite this same GIR; never skip it or regenerate by seed.
            rewritten = rewrite_gir(gir, matcher)
            gir = rewritten or gir
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
        self.prevented_count = 0
        self.prevented_by_pattern = Counter()

        if pattern_dir:
            import os
            if os.path.isdir(pattern_dir):
                all_patterns = load_patterns(pattern_dir)
            elif os.path.isfile(pattern_dir):
                from nnsmith.dedup.parser import parse_pattern_file
                all_patterns = parse_pattern_file(pattern_dir)
            else:
                all_patterns = []
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

    def match_first(self, ir) -> Optional[PatternDef]:
        matched = self.matches(ir)
        return matched[0] if matched else None

    def on_node_generated(
        self,
        gir: Any,
        candidate_op: Any,
        input_vars: List[str],
        otensors: List[Any],
        output_vars: Optional[List[str]] = None,
        input_tensors: Optional[List[Any]] = None,
    ) -> Optional[PatternDef]:
        """aiFuzzer-style online check: evaluate candidate node before committing to graph."""
        if not self.patterns:
            return None
        from nnsmith.dedup.normalizer import normalize_graph, normalize_node_dict
        nodes = normalize_graph(gir, backend=self.backend)
        cand_node = normalize_node_dict(
            candidate_op,
            input_vars,
            otensors,
            backend=self.backend,
            output_vars=output_vars,
        )
        vars_map = dict(gir.vars)
        for idx, var_id in enumerate(input_vars):
            tensor = vars_map.get(var_id)
            if tensor is None and input_tensors is not None and idx < len(input_tensors):
                tensor = input_tensors[idx]
            if tensor is not None:
                from nnsmith.dedup.normalizer import _dtype_to_str, _shape_to_list
                cand_node['inputs'][idx]['shape'] = _shape_to_list(tensor.shape)
                cand_node['inputs'][idx]['dtype'] = _dtype_to_str(tensor.dtype)

        if output_vars:
            output_set = set(output_vars)
            insert_at = next(
                (
                    idx
                    for idx, node in enumerate(nodes)
                    if any(ref.get('var_id') in output_set for ref in node['inputs'])
                ),
                len(nodes),
            )
            nodes.insert(insert_at, cand_node)
        else:
            nodes.append(cand_node)

        cand_matches = self.matcher.match(nodes, required_node=cand_node)
        if cand_matches:
            matched = cand_matches[0]
            self.prevented_count += 1
            self.prevented_by_pattern[matched.id] += 1
            return matched
        return None