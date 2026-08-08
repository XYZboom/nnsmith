"""
Pattern definitions and parser — loads AIFuzzer-compatible JSON pattern files.
"""

import json
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

from nnsmith.dedup.constraints import (
    DimMatcher, parse_dim_matcher,
    DtypeMatcher, parse_dtype_matcher,
    AttrMatcher, parse_attr_matcher,
    ValueRangeMatcher, ValueRangeAny, parse_value_range_matcher,
    ExpressionConstraint, parse_expression_constraints,
)


# =============================================================================
# Data classes
# =============================================================================

class PatternNodeDef:
    def __init__(self, id: str, op: str, inputs: List[str], outputs: List[str],
                 attrs: Dict[str, AttrMatcher]):
        self.id = id
        self.op = op
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs


class PatternValueDef:
    def __init__(self, id: str, ndim: Optional[DimMatcher], shape: List[DimMatcher],
                 dtype: DtypeMatcher,
                 expression_constraints: Optional[List[ExpressionConstraint]] = None,
                 range: Optional[ValueRangeMatcher] = None):
        self.id = id
        self.ndim = ndim
        self.shape = shape
        self.dtype = dtype
        self.expression_constraints = expression_constraints or []
        self.range = range or ValueRangeAny()


class FlowConstraint:
    def __init__(self, from_node: str, from_output: int, to_node: str, to_input: int):
        self.from_node = from_node
        self.from_output = from_output
        self.to_node = to_node
        self.to_input = to_input


class GraphConstraints:
    def __init__(self, min_nodes: Optional[int], max_nodes: Optional[int],
                 required_ops: Optional[List[str]]):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.required_ops = required_ops


class PatternDef:
    def __init__(self, id: str, compiler: str, target: Optional[str],
                 description: Optional[str], severity: Optional[str],
                 nodes: List[PatternNodeDef],
                 values: Dict[str, PatternValueDef],
                 graph_constraints: Optional[GraphConstraints] = None,
                 flow_constraints: Optional[List[FlowConstraint]] = None):
        self.id = id
        self.compiler = compiler
        self.target = target
        self.description = description
        self.severity = severity
        self.nodes = nodes
        self.values = values
        self.graph_constraints = graph_constraints
        self.flow_constraints = flow_constraints

    def __repr__(self):
        return f"Pattern({self.id}, {self.compiler}/{self.target}, {len(self.nodes)} nodes)"


# =============================================================================
# Parser
# =============================================================================

def parse_pattern_file(filepath: str) -> List[PatternDef]:
    """Parse a pattern JSON file. Returns list of PatternDef."""
    with open(filepath) as f:
        data = json.load(f)
    patterns = []
    for p_json in data.get('patterns', []):
        p = p_json
        nodes = []
        for n_json in p.get('nodes', []):
            attrs = {}
            for key, val in n_json.get('attrs', {}).items():
                attrs[key] = parse_attr_matcher(val)
            nodes.append(PatternNodeDef(
                id=n_json['id'],
                op=n_json['op'],
                inputs=n_json.get('inputs', []),
                outputs=n_json.get('outputs', []),
                attrs=attrs,
            ))
        values = {}
        for v_json in p.get('values', []):
            vid = v_json['id']
            ndim = parse_dim_matcher(v_json['ndim']) if 'ndim' in v_json else None
            shape = [parse_dim_matcher(d) for d in v_json.get('shape', [])]
            dtype = parse_dtype_matcher(v_json.get('dtype'))
            ecs = parse_expression_constraints(v_json.get('expressionConstraints'))
            values[vid] = PatternValueDef(
                id=vid, ndim=ndim, shape=shape, dtype=dtype,
                expression_constraints=ecs,
                range=parse_value_range_matcher(v_json.get('range')),
            )
        gc = None
        if 'graphConstraints' in p:
            gc_obj = p['graphConstraints']
            gc = GraphConstraints(
                min_nodes=gc_obj.get('minNodes'),
                max_nodes=gc_obj.get('maxNodes'),
                required_ops=gc_obj.get('requiredOps'),
            )
        fc = None
        if 'flowConstraints' in p:
            fc = [FlowConstraint(
                from_node=f['fromNode'],
                from_output=f.get('fromOutput', 0),
                to_node=f['toNode'],
                to_input=f.get('toInput', 0),
            ) for f in p['flowConstraints']]
        patterns.append(PatternDef(
            id=p['id'], compiler=p.get('compiler', ''),
            target=p.get('target'), description=p.get('description'),
            severity=p.get('severity'),
            nodes=nodes, values=values,
            graph_constraints=gc, flow_constraints=fc,
        ))
    return patterns


def load_patterns(pattern_dir: str) -> List[PatternDef]:
    """Load all pattern JSON files from a directory."""
    all_patterns = []
    for filepath in sorted(glob.glob(os.path.join(pattern_dir, '*.json'))):
        try:
            patterns = parse_pattern_file(filepath)
            all_patterns.extend(patterns)
        except Exception as e:
            print(f"Warning: failed to parse {filepath}: {e}")
    return all_patterns