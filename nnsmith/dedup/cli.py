"""
nnsmith Pattern Dedup CLI — offline analysis tool.

Usage:
    python3 -m nnsmith.dedup.cli --pattern-dir <dir> --onnx-model <model.onnx>
    python3 -m nnsmith.dedup.cli --pattern-dir <dir> --gir <gir.pkl> --backend torch
    python3 -m nnsmith.dedup.cli --pattern-dir <dir> --list
"""

import argparse
import os
import pickle
import sys


def _load_onnx_nodes(model_path):
    import onnx
    from nnsmith.dedup.normalizer import NNSMITH_TO_PATTERN_OP, NNSMITH_ATTR_MAP
    model = onnx.load(model_path)
    nodes = []
    for node in model.graph.node:
        input_info = []
        for inp_name in node.input:
            shape = []
            dtype = 'float32'
            for vi in list(model.graph.value_info) + list(model.graph.input):
                if vi.name == inp_name:
                    dims = vi.type.tensor_type.shape.dim
                    shape = [d.dim_value for d in dims]
                    dtype = str(vi.type.tensor_type.elem_type)
                    break
            input_info.append({'shape': shape, 'dtype': dtype})
        output_info = []
        for out_name in node.output:
            shape = []
            dtype = 'float32'
            for vi in list(model.graph.value_info) + list(model.graph.output):
                if vi.name == out_name:
                    dims = vi.type.tensor_type.shape.dim
                    shape = [d.dim_value for d in dims]
                    dtype = str(vi.type.tensor_type.elem_type)
                    break
            output_info.append({'shape': shape, 'dtype': dtype})
        attrs = {}
        for attr in node.attribute:
            if attr.type == 1:
                attrs[attr.name] = attr.f
            elif attr.type == 2:
                attrs[attr.name] = attr.i
            elif attr.type == 4:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == 7:
                attrs[attr.name] = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
        mapped_attrs = {}
        for k, v in attrs.items():
            mapped_attrs[NNSMITH_ATTR_MAP.get(k, k)] = v
        nodes.append({
            'op': NNSMITH_TO_PATTERN_OP.get(node.op_type, node.op_type.upper()),
            'attrs': mapped_attrs,
            'inputs': input_info,
            'outputs': output_info,
            'raw_op': node.op_type,
        })
    return nodes


def main():
    parser = argparse.ArgumentParser(description='nnsmith Pattern Dedup CLI')
    parser.add_argument('--pattern-dir', required=True, help='Pattern JSON directory')
    parser.add_argument('--gettir', '--gir', dest='gir_path', help='Path to a gir.pkl file')
    parser.add_argument('--onnx-model', help='Path to an ONNX model file')
    parser.add_argument('--backend', default='torch', help='nnsmith backend (torch/onnx)')
    parser.add_argument('--compiler', default=None)
    parser.add_argument('--target', default=None)
    parser.add_argument('--list', action='store_true', help='List loaded patterns')
    args = parser.parse_args()

    from nnsmith.dedup import DedupMatcher, normalize_graph, load_patterns
    from nnsmith.dedup.matcher import PatternMatcher

    all_patterns = load_patterns(args.pattern_dir)
    print(f"Loaded {len(all_patterns)} patterns from {args.pattern_dir}")

    if args.list:
        for p in all_patterns:
            ops = ', '.join(n.op for n in p.nodes)
            print(f"  {p.id:40s} {p.compiler:8s}/{str(p.target or '*'):8s}  ops=[{ops}]")
        return

    if args.gir_path:
        with open(args.gir_path, 'rb') as f:
            ir = pickle.load(f)
        matcher = DedupMatcher(args.pattern_dir, args.compiler, args.target, args.backend)
        nodes = normalize_graph(ir, backend=args.backend)
        print(f"\nGraph: {len(nodes)} compute nodes")
        for n in nodes:
            print(f"  {n['op']:20s} attrs={n['attrs']}")
        matched = matcher.matches(ir)
        if matched:
            print(f"\n✅ Matched {len(matched)} patterns:")
            for p in matched:
                print(f"  - {p.id}")
        else:
            print("\n❌ No pattern matched")

    elif args.onnx_model:
        nodes = _load_onnx_nodes(args.onnx_model)
        matcher = PatternMatcher(all_patterns, args.compiler, args.target)
        matched = matcher.match(nodes)
        print(f"\nONNX model: {len(nodes)} nodes")
        if matched:
            print(f"✅ Matched {len(matched)} patterns:")
            for p in matched:
                print(f"  - {p.id}")
        else:
            print("❌ No pattern matched")


if __name__ == '__main__':
    main()