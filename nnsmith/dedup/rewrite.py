"""nnsmith GIR 局部改写（pre-dedup）：命中已知 bug pattern 后改写触发节点，
保留图其余部分，使新程序不再触发该 pattern（与 AIFuzzer 的 rollback-rewrite 同哲学）。

判据：改写后 relax.build(cuda) 完全成功 且 不再命中任何 pattern。
用法：rewrite_gir(gir) -> 改写后的 gir 或 None（失败则调用方接受原程序照跑）
"""
import copy
import random

import tvm
from nnsmith.abstract import op as AOP
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.gir import InstIR, InstExpr
from nnsmith.materialize.tvm import TVMModel

# 触发算子 -> 候选替换（按批量实测优先排序；同 arity）
REDUCE_FAMILY = ['ReduceSum', 'ReduceMean', 'ReduceMax', 'ReduceMin']
REDUCE_EXTRAS = ['Squeeze', 'ArgMax', 'ArgMin', 'ReduceProd']
BINARY_CANDS = ['Add', 'Sub', 'Mul', 'Div', 'Min', 'Max', 'Where', 'Equal', 'Greater', 'Less']

# pattern op 名 -> nnsmith 类名（反向映射，用于定位触发节点）
PATTERN_TO_NNSMITH = {
    'REDUCE_SUM': 'ReduceSum', 'REDUCE_MEAN': 'ReduceMean', 'REDUCE_MAX': 'ReduceMax',
    'REDUCE_MIN': 'ReduceMin', 'REDUCE_PROD': 'ReduceProd', 'MATMUL': 'MatMul',
    'ADD': 'Add', 'SUB': 'Subtract', 'SUBTRACT': 'Subtract', 'DIVIDE': 'Div',
    'MULTIPLY': 'Mul', 'MAXIMUM': 'Max', 'MINIMUM': 'Min', 'WHERE': 'Where',
    'GREATER': 'Greater', 'SOFTMAX': 'Softmax', 'SIGMOID': 'Sigmoid',
    'CONCAT': 'Concat2', 'EXPAND': 'ExpandLast2', 'REFLECT_PAD': 'ReflectPad',
    'CONST_PAD': 'ConstPad', 'REPLICATE_PAD': 'ReplicatePad', 'SLICE': 'Slice',
    'BICUBIC_INTERP': 'BicubicInterp', 'TRILINEAR_INTERP': 'TrilinearInterp',
    'AVG_POOL2D': 'AvgPool2d', 'MAX_POOL2D': 'MaxPool2d', 'GELU': 'GELU',
    'RELU': 'ReLU', 'COS': 'Cos', 'FLOOR': 'Floor', 'CEIL': 'Ceil', 'ROUND': 'Round',
    'NEG': 'Neg', 'ABS': 'Abs', 'SIGN': 'SIGN', 'CLIP': 'CLIP', 'CAST': 'Cast',
}

def _candidates_for(op_name):
    if op_name.startswith('Reduce'):
        base = [c for c in REDUCE_FAMILY if c != op_name]
        rest = REDUCE_EXTRAS
        return base + [c for c in rest if c != op_name]
    if op_name == 'MatMul':
        return BINARY_CANDS
    return []

def _find_trigger_inst(gir, pat_op):
    """在 gir 中找与 pattern 首个节点 op 对应的实际节点"""
    nnsmith_name = PATTERN_TO_NNSMITH.get(pat_op)
    for inst in gir.insts:
        if nnsmith_name is None:
            # 退而求其次：找同 arity 同族算子
            n = type(inst.iexpr.op).__name__
            if pat_op.startswith('REDUCE_') and n.startswith('Reduce'):
                return inst
            continue
        if type(inst.iexpr.op).__name__ == nnsmith_name:
            return inst
    return None

def _build_ok(gir):
    try:
        model = TVMModel.from_gir(gir)
        mod = model.native_model
        if mod is None:
            return False
        with tvm.transform.PassContext(opt_level=3):
            tvm.relax.build(mod, target="cuda")
        return True
    except Exception:
        return False

def _rewrite_inst(gir, idx, cand_name):
    inst = gir.insts[idx]
    Cand = getattr(AOP, cand_name, None)
    if Cand is None:
        return None
    try:
        new_op = Cand()
    except Exception:
        return None
    in_tensors = []
    for arg in inst.iexpr.args:
        if arg not in gir.vars:
            return None
        t = gir.vars[arg]
        in_tensors.append(AbsTensor(list(t.shape), t.dtype))
    # reduce 候选：保持原 reduce_dim（输出 shape 不变，后续节点才一致，
    # build 才不会因形状不匹配失败）。换 axis 会改输出 shape -> 破坏图一致性。
    if hasattr(new_op, 'extra_attrs'):
        mro = [c.__name__ for c in getattr(Cand, '__mro__', ())]
        if 'ReduceBase' in mro:
            old = inst.iexpr.op.extra_attrs.get('reduce_dim', None) if getattr(inst.iexpr.op, 'extra_attrs', None) else None
            if old is not None:
                new_op.extra_attrs['reduce_dim'] = old
    try:
        outs = new_op.type_transfer(in_tensors)
    except Exception:
        return None
    gir2 = copy.deepcopy(gir)
    inst2 = gir2.insts[idx]
    inst2.iexpr.op = new_op
    for ridx in range(inst2.n_output()):
        name = inst2.retval(ridx)
        if name in gir2.vars and ridx < len(outs):
            out_t = outs[ridx]
            gir2.vars[name] = AbsTensor(list(out_t.shape), out_t.dtype)
    return gir2

def rewrite_gir(gir, dedup_matcher, max_candidates=6):
    """改写 gir 中触发节点，返回改写后的 gir（build OK 且不再命中）或 None"""
    if dedup_matcher is None:
        return None
    matched = dedup_matcher.matches(gir)
    if not matched:
        return None
    # 优先找含"与崩溃直接相关"算子（REDUCE_*/MATMUL）的 pattern 作为改写目标；
    # 改前置 elementwise 节点（如 bind-26 的 MULTIPLY）不会消除 reduce 触发的崩溃。
    first_pat = None
    for pat in matched:
        if any(PATTERN_TO_NNSMITH.get(n.op, '').startswith('Reduce')
               or PATTERN_TO_NNSMITH.get(n.op) == 'MatMul' for n in pat.nodes):
            first_pat = pat
            break
    if first_pat is None:
        first_pat = matched[0]
    # 在选定 pattern 节点里优先选 reduce/matmul，否则用第一个节点
    target_op = None
    for n in first_pat.nodes:
        nm = PATTERN_TO_NNSMITH.get(n.op)
        if nm and (nm.startswith('Reduce') or nm == 'MatMul'):
            target_op = n.op
            break
    if target_op is None:
        target_op = first_pat.nodes[0].op if first_pat.nodes else None
    if target_op is None:
        return None
    inst = _find_trigger_inst(gir, target_op)
    if inst is None:
        return None
    idx = gir.insts.index(inst)
    op_name = type(inst.iexpr.op).__name__
    cands = _candidates_for(op_name)[:max_candidates]
    for cand in cands:
        g2 = _rewrite_inst(gir, idx, cand)
        if g2 is None:
            continue
        if _build_ok(g2) and not dedup_matcher.matches(g2):
            return g2
    return None