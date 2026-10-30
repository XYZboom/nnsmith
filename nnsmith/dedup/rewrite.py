"""nnsmith GIR 纯静态局部改写（pre-dedup）：命中已知 bug pattern 后改写触发节点，
保留图其余部分，使新程序不再触发该 pattern（与 AIFuzzer 的 rollback-rewrite 机制一致）。

判据：纯静态保形推导（输出 Shape/Dtype 完全一致）且 不再命中任何已知 pattern。
完全不调用任何编译器后端（零编译/执行开销）。
用法：rewrite_gir(gir, dedup_matcher) -> 改写后的 gir 或 None（失败则保留原程序）
"""
import copy

from nnsmith.abstract import op as AOP
from nnsmith.abstract.dtype import DType
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.gir import InstExpr, InstIR

# 触发算子 -> 候选替换（按批量实测优先排序；同 arity）
REDUCE_FAMILY = ['ReduceMean', 'ReduceMax', 'ReduceMin']
REDUCE_EXTRAS = ['Squeeze', 'ArgMax', 'ArgMin']
BINARY_CANDS = ['Add', 'Sub', 'Mul', 'Div', 'Min', 'Max']
TRIG_CANDS = {
    'Sin': ['Cos', 'Tan', 'Atan'],
    'Cos': ['Sin', 'Tan', 'Atan'],
    'Tan': ['Sin', 'Cos', 'Atan'],
    'Atan': ['Sin', 'Cos', 'Tan'],
}
UNARY_CANDS = {
    'Floor': ['Ceil', 'Round', 'Abs'],
    'Ceil': ['Floor', 'Round', 'Abs'],
    'Round': ['Floor', 'Ceil', 'Abs'],
}
# Interpolation-mode swaps are not an established escape from fp16 CPU failures;
# the observed NearestInterp rewrite still produced a wrong result. Keep these
# matches fail-closed until a verified local workaround exists.

def _candidates_for(op_name):
    if op_name.startswith('Reduce'):
        base = [c for c in REDUCE_FAMILY if c != op_name]
        rest = REDUCE_EXTRAS
        return base + [c for c in rest if c != op_name]
    if op_name == 'MatMul':
        return BINARY_CANDS
    if op_name in BINARY_CANDS:
        return [candidate for candidate in BINARY_CANDS if candidate != op_name]
    if op_name in TRIG_CANDS:
        return TRIG_CANDS[op_name]
    if op_name in UNARY_CANDS:
        return UNARY_CANDS[op_name]
    return []

def _rewrite_state(gir):
    return tuple(
        (
            getattr(inst, 'identifier', None),
            type(inst.iexpr.op).__name__,
            repr(getattr(inst.iexpr.op, '__dict__', {})),
        )
        for inst in gir.insts
    )


def _rewrite_inst(gir, idx, cand_name):
    inst = gir.insts[idx]
    Cand = getattr(AOP, cand_name, None)
    if Cand is None:
        return None
    try:
        if isinstance(inst.iexpr.op, AOP.InterpBase):
            new_op = Cand(*inst.iexpr.op.size)
        else:
            new_op = Cand()
    except Exception:
        return None
    in_tensors = []
    for arg in inst.iexpr.args:
        if arg not in gir.vars:
            return None
        t = gir.vars[arg]
        in_tensors.append(AbsTensor(list(t.shape), t.dtype))
    allowed_input_dtypes = getattr(Cand, 'in_dtypes', None)
    if allowed_input_dtypes and tuple(t.dtype for t in in_tensors) not in allowed_input_dtypes:
        return None
    # reduce 候选：保持原 reduce_dim（输出 shape 不变，后续节点一致）
    if hasattr(new_op, 'extra_attrs'):
        mro = [c.__name__ for c in getattr(Cand, '__mro__', ())]
        if 'ReduceBase' in mro:
            old_attrs = getattr(inst.iexpr.op, 'extra_attrs', None)
            if old_attrs is not None and 'reduce_dim' in old_attrs:
                new_op.extra_attrs['reduce_dim'] = old_attrs['reduce_dim']
    try:
        outs = new_op.type_transfer(in_tensors)
        constraints = new_op.checked_requires(in_tensors)
    except Exception:
        return None
    for constraint in constraints:
        try:
            if not bool(constraint):
                return None
        except Exception:
            return None

    new_op.bind_input_like(in_tensors)
    new_op.bind_output_like(outs)

    if len(outs) != inst.n_output():
        return None

    # 纯静态保形检查：严格要求输出张量的数量、形状与数据类型同原节点完全一致。
    # 这只保证图结构兼容；是否真正绕开目标缺陷仍必须通过后端执行验证。
    orig_outs = [gir.vars.get(inst.retval(i)) for i in range(inst.n_output())]
    if any(o is None for o in orig_outs):
        return None
    for new_t, orig_t in zip(outs, orig_outs):
        if list(new_t.shape) != list(orig_t.shape) or new_t.dtype != orig_t.dtype:
            return None

    gir2 = copy.deepcopy(gir)
    inst2 = gir2.insts[idx]
    inst2.iexpr.op = new_op
    for ridx in range(inst2.n_output()):
        name = inst2.retval(ridx)
        out_t = outs[ridx]
        gir2.vars[name] = AbsTensor(list(out_t.shape), out_t.dtype)
    return gir2


def _rewrite_byte_argreduce_with_int64(gir, idx):
    """Widen byte inputs before ArgMin/ArgMax to avoid Inductor CPU wrong-code."""
    inst = gir.insts[idx]
    if type(inst.iexpr.op).__name__ not in ('ArgMin', 'ArgMax') or len(inst.iexpr.args) != 1:
        return None
    source = inst.iexpr.args[0]
    source_tensor = gir.vars.get(source)
    if source_tensor is None or source_tensor.dtype not in (DType.int8, DType.uint8):
        return None

    input_tensor = AbsTensor(list(source_tensor.shape), source_tensor.dtype)
    cast_op = AOP.CastI64()
    try:
        cast_outputs = cast_op.type_transfer([input_tensor])
        constraints = cast_op.checked_requires([input_tensor])
        if len(cast_outputs) != 1 or cast_outputs[0].dtype != DType.int64:
            return None
        if any(not bool(constraint) for constraint in constraints):
            return None
        cast_op.bind_input_like([input_tensor])
        cast_op.bind_output_like(cast_outputs)

        argreduce = copy.deepcopy(inst.iexpr.op)
        rewritten_outputs = argreduce.type_transfer(cast_outputs)
        original_outputs = [gir.vars.get(inst.retval(i)) for i in range(inst.n_output())]
        if len(rewritten_outputs) != len(original_outputs):
            return None
        if any(
            original is None
            or list(rewritten.shape) != list(original.shape)
            or rewritten.dtype != original.dtype
            for rewritten, original in zip(rewritten_outputs, original_outputs)
        ):
            return None

        rewritten_gir = copy.deepcopy(gir)
        rewritten_inst = rewritten_gir.insts[idx]
        cast_inst = rewritten_gir.add_inst(InstExpr(cast_op, (source,)))
        rewritten_inst.iexpr.args = list(rewritten_inst.iexpr.args)
        rewritten_gir.replace_arg(rewritten_inst, 0, cast_inst.retval(), type_check=False)
        rewritten_inst.iexpr.args = tuple(rewritten_inst.iexpr.args)
        rewritten_gir.assert_wellform()
        return rewritten_gir
    except Exception:
        return None


def _rewrite_int_abs_clip_with_int64(gir, idx):
    """Widen an Abs->Clip boundary to avoid int8 C++ promotion mismatch."""
    inst = gir.insts[idx]
    if type(inst.iexpr.op).__name__ != 'Clip' or len(inst.iexpr.args) != 1:
        return None
    source = inst.iexpr.args[0]
    source_tensor = gir.vars.get(source)
    if source_tensor is None or source_tensor.dtype not in (DType.int8, DType.int16):
        return None
    producer_id, _ = InstIR.var_inst_idx(source)
    producer = gir.find_inst_by_id(producer_id)
    if producer is None or type(producer.iexpr.op).__name__ != 'Abs':
        return None

    original_output = gir.vars.get(inst.retval())
    if original_output is None:
        return None
    rewritten = copy.deepcopy(gir)
    inst = rewritten.find_inst_by_id(inst.identifier)
    if inst is None:
        return None
    old_output = inst.retval()
    old_users = list(inst.users[0])
    source_tensor = rewritten.vars[source]

    try:
        cast_up = AOP.CastI64()
        cast_up_inputs = [AbsTensor(list(source_tensor.shape), source_tensor.dtype)]
        cast_up_outputs = cast_up.type_transfer(cast_up_inputs)
        cast_up_constraints = cast_up.checked_requires(cast_up_inputs)
        if len(cast_up_outputs) != 1 or cast_up_outputs[0].dtype != DType.int64:
            return None
        if any(not bool(constraint) for constraint in cast_up_constraints):
            return None
        cast_up.bind_input_like(cast_up_inputs)
        cast_up.bind_output_like(cast_up_outputs)
        cast_up_inst = rewritten.add_inst(InstExpr(cast_up, (source,)))

        inst.iexpr.args = list(inst.iexpr.args)
        rewritten.replace_arg(inst, 0, cast_up_inst.retval(), type_check=False)
        inst.iexpr.args = tuple(inst.iexpr.args)

        clip_op = copy.deepcopy(inst.iexpr.op)
        clip_inputs = [AbsTensor(list(cast_up_outputs[0].shape), DType.int64)]
        clip_outputs = clip_op.type_transfer(clip_inputs)
        clip_constraints = clip_op.checked_requires(clip_inputs)
        if len(clip_outputs) != 1 or clip_outputs[0].dtype != DType.int64:
            return None
        if list(clip_outputs[0].shape) != list(original_output.shape):
            return None
        if any(not bool(constraint) for constraint in clip_constraints):
            return None
        clip_op.bind_input_like(clip_inputs)
        clip_op.bind_output_like(clip_outputs)
        inst.iexpr.op = clip_op
        rewritten.vars[old_output] = AbsTensor(
            list(clip_outputs[0].shape), clip_outputs[0].dtype
        )

        cast_down = AOP.Cast(source_tensor.dtype)
        cast_down_inputs = [rewritten.vars[old_output]]
        cast_down_outputs = cast_down.type_transfer(cast_down_inputs)
        cast_down_constraints = cast_down.checked_requires(cast_down_inputs)
        if len(cast_down_outputs) != 1:
            return None
        if (
            list(cast_down_outputs[0].shape) != list(original_output.shape)
            or cast_down_outputs[0].dtype != original_output.dtype
        ):
            return None
        if any(not bool(constraint) for constraint in cast_down_constraints):
            return None
        cast_down.bind_input_like(cast_down_inputs)
        cast_down.bind_output_like(cast_down_outputs)
        cast_down_inst = rewritten.add_inst(InstExpr(cast_down, (old_output,)))

        for user in old_users:
            for arg_idx, arg in enumerate(list(user.iexpr.args)):
                if arg == old_output:
                    user.iexpr.args = list(user.iexpr.args)
                    rewritten.replace_arg(
                        user, arg_idx, cast_down_inst.retval(), type_check=False
                    )
                    user.iexpr.args = tuple(user.iexpr.args)
        rewritten.assert_wellform()
        return rewritten
    except Exception:
        return None


def rewrite_gir(gir, dedup_matcher, max_candidates=6):
    """Locally rewrite matched nodes on a copy; failure leaves the input untouched."""
    if dedup_matcher is None:
        return None

    working = copy.deepcopy(gir)
    changed = False
    seen_states = {_rewrite_state(working)}
    max_rewrites = max(1, len(gir.insts)) * max(1, max_candidates)

    for _ in range(max_rewrites):
        occurrences = dedup_matcher.match_occurrences(working)
        if not occurrences:
            return working if changed else None

        accepted = False
        for pattern, matched_nodes in occurrences:
            inst_ids = tuple(node.get('inst_id') for node in matched_nodes)
            if any(inst_id is None for inst_id in inst_ids):
                continue
            occurrence_key = (pattern.id, inst_ids)

            # Rewrite the terminal matched node first, preserving all other nodes.
            for node in reversed(matched_nodes):
                inst_idx = next(
                    (idx for idx, inst in enumerate(working.insts)
                     if inst.identifier == node.get('inst_id')),
                    None,
                )
                if inst_idx is None:
                    continue
                op_name = type(working.insts[inst_idx].iexpr.op).__name__
                candidate_rewrites = []
                if pattern.id == 'pt-inductor-cpu-byte-argreduce-tie':
                    widened = _rewrite_byte_argreduce_with_int64(working, inst_idx)
                    if widened is not None:
                        candidate_rewrites.append(widened)
                elif pattern.id == 'pt-inductor-int-clip-minmax':
                    widened_clip = _rewrite_int_abs_clip_with_int64(working, inst_idx)
                    if widened_clip is not None:
                        candidate_rewrites.append(widened_clip)
                candidate_rewrites.extend(
                    _rewrite_inst(working, inst_idx, candidate_name)
                    for candidate_name in _candidates_for(op_name)[:max_candidates]
                )
                for rewritten in candidate_rewrites:
                    if rewritten is None:
                        continue
                    remaining = {
                        (remaining_pattern.id,
                         tuple(item.get('inst_id') for item in remaining_nodes))
                        for remaining_pattern, remaining_nodes
                        in dedup_matcher.match_occurrences(rewritten)
                    }
                    candidate_state = _rewrite_state(rewritten)
                    if occurrence_key not in remaining and candidate_state not in seen_states:
                        working = rewritten
                        seen_states.add(candidate_state)
                        changed = True
                        accepted = True
                        break
                if accepted:
                    break
            if accepted:
                break

        if not accepted:
            # Preserve successful rewrites for other occurrences when this one
            # has no safe candidate; the caller records the residual matches.
            return working if changed else None

    # At the rewrite limit, return the changed copy even if some occurrences
    # remain; the caller records those residual matches. An unchanged input is
    # still reported as a failed rewrite and remains the caller's original GIR.
    return working if changed else None
