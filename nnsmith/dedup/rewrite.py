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


def _has_abs_input_ancestor(gir, inst, max_depth=4):
    """Find Abs on a bounded value path, including Where's selected-value arms."""
    pending = [(arg, 0) for arg in inst.iexpr.args]
    visited = set()
    while pending:
        var, depth = pending.pop()
        if depth >= max_depth:
            continue
        producer_id, _ = InstIR.var_inst_idx(var)
        if producer_id in visited:
            continue
        visited.add(producer_id)
        producer = gir.find_inst_by_id(producer_id)
        if producer is None:
            continue
        op_name = type(producer.iexpr.op).__name__
        if op_name == 'Abs':
            return True
        args = producer.iexpr.args
        if len(args) == 1:
            pending.append((args[0], depth + 1))
        elif op_name == 'Where' and len(args) == 3:
            # The first Where input is a predicate, not a value contributor.
            # Only the selected data arms can carry Abs into this value.
            pending.extend((parent, depth + 1) for parent in args[1:])
    return False


def _has_direct_int16_neg_input_clip(gir, inst):
    if type(inst.iexpr.op).__name__ != 'Clip' or len(inst.iexpr.args) != 1:
        return False
    source_tensor = gir.vars.get(inst.iexpr.args[0])
    if source_tensor is None or source_tensor.dtype != DType.int16:
        return False
    neg_id, _ = InstIR.var_inst_idx(inst.iexpr.args[0])
    neg = gir.find_inst_by_id(neg_id)
    if neg is None or type(neg.iexpr.op).__name__ != 'Neg' or len(neg.iexpr.args) != 1:
        return False
    parent_id, _ = InstIR.var_inst_idx(neg.iexpr.args[0])
    parent = gir.find_inst_by_id(parent_id)
    return parent is not None and type(parent.iexpr.op).__name__ == 'Input'


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


def _rewrite_narrow_binary_minmax_with_int64(gir, idx):
    """Widen narrow integer Min/Max operands, preserving the operation semantics."""
    inst = gir.insts[idx]
    op_name = type(inst.iexpr.op).__name__
    if op_name not in ('Min', 'Max') or len(inst.iexpr.args) != 2 or inst.n_output() != 1:
        return None
    input_vars = list(inst.iexpr.args)
    input_tensors = [gir.vars.get(var) for var in input_vars]
    if any(tensor is None for tensor in input_tensors):
        return None
    dtype = input_tensors[0].dtype
    if dtype not in (DType.int8, DType.int16) or any(tensor.dtype != dtype for tensor in input_tensors):
        return None
    original_output = gir.vars.get(inst.retval())
    if original_output is None or original_output.dtype != dtype:
        return None
    old_users = list(inst.users[0])

    try:
        rewritten = copy.deepcopy(gir)
        rewritten_inst = rewritten.find_inst_by_id(inst.identifier)
        if rewritten_inst is None:
            return None
        widened_vars = []
        widened_inputs = []
        for arg in input_vars:
            source_tensor = rewritten.vars.get(arg)
            if source_tensor is None or source_tensor.dtype != dtype:
                return None
            cast_up = AOP.CastI64()
            cast_inputs = [AbsTensor(list(source_tensor.shape), source_tensor.dtype)]
            cast_outputs = cast_up.type_transfer(cast_inputs)
            if len(cast_outputs) != 1 or cast_outputs[0].dtype != DType.int64:
                return None
            if any(not bool(c) for c in cast_up.checked_requires(cast_inputs)):
                return None
            cast_up.bind_input_like(cast_inputs)
            cast_up.bind_output_like(cast_outputs)
            cast_inst = rewritten.add_inst(InstExpr(cast_up, (arg,)))
            widened_vars.append(cast_inst.retval())
            widened_inputs.append(cast_outputs[0])

        wide_op = copy.deepcopy(rewritten_inst.iexpr.op)
        wide_outputs = wide_op.type_transfer(widened_inputs)
        if (
            len(wide_outputs) != 1
            or list(wide_outputs[0].shape) != list(original_output.shape)
            or wide_outputs[0].dtype != DType.int64
        ):
            return None
        if any(not bool(c) for c in wide_op.checked_requires(widened_inputs)):
            return None
        wide_op.bind_input_like(widened_inputs)
        wide_op.bind_output_like(wide_outputs)

        rewritten_inst.iexpr.args = list(rewritten_inst.iexpr.args)
        for arg_idx, widened_var in enumerate(widened_vars):
            rewritten.replace_arg(rewritten_inst, arg_idx, widened_var, type_check=False)
        rewritten_inst.iexpr.args = tuple(rewritten_inst.iexpr.args)
        rewritten_inst.iexpr.op = wide_op
        old_output = rewritten_inst.retval()
        rewritten.vars[old_output] = AbsTensor(list(wide_outputs[0].shape), DType.int64)

        cast_down = AOP.Cast(dtype)
        cast_down_inputs = [rewritten.vars[old_output]]
        cast_down_outputs = cast_down.type_transfer(cast_down_inputs)
        if (
            len(cast_down_outputs) != 1
            or list(cast_down_outputs[0].shape) != list(original_output.shape)
            or cast_down_outputs[0].dtype != dtype
        ):
            return None
        if any(not bool(c) for c in cast_down.checked_requires(cast_down_inputs)):
            return None
        cast_down.bind_input_like(cast_down_inputs)
        cast_down.bind_output_like(cast_down_outputs)
        cast_down_inst = rewritten.add_inst(InstExpr(cast_down, (old_output,)))
        for old_user in old_users:
            user = rewritten.find_inst_by_id(old_user.identifier)
            if user is None:
                return None
            for arg_idx, arg in enumerate(list(user.iexpr.args)):
                if arg == old_output:
                    user.iexpr.args = list(user.iexpr.args)
                    rewritten.replace_arg(user, arg_idx, cast_down_inst.retval(), type_check=False)
                    user.iexpr.args = tuple(user.iexpr.args)
        rewritten.assert_wellform()
        return rewritten
    except Exception:
        return None


def _rewrite_narrow_reduce_minmax_with_int64(gir, idx, min_axis_len=2):
    """Widen a narrow ReduceMin/ReduceMax input and cast its result back exactly."""
    inst = gir.insts[idx]
    if type(inst.iexpr.op).__name__ not in ('ReduceMin', 'ReduceMax') or len(inst.iexpr.args) != 1:
        return None
    source = inst.iexpr.args[0]
    source_tensor = gir.vars.get(source)
    if source_tensor is None or source_tensor.dtype not in (DType.int8, DType.uint8, DType.int16):
        return None
    reduce_dim = getattr(inst.iexpr.op, 'extra_attrs', {}).get('reduce_dim')
    if reduce_dim is None:
        return None
    try:
        axis = int(reduce_dim)
        if axis < 0:
            axis += len(source_tensor.shape)
        if (
            axis < 0
            or axis >= len(source_tensor.shape)
            or int(source_tensor.shape[axis]) < min_axis_len
        ):
            return None
    except (TypeError, ValueError):
        return None

    original_outputs = [gir.vars.get(inst.retval(i)) for i in range(inst.n_output())]
    if not original_outputs or any(output is None for output in original_outputs):
        return None
    old_users = list(inst.users[0])
    unary_chain = _collect_narrow_unary_nodes(gir, source, source_tensor.dtype)
    if unary_chain is None:
        return None
    rewritten = copy.deepcopy(gir)
    for unary_id in reversed(unary_chain[0]):
        if not _widen_narrow_unary_preserving_wrap(rewritten, unary_id):
            return None
    rewritten_inst = rewritten.find_inst_by_id(inst.identifier)
    if rewritten_inst is None:
        return None
    source = rewritten_inst.iexpr.args[0]
    source_tensor = rewritten.vars.get(source)
    if source_tensor is None:
        return None

    try:
        cast_up = AOP.CastI64()
        cast_inputs = [AbsTensor(list(source_tensor.shape), source_tensor.dtype)]
        cast_outputs = cast_up.type_transfer(cast_inputs)
        if len(cast_outputs) != 1 or cast_outputs[0].dtype != DType.int64:
            return None
        if any(not bool(c) for c in cast_up.checked_requires(cast_inputs)):
            return None
        cast_up.bind_input_like(cast_inputs)
        cast_up.bind_output_like(cast_outputs)

        reduce_op = copy.deepcopy(rewritten_inst.iexpr.op)
        reduced_outputs = reduce_op.type_transfer(cast_outputs)
        if len(reduced_outputs) != len(original_outputs):
            return None
        if any(
            list(reduced.shape) != list(original.shape)
            or reduced.dtype != DType.int64
            for reduced, original in zip(reduced_outputs, original_outputs)
        ):
            return None
        if any(not bool(c) for c in reduce_op.checked_requires(cast_outputs)):
            return None
        reduce_op.bind_input_like(cast_outputs)
        reduce_op.bind_output_like(reduced_outputs)

        cast_inst = rewritten.add_inst(InstExpr(cast_up, (source,)))
        rewritten_inst.iexpr.args = list(rewritten_inst.iexpr.args)
        rewritten.replace_arg(rewritten_inst, 0, cast_inst.retval(), type_check=False)
        rewritten_inst.iexpr.args = tuple(rewritten_inst.iexpr.args)
        rewritten_inst.iexpr.op = reduce_op
        old_output = rewritten_inst.retval()
        rewritten.vars[old_output] = AbsTensor(list(reduced_outputs[0].shape), DType.int64)

        output_dtype = original_outputs[0].dtype
        if output_dtype not in (DType.int8, DType.uint8, DType.int16):
            return None
        cast_down = AOP.Cast(output_dtype)
        cast_down_inputs = [rewritten.vars[old_output]]
        cast_down_outputs = cast_down.type_transfer(cast_down_inputs)
        if (
            len(cast_down_outputs) != 1
            or list(cast_down_outputs[0].shape) != list(original_outputs[0].shape)
            or cast_down_outputs[0].dtype != output_dtype
        ):
            return None
        if any(not bool(c) for c in cast_down.checked_requires(cast_down_inputs)):
            return None
        cast_down.bind_input_like(cast_down_inputs)
        cast_down.bind_output_like(cast_down_outputs)
        cast_down_inst = rewritten.add_inst(InstExpr(cast_down, (old_output,)))

        for old_user in old_users:
            user = rewritten.find_inst_by_id(old_user.identifier)
            if user is None:
                return None
            for arg_idx, arg in enumerate(list(user.iexpr.args)):
                if arg == old_output:
                    user.iexpr.args = list(user.iexpr.args)
                    rewritten.replace_arg(user, arg_idx, cast_down_inst.retval(), type_check=False)
                    user.iexpr.args = tuple(user.iexpr.args)
        rewritten.assert_wellform()
        return rewritten
    except Exception:
        return None


def _collect_narrow_unary_nodes(gir, value, dtype):
    """Return promotion-sensitive unary IDs on a safe single-input ancestor path."""
    view_ops = {
        'Concat1', 'Slice', 'Reshape', 'Transpose', 'Squeeze', 'Flatten',
        'Expand', 'ExpandLast1', 'ExpandLast2', 'ExpandLast3', 'ExpandLast4',
    }
    unary_ids = []
    visited = set()
    has_narrowing_cast = False
    current = value

    for _ in range(len(gir.insts) + 1):
        producer_id, _ = InstIR.var_inst_idx(current)
        if producer_id in visited:
            return None
        visited.add(producer_id)
        producer = gir.find_inst_by_id(producer_id)
        tensor = gir.vars.get(current)
        if producer is None or tensor is None:
            return None
        op = producer.iexpr.op
        op_name = type(op).__name__

        if op_name in ('Input', 'Constant'):
            return unary_ids, has_narrowing_cast
        if op_name in view_ops and len(producer.iexpr.args) == 1:
            current = producer.iexpr.args[0]
            continue
        if op_name in ('Abs', 'Neg') and tensor.dtype == dtype and len(producer.iexpr.args) == 1:
            input_tensor = gir.vars.get(producer.iexpr.args[0])
            if input_tensor is None or input_tensor.dtype != dtype:
                return None
            unary_ids.append(producer.identifier)
            current = producer.iexpr.args[0]
            continue
        if op_name == 'Cast' and len(producer.iexpr.args) == 1:
            input_tensor = gir.vars.get(producer.iexpr.args[0])
            if input_tensor is None:
                return None
            if tensor.dtype == dtype and input_tensor.dtype != dtype:
                return unary_ids, True
            if tensor.dtype == input_tensor.dtype:
                current = producer.iexpr.args[0]
                continue
            return unary_ids, has_narrowing_cast
        # Stop at the first unmodified producer. Casts are inserted only around
        # the collected Abs/Neg nodes, so this producer retains its narrow semantics.
        return unary_ids, has_narrowing_cast

    return None


def _widen_narrow_unary_preserving_wrap(gir, inst_id):
    """Compute int8/int16 Abs/Neg in int64, then restore the original wrap width."""
    inst = gir.find_inst_by_id(inst_id)
    if (
        inst is None
        or type(inst.iexpr.op).__name__ not in ('Abs', 'Neg')
        or len(inst.iexpr.args) != 1
        or inst.n_output() != 1
    ):
        return False
    input_var = inst.iexpr.args[0]
    output_var = inst.retval()
    input_tensor = gir.vars.get(input_var)
    output_tensor = gir.vars.get(output_var)
    if (
        input_tensor is None
        or output_tensor is None
        or input_tensor.dtype not in (DType.int8, DType.int16)
        or input_tensor.dtype != output_tensor.dtype
    ):
        return False
    old_users = list(inst.users[0])

    try:
        cast_up = AOP.CastI64()
        cast_up_inputs = [AbsTensor(list(input_tensor.shape), input_tensor.dtype)]
        cast_up_outputs = cast_up.type_transfer(cast_up_inputs)
        if len(cast_up_outputs) != 1 or cast_up_outputs[0].dtype != DType.int64:
            return False
        if any(not bool(c) for c in cast_up.checked_requires(cast_up_inputs)):
            return False
        cast_up.bind_input_like(cast_up_inputs)
        cast_up.bind_output_like(cast_up_outputs)
        cast_up_inst = gir.add_inst(InstExpr(cast_up, (input_var,)))

        inst.iexpr.args = list(inst.iexpr.args)
        gir.replace_arg(inst, 0, cast_up_inst.retval(), type_check=False)
        inst.iexpr.args = tuple(inst.iexpr.args)

        wide_op = copy.deepcopy(inst.iexpr.op)
        wide_outputs = wide_op.type_transfer(cast_up_outputs)
        if (
            len(wide_outputs) != 1
            or list(wide_outputs[0].shape) != list(output_tensor.shape)
            or wide_outputs[0].dtype != DType.int64
        ):
            return False
        if any(not bool(c) for c in wide_op.checked_requires(cast_up_outputs)):
            return False
        wide_op.bind_input_like(cast_up_outputs)
        wide_op.bind_output_like(wide_outputs)
        inst.iexpr.op = wide_op
        gir.vars[output_var] = AbsTensor(list(wide_outputs[0].shape), DType.int64)

        cast_down = AOP.Cast(input_tensor.dtype)
        cast_down_inputs = [gir.vars[output_var]]
        cast_down_outputs = cast_down.type_transfer(cast_down_inputs)
        if (
            len(cast_down_outputs) != 1
            or list(cast_down_outputs[0].shape) != list(output_tensor.shape)
            or cast_down_outputs[0].dtype != output_tensor.dtype
        ):
            return False
        if any(not bool(c) for c in cast_down.checked_requires(cast_down_inputs)):
            return False
        cast_down.bind_input_like(cast_down_inputs)
        cast_down.bind_output_like(cast_down_outputs)
        cast_down_inst = gir.add_inst(InstExpr(cast_down, (output_var,)))

        for old_user in old_users:
            user = gir.find_inst_by_id(old_user.identifier)
            if user is None:
                return False
            for arg_idx, arg in enumerate(list(user.iexpr.args)):
                if arg == output_var:
                    user.iexpr.args = list(user.iexpr.args)
                    gir.replace_arg(user, arg_idx, cast_down_inst.retval(), type_check=False)
                    user.iexpr.args = tuple(user.iexpr.args)
        gir.assert_wellform()
        return True
    except Exception:
        return False


def _rewrite_where_abs_before_reduce(gir, reduce_idx, abs_inst_id):
    """Restore an Abs branch's narrow dtype before widening a downstream reduction."""
    if reduce_idx < 0 or reduce_idx >= len(gir.insts):
        return None
    reduce_inst_id = gir.insts[reduce_idx].identifier
    rewritten = copy.deepcopy(gir)
    if not _widen_narrow_unary_preserving_wrap(rewritten, abs_inst_id):
        return None
    rewritten_idx = next(
        (idx for idx, inst in enumerate(rewritten.insts) if inst.identifier == reduce_inst_id),
        None,
    )
    if rewritten_idx is None:
        return None
    return _rewrite_narrow_reduce_minmax_with_int64(rewritten, rewritten_idx)


def _rewrite_int16_argreduce_after_safe_producer(gir, idx):
    """Widen only after preserving every int16 wrap in the ArgMin/ArgMax input path."""
    inst = gir.insts[idx]
    if type(inst.iexpr.op).__name__ not in ('ArgMin', 'ArgMax') or len(inst.iexpr.args) != 1:
        return None
    source = inst.iexpr.args[0]
    source_tensor = gir.vars.get(source)
    if source_tensor is None or source_tensor.dtype != DType.int16:
        return None
    producer_id, _ = InstIR.var_inst_idx(source)
    producer = gir.find_inst_by_id(producer_id)
    producer_name = type(producer.iexpr.op).__name__ if producer is not None else None
    if producer_name in ('Abs', 'Concat1', 'Cast', 'Flatten'):
        pass
    elif producer_name in ('ExpandLast4', 'Slice', 'ReduceMin', 'ReduceMax'):
        if producer is None or len(producer.iexpr.args) != 1:
            return None
        upstream_id, _ = InstIR.var_inst_idx(producer.iexpr.args[0])
        upstream = gir.find_inst_by_id(upstream_id)
        upstream_name = type(upstream.iexpr.op).__name__ if upstream is not None else None
        if upstream_name != 'Abs':
            if producer_name != 'ReduceMin' or upstream_name != 'Squeeze':
                return None
            if upstream is None or len(upstream.iexpr.args) != 1:
                return None
            ancestor_id, _ = InstIR.var_inst_idx(upstream.iexpr.args[0])
            ancestor = gir.find_inst_by_id(ancestor_id)
            if ancestor is None or type(ancestor.iexpr.op).__name__ != 'Abs':
                return None
    else:
        return None
    unary_nodes = _collect_narrow_unary_nodes(gir, source, DType.int16)
    if unary_nodes is None:
        return None

    original_outputs = [gir.vars.get(inst.retval(i)) for i in range(inst.n_output())]
    if any(output is None for output in original_outputs):
        return None
    rewritten = copy.deepcopy(gir)
    for unary_id in reversed(unary_nodes[0]):
        if not _widen_narrow_unary_preserving_wrap(rewritten, unary_id):
            return None
    rewritten_inst = rewritten.find_inst_by_id(inst.identifier)
    if rewritten_inst is None:
        return None
    source = rewritten_inst.iexpr.args[0]
    source_tensor = rewritten.vars.get(source)
    if source_tensor is None or source_tensor.dtype != DType.int16:
        return None

    try:
        cast_op = AOP.CastI64()
        cast_inputs = [AbsTensor(list(source_tensor.shape), source_tensor.dtype)]
        cast_outputs = cast_op.type_transfer(cast_inputs)
        cast_constraints = cast_op.checked_requires(cast_inputs)
        if len(cast_outputs) != 1 or cast_outputs[0].dtype != DType.int64:
            return None
        if any(not bool(constraint) for constraint in cast_constraints):
            return None
        cast_op.bind_input_like(cast_inputs)
        cast_op.bind_output_like(cast_outputs)

        argreduce_op = copy.deepcopy(rewritten_inst.iexpr.op)
        widened_outputs = argreduce_op.type_transfer(cast_outputs)
        argreduce_constraints = argreduce_op.checked_requires(cast_outputs)
        if any(not bool(constraint) for constraint in argreduce_constraints):
            return None
        if len(widened_outputs) != len(original_outputs):
            return None
        for widened, original in zip(widened_outputs, original_outputs):
            if (
                list(widened.shape) != list(original.shape)
                or widened.dtype != original.dtype
            ):
                return None
        argreduce_op.bind_input_like(cast_outputs)
        argreduce_op.bind_output_like(widened_outputs)

        cast_inst = rewritten.add_inst(InstExpr(cast_op, (source,)))
        rewritten_inst.iexpr.args = list(rewritten_inst.iexpr.args)
        rewritten.replace_arg(rewritten_inst, 0, cast_inst.retval(), type_check=False)
        rewritten_inst.iexpr.args = tuple(rewritten_inst.iexpr.args)
        rewritten_inst.iexpr.op = argreduce_op
        rewritten.assert_wellform()
        return rewritten
    except Exception:
        return None


def _rewrite_abs_neg_squeeze_clip_with_int64(gir, idx):
    """Widen Clip after explicitly restoring narrow Abs/Neg wrap semantics."""
    inst = gir.insts[idx]
    if type(inst.iexpr.op).__name__ != 'Clip' or len(inst.iexpr.args) != 1:
        return None
    source = inst.iexpr.args[0]
    source_tensor = gir.vars.get(source)
    if source_tensor is None or source_tensor.dtype not in (DType.int8, DType.int16):
        return None
    chain = _collect_narrow_unary_nodes(gir, source, source_tensor.dtype)
    if chain is None or not chain[0] and not chain[1]:
        return None
    for position, unary_id in enumerate(chain[0]):
        unary_inst = gir.find_inst_by_id(unary_id)
        if unary_inst is None:
            return None
        if type(unary_inst.iexpr.op).__name__ == 'Neg':
            neg_input = gir.vars.get(unary_inst.iexpr.args[0])
            if neg_input is None:
                return None
            if neg_input.dtype == DType.int16:
                continue
            # int8 Neg -> Clip alone remains fail-closed. When a downstream
            # Abs is present, each unary op is widened and cast back separately,
            # preserving the int8 wrap at both Neg and Abs boundaries.
            if neg_input.dtype != DType.int8 or not any(
                type(gir.find_inst_by_id(downstream_id).iexpr.op).__name__ == 'Abs'
                for downstream_id in chain[0][:position]
                if gir.find_inst_by_id(downstream_id) is not None
            ):
                return None
    original_output = gir.vars.get(inst.retval())
    if original_output is None:
        return None

    rewritten = copy.deepcopy(gir)
    for unary_id in reversed(chain[0]):
        if not _widen_narrow_unary_preserving_wrap(rewritten, unary_id):
            return None
    inst = rewritten.find_inst_by_id(inst.identifier)
    if inst is None:
        return None
    source = inst.iexpr.args[0]
    source_tensor = rewritten.vars.get(source)
    if source_tensor is None or source_tensor.dtype not in (DType.int8, DType.int16):
        return None
    old_output = inst.retval()
    old_users = list(inst.users[0])

    try:
        cast_up = AOP.CastI64()
        cast_up_inputs = [AbsTensor(list(source_tensor.shape), source_tensor.dtype)]
        cast_up_outputs = cast_up.type_transfer(cast_up_inputs)
        if len(cast_up_outputs) != 1 or cast_up_outputs[0].dtype != DType.int64:
            return None
        if any(not bool(c) for c in cast_up.checked_requires(cast_up_inputs)):
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
        if (
            len(clip_outputs) != 1
            or clip_outputs[0].dtype != DType.int64
            or list(clip_outputs[0].shape) != list(original_output.shape)
        ):
            return None
        if any(not bool(c) for c in clip_op.checked_requires(clip_inputs)):
            return None
        clip_op.bind_input_like(clip_inputs)
        clip_op.bind_output_like(clip_outputs)
        inst.iexpr.op = clip_op
        rewritten.vars[old_output] = AbsTensor(list(clip_outputs[0].shape), DType.int64)

        cast_down = AOP.Cast(source_tensor.dtype)
        cast_down_inputs = [rewritten.vars[old_output]]
        cast_down_outputs = cast_down.type_transfer(cast_down_inputs)
        if (
            len(cast_down_outputs) != 1
            or list(cast_down_outputs[0].shape) != list(original_output.shape)
            or cast_down_outputs[0].dtype != original_output.dtype
        ):
            return None
        if any(not bool(c) for c in cast_down.checked_requires(cast_down_inputs)):
            return None
        cast_down.bind_input_like(cast_down_inputs)
        cast_down.bind_output_like(cast_down_outputs)
        cast_down_inst = rewritten.add_inst(InstExpr(cast_down, (old_output,)))
        for old_user in old_users:
            user = rewritten.find_inst_by_id(old_user.identifier)
            if user is None:
                return None
            for arg_idx, arg in enumerate(list(user.iexpr.args)):
                if arg == old_output:
                    user.iexpr.args = list(user.iexpr.args)
                    rewritten.replace_arg(user, arg_idx, cast_down_inst.retval(), type_check=False)
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
                if pattern.id == 'pt-inductor-int-clip-minmax':
                    matched_inst = working.insts[inst_idx]
                    has_abs = _has_abs_input_ancestor(working, matched_inst)
                    has_validated_direct_clip = _has_direct_int16_neg_input_clip(
                        working, matched_inst
                    )
                    if not has_abs and not has_validated_direct_clip:
                        continue
                candidate_rewrites = []
                if pattern.id == 'pt-inductor-cpu-byte-argreduce-tie':
                    widened = _rewrite_byte_argreduce_with_int64(working, inst_idx)
                    if widened is not None:
                        candidate_rewrites.append(widened)
                elif pattern.id in (
                    'pt-inductor-int16-abs-where-reduce-max-true-arm',
                    'pt-inductor-int16-abs-where-squeeze-reduce-min-false-arm',
                ):
                    abs_inst_id = matched_nodes[0].get('inst_id')
                    if abs_inst_id is not None:
                        widened_reduce = _rewrite_where_abs_before_reduce(
                            working, inst_idx, abs_inst_id
                        )
                        if widened_reduce is not None:
                            candidate_rewrites.append(widened_reduce)
                elif pattern.id in (
                    'pt-inductor-int8-abs-reduce-minmax',
                    'pt-inductor-int8-abs-flatten-reduce-minmax',
                    'pt-inductor-int16-abs-reduce-minmax',
                    'pt-inductor-int16-abs-squeeze-reduce-minmax',
                    'pt-inductor-int16-abs-slice-reduce-minmax',
                    'pt-inductor-int16-abs-transpose-reduce-minmax',
                    'pt-inductor-int16-double-abs-reduce-minmax',
                    'pt-inductor-int16-abs-squeeze-squeeze-neg-reduce-min',
                    'pt-inductor-int16-abs-squeeze-expand-reduce-minmax',
                ):
                    min_axis_len = (
                        1
                        if pattern.id == 'pt-inductor-int16-double-abs-reduce-minmax'
                        else 2
                    )
                    widened_reduce = _rewrite_narrow_reduce_minmax_with_int64(
                        working, inst_idx, min_axis_len=min_axis_len
                    )
                    if widened_reduce is not None:
                        candidate_rewrites.append(widened_reduce)
                elif pattern.id == 'pt-inductor-int16-argminmax':
                    widened_argreduce = _rewrite_int16_argreduce_after_safe_producer(
                        working, inst_idx
                    )
                    if widened_argreduce is not None:
                        candidate_rewrites.append(widened_argreduce)
                elif pattern.id == 'pt-inductor-int-clip-minmax':
                    if op_name in ('Min', 'Max'):
                        widened_minmax = _rewrite_narrow_binary_minmax_with_int64(
                            working, inst_idx
                        )
                        if widened_minmax is not None:
                            candidate_rewrites.append(widened_minmax)
                    elif op_name == 'Clip':
                        widened_clip = _rewrite_abs_neg_squeeze_clip_with_int64(
                            working, inst_idx
                        )
                        if widened_clip is not None:
                            candidate_rewrites.append(widened_clip)
                if pattern.id != 'pt-inductor-int-clip-minmax':
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
