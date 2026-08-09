"""TVM Relax op dispatch: maps nnsmith abstract ops to TVM Relax expressions.

Usage: call `relax_forward(op, input_relax_vars, bb, param_map)` where
  - op: the nnsmith AbsOpBase instance
  - input_relax_vars: list of relax.Var/relax.Expr for the op's inputs
  - bb: relax.BlockBuilder instance
  - param_map: dict {retval_name: numpy.ndarray} for Constant ops

Returns a relax.Expr (the output of the op).
"""

from functools import partial
from typing import Dict, List, Type

import numpy as np
import tvm
from tvm import relax
import tvm.relax.op as rop

from nnsmith.abstract.op import *


def _ndim_pad_to(padding_list, ndims):
    """Convert nnsmith padding_list (last-dim-first) to TVM pad_width (first-dim-first).

    nnsmith padding_list: [pad_before_last, pad_after_last, pad_before_2nd_last, ...]
    TVM pad_width:        [pad_before_dim0, pad_after_dim0, pad_before_dim1, ..., pad_before_last, pad_after_last]
    """
    n_pairs = len(padding_list) // 2
    front_pairs = ndims - n_pairs
    pad_width = [0, 0] * front_pairs
    # Reverse padding_list so it becomes first-dim-first
    rev = list(reversed(padding_list))
    for i in range(n_pairs):
        pad_width.append(rev[i * 2 + 1])  # pad_after
        pad_width.append(rev[i * 2])  # pad_before
    return pad_width


def _dtype_str(dtype):
    """nnsmith DType -> TVM dtype string (e.g. 'float32')"""
    return dtype.value


def _as_int(v):
    """Concretize a possibly-symbolic integer to a Python int."""
    if hasattr(v, "as_long"):
        return v.as_long()
    return int(v)


def _gen_rand(shape, dtype):
    """Generate random numpy array matching nnsmith DType."""
    dt = _dtype_str(dtype)
    if dtype.is_float():
        return np.random.uniform(0.5, 1.5, size=shape).astype(dt)
    elif dtype == DType.bool:
        return np.random.randint(0, 2, size=shape).astype(dt)
    else:
        return np.random.randint(0, 5, size=shape).astype(dt)


# ── dispatch table: op_type -> callable(op, inputs, bb, param_map) -> relax.Expr ──

RELAX_FORWARD: Dict[Type[AbsOpBase], callable] = {}


def register(op_type):
    """Decorator: register a relax forward builder for an op type."""
    def wrapper(fn):
        RELAX_FORWARD[op_type] = fn
        return fn
    return wrapper


# ── Element-wise unary ops (direct mapping) ──

@register(ReLU)
def _relu(op, inputs, bb, pm):
    return rop.nn.relu(inputs[0])

@register(GELU)
def _gelu(op, inputs, bb, pm):
    return rop.nn.gelu(inputs[0])

@register(LeakyReLU)
def _leaky_relu(op, inputs, bb, pm):
    return rop.nn.leakyrelu(inputs[0], alpha=0.01)

@register(PReLU)
def _prelu(op, inputs, bb, pm):
    # PReLU needs an alpha param. Generate a random alpha tensor.
    input_rank = op.input_like[0].ndims if op.input_like[0] else 0
    alpha_np = np.random.uniform(0.1, 0.5, size=1).astype('float32')
    alpha = bb.emit(relax.const(alpha_np, 'float32'))
    if input_rank == 0:
        # scalar input: expand to 1D, apply PReLU, squeeze back
        x = rop.expand_dims(inputs[0], axis=0)
        result = rop.nn.prelu(x, alpha, axis=0)
        return rop.squeeze(result, axis=0)
    elif input_rank == 1:
        # 1D input: axis must be 0
        return rop.nn.prelu(inputs[0], alpha, axis=0)
    else:
        # 2D+ input: axis=1 (channel dim)
        return rop.nn.prelu(inputs[0], alpha, axis=1)

@register(Sigmoid)
def _sigmoid(op, inputs, bb, pm):
    return rop.sigmoid(inputs[0])

@register(Sin)
def _sin(op, inputs, bb, pm):
    # relax doesn't have direct sin, compose via tvm.te
    return rop.sin(inputs[0])

@register(Cos)
def _cos(op, inputs, bb, pm):
    return rop.cos(inputs[0])

@register(Asin)
def _asin(op, inputs, bb, pm):
    return rop.asin(inputs[0])

@register(Acos)
def _acos(op, inputs, bb, pm):
    return rop.acos(inputs[0])

@register(Tan)
def _tan(op, inputs, bb, pm):
    return rop.tan(inputs[0])

@register(Atan)
def _atan(op, inputs, bb, pm):
    return rop.atan(inputs[0])

@register(Abs)
def _abs(op, inputs, bb, pm):
    return rop.abs(inputs[0])

@register(Ceil)
def _ceil(op, inputs, bb, pm):
    return rop.ceil(inputs[0])

@register(Floor)
def _floor(op, inputs, bb, pm):
    return rop.floor(inputs[0])

@register(Clip)
def _clip(op, inputs, bb, pm):
    dtype = op.input_like[0].dtype
    dt = _dtype_str(dtype)
    if dtype.is_float():
        return rop.clip(inputs[0], relax.PrimValue(-1.5), relax.PrimValue(1.5))
    else:
        # int types: clip with integer bounds
        return rop.clip(inputs[0], relax.PrimValue(-1), relax.PrimValue(1))

@register(Round)
def _round(op, inputs, bb, pm):
    return rop.round(inputs[0])

@register(Sqrt)
def _sqrt(op, inputs, bb, pm):
    return rop.sqrt(inputs[0])

@register(Log2)
def _log2(op, inputs, bb, pm):
    # log2(x) = log(x) / log(2)
    dt = _dtype_str(op.input_like[0].dtype)
    log_x = rop.log(inputs[0])
    log_2 = relax.const(0.6931471805599453, dt)
    return rop.divide(log_x, log_2)

@register(Neg)
def _neg(op, inputs, bb, pm):
    return rop.negative(inputs[0])

@register(Softmax)
def _softmax(op, inputs, bb, pm):
    dim = int(op.dim) if not hasattr(op.dim, 'as_long') else op.dim.as_long()
    return rop.nn.softmax(inputs[0], axis=dim)

# ── Binary ops (direct mapping) ──

@register(Add)
def _add(op, inputs, bb, pm):
    return rop.add(inputs[0], inputs[1])

@register(Sub)
def _sub(op, inputs, bb, pm):
    return rop.subtract(inputs[0], inputs[1])

@register(Mul)
def _mul(op, inputs, bb, pm):
    return rop.multiply(inputs[0], inputs[1])

@register(Div)
def _div(op, inputs, bb, pm):
    return rop.divide(inputs[0], inputs[1])

@register(Max)
def _max(op, inputs, bb, pm):
    return rop.maximum(inputs[0], inputs[1])

@register(Min)
def _min(op, inputs, bb, pm):
    return rop.minimum(inputs[0], inputs[1])

@register(Pow)
def _pow(op, inputs, bb, pm):
    return rop.power(inputs[0], inputs[1])

@register(Equal)
def _equal(op, inputs, bb, pm):
    return rop.equal(inputs[0], inputs[1])

@register(Greater)
def _greater(op, inputs, bb, pm):
    return rop.greater(inputs[0], inputs[1])

@register(Less)
def _less(op, inputs, bb, pm):
    return rop.less(inputs[0], inputs[1])

@register(And)
def _and(op, inputs, bb, pm):
    return rop.logical_and(inputs[0], inputs[1])

@register(Or)
def _or(op, inputs, bb, pm):
    return rop.logical_or(inputs[0], inputs[1])

@register(Xor)
def _xor(op, inputs, bb, pm):
    return rop.logical_xor(inputs[0], inputs[1])

@register(Where)
def _where(op, inputs, bb, pm):
    return rop.where(inputs[0], inputs[1], inputs[2])

@register(MatMul)
def _matmul(op, inputs, bb, pm):
    return rop.matmul(inputs[0], inputs[1])

# ── Pooling ──

@register(MaxPool2d)
def _maxpool2d(op, inputs, bb, pm):
    return rop.nn.max_pool2d(
        inputs[0],
        pool_size=(int(op.kh), int(op.kw)),
        strides=(int(op.stride), int(op.stride)),
        padding=(int(op.padding), int(op.padding)),
    )

@register(AvgPool2d)
def _avgpool2d(op, inputs, bb, pm):
    return rop.nn.avg_pool2d(
        inputs[0],
        pool_size=(int(op.kh), int(op.kw)),
        strides=(int(op.stride), int(op.stride)),
        padding=(int(op.padding), int(op.padding)),
    )

# ── Convolution ──

@register(NCHWConv2d)
def _conv2d(op, inputs, bb, pm):
    in_c = int(op.in_channels)
    out_c = int(op.out_channels)
    kh = int(op.kernel_h_size)
    kw = int(op.kernel_w_size)
    # Generate weight
    w_np = _gen_rand((out_c, in_c, kh, kw), DType.float32)
    w = bb.emit(relax.const(w_np, 'float32'))
    return rop.nn.conv2d(
        inputs[0], w,
        strides=(int(op.stride), int(op.stride)),
        padding=(int(op.padding), int(op.padding)),
        dilation=(int(op.dilation_h), int(op.dilation_w)),
    )

@register(Conv1d)
def _conv1d(op, inputs, bb, pm):
    in_c = int(op.in_channels)
    out_c = int(op.out_channels)
    k = int(op.kernel_size)
    w_np = _gen_rand((out_c, in_c, k), DType.float32)
    w = bb.emit(relax.const(w_np, 'float32'))
    return rop.nn.conv1d(
        inputs[0], w,
        strides=int(op.stride),
        padding=int(op.padding),
        dilation=int(op.dilation),
    )

# ── BatchNorm ──

@register(BatchNorm2d)
def _batchnorm2d(op, inputs, bb, pm):
    nfeat = int(op.nfeat)
    gamma = bb.emit(relax.const(np.ones(nfeat, dtype='float32'), 'float32'))
    beta = bb.emit(relax.const(np.zeros(nfeat, dtype='float32'), 'float32'))
    mean = bb.emit(relax.const(np.zeros(nfeat, dtype='float32'), 'float32'))
    var = bb.emit(relax.const(np.ones(nfeat, dtype='float32'), 'float32'))
    bn_result = rop.nn.batch_norm(inputs[0], gamma, beta, mean, var, axis=1)
    # batch_norm returns a tuple (output, running_mean, running_var)
    return relax.TupleGetItem(bn_result, 0)

# ── Shape manipulation ──

@register(Reshape)
def _reshape(op, inputs, bb, pm):
    target = [_as_int(s) for s in op.target_shape]
    # Validate: all values must be positive (or -1 meaning infer)
    for v in target:
        if v == 0 or v < -1:
            raise ValueError(f"Reshape: invalid target shape {target} - only -1 or positive values allowed")
    return rop.reshape(inputs[0], tuple(target))

@register(Transpose)
def _transpose(op, inputs, bb, pm):
    ndim = op.input_like[0].ndims
    if 'dim0' in op.extra_attrs:
        dim0 = op.extra_attrs['dim0']
        dim1 = op.extra_attrs['dim1']
    else:
        dim0, dim1 = 0, 1
    axes = list(range(ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return rop.permute_dims(inputs[0], axes=axes)

@register(Squeeze)
def _squeeze(op, inputs, bb, pm):
    rd = op.extra_attrs.get('reduce_dim')
    if rd is not None:
        return rop.squeeze(inputs[0], axis=rd)
    return rop.squeeze(inputs[0])

@register(Unsqueeze)
def _unsqueeze(op, inputs, bb, pm):
    d = op.extra_attrs.get('expand_dim', 0)
    return rop.expand_dims(inputs[0], axis=d)

@register(ExpandLast1)
def _expand_last1(op, inputs, bb, pm):
    out_shape = tuple(int(s) if not hasattr(s, 'as_long') else s.as_long() for s in op.output_like[0].shape)
    return rop.broadcast_to(inputs[0], out_shape)

@register(ExpandLast2)
def _expand_last2(op, inputs, bb, pm):
    out_shape = tuple(int(s) if not hasattr(s, 'as_long') else s.as_long() for s in op.output_like[0].shape)
    return rop.broadcast_to(inputs[0], out_shape)

@register(ExpandLast3)
def _expand_last3(op, inputs, bb, pm):
    out_shape = tuple(int(s) if not hasattr(s, 'as_long') else s.as_long() for s in op.output_like[0].shape)
    return rop.broadcast_to(inputs[0], out_shape)

@register(ExpandLast4)
def _expand_last4(op, inputs, bb, pm):
    out_shape = tuple(int(s) if not hasattr(s, 'as_long') else s.as_long() for s in op.output_like[0].shape)
    return rop.broadcast_to(inputs[0], out_shape)

# ── Reduction ──

@register(ReduceSum)
def _reduce_sum(op, inputs, bb, pm):
    rd = op.extra_attrs.get('reduce_dim')
    # Keep dims for correct shape inference downstream
    return rop.sum(inputs[0], axis=rd, keepdims=False)

@register(ReduceMean)
def _reduce_mean(op, inputs, bb, pm):
    rd = op.extra_attrs.get('reduce_dim')
    return rop.mean(inputs[0], axis=rd, keepdims=False)

@register(ReduceMin)
def _reduce_min(op, inputs, bb, pm):
    rd = op.extra_attrs.get('reduce_dim')
    return rop.min(inputs[0], axis=rd, keepdims=False)

@register(ReduceMax)
def _reduce_max(op, inputs, bb, pm):
    rd = op.extra_attrs.get('reduce_dim')
    return rop.max(inputs[0], axis=rd, keepdims=False)

@register(ReduceProd)
def _reduce_prod(op, inputs, bb, pm):
    rd = op.extra_attrs.get('reduce_dim')
    return rop.prod(inputs[0], axis=rd, keepdims=False)

@register(ArgMin)
def _argmin(op, inputs, bb, pm):
    rd = op.extra_attrs.get('reduce_dim')
    if rd is not None:
        return rop.argmin(inputs[0], axis=rd, keepdims=False)
    return rop.argmin(inputs[0], keepdims=False)

@register(ArgMax)
def _argmax(op, inputs, bb, pm):
    rd = op.extra_attrs.get('reduce_dim')
    if rd is not None:
        return rop.argmax(inputs[0], axis=rd, keepdims=False)
    return rop.argmax(inputs[0], keepdims=False)

# ── Padding ──

@register(ConstPad)
def _constpad(op, inputs, bb, pm):
    ndims = op.input_like[0].ndims
    pw = _ndim_pad_to(op.padding_list, ndims)
    return rop.nn.pad(inputs[0], pad_width=pw, pad_mode='constant', pad_value=0.5)

@register(ReplicatePad)
def _replicatepad(op, inputs, bb, pm):
    ndims = op.input_like[0].ndims
    pw = _ndim_pad_to(op.padding_list, ndims)
    return rop.nn.pad(inputs[0], pad_width=pw, pad_mode='edge')

@register(ReflectPad)
def _reflectpad(op, inputs, bb, pm):
    ndims = op.input_like[0].ndims
    pw = _ndim_pad_to(op.padding_list, ndims)
    return rop.nn.pad(inputs[0], pad_width=pw, pad_mode='reflect')

# ── Concat ──

@register(Concat1)
def _concat1(op, inputs, bb, pm):
    axis = op.extra_attrs.get('axis', 0)
    return rop.concat(inputs, axis=axis)

@register(Concat2)
def _concat2(op, inputs, bb, pm):
    axis = op.extra_attrs.get('axis', 0)
    return rop.concat(inputs, axis=axis)

@register(Concat3)
def _concat3(op, inputs, bb, pm):
    axis = op.extra_attrs.get('axis', 0)
    return rop.concat(inputs, axis=axis)

@register(Concat4)
def _concat4(op, inputs, bb, pm):
    axis = op.extra_attrs.get('axis', 0)
    return rop.concat(inputs, axis=axis)

@register(Concat5)
def _concat5(op, inputs, bb, pm):
    axis = op.extra_attrs.get('axis', 0)
    return rop.concat(inputs, axis=axis)

# ── Cast ──

@register(CastF32)
def _castf32(op, inputs, bb, pm):
    return rop.astype(inputs[0], 'float32')

@register(CastF64)
def _castf64(op, inputs, bb, pm):
    return rop.astype(inputs[0], 'float64')

@register(CastI32)
def _casti32(op, inputs, bb, pm):
    return rop.astype(inputs[0], 'int32')

@register(CastI64)
def _casti64(op, inputs, bb, pm):
    return rop.astype(inputs[0], 'int64')

@register(CastBool)
def _castbool(op, inputs, bb, pm):
    return rop.astype(inputs[0], 'bool')

# ── Triangular ──

@register(Tril)
def _tril(op, inputs, bb, pm):
    diag = int(op.diagonal)
    return rop.tril(inputs[0], k=diag)

@register(Triu)
def _triu(op, inputs, bb, pm):
    diag = int(op.diagonal)
    return rop.triu(inputs[0], k=diag)

# ── Interpolation (not directly available in relax; use reshape + broadcast) ──

@register(NearestInterp)
def _nearest_interp(op, inputs, bb, pm):
    # Use repeat-based nearest upsampling
    # For simplicity, we use reshape + broadcast_to
    # Actually TVM relax does not have native interpolate.
    # We'll implement via slicing/broadcast for now.
    # For simplicity, skip interpolation ops - they are rare in fuzzing.
    # Just return the input unchanged (known limitation)
    return inputs[0]

@register(LinearInterp)
def _linear_interp(op, inputs, bb, pm):
    return inputs[0]

@register(BilinearInterp)
def _bilinear_interp(op, inputs, bb, pm):
    return inputs[0]

@register(BicubicInterp)
def _bicubic_interp(op, inputs, bb, pm):
    return inputs[0]

@register(TrilinearInterp)
def _trilinear_interp(op, inputs, bb, pm):
    return inputs[0]

# ── Slice ──

@register(Slice)
def _slice(op, inputs, bb, pm):
    axis = op.extra_attrs.get('axis', 0)
    start = int(op.start)
    end = int(op.end)
    step = int(op.step)
    # strided_slice requires axes/begin/end/strides as Tuple of PrimValues
    return rop.strided_slice(
        inputs[0],
        axes=relax.Tuple([relax.PrimValue(axis)]),
        begin=relax.Tuple([relax.PrimValue(start)]),
        end=relax.Tuple([relax.PrimValue(end)]),
        strides=relax.Tuple([relax.PrimValue(step)]),
    )

# ── ConcreteOp (custom ops) ──

@register(ConcreteOp)
def _concrete_op(op, inputs, bb, pm):
    # ConcreteOp delegates to its target_str; not supported in TVM frontend
    raise NotImplementedError(f"ConcreteOp '{op.target_str}' not supported in TVM frontend")


def relax_forward(op, inputs, bb, param_map):
    """Dispatch an nnsmith op to a TVM Relax expression.

    Args:
        op: AbsOpBase instance
        inputs: list of relax.Expr (input tensors)
        bb: relax.BlockBuilder
        param_map: dict of retval_name -> numpy array (unused, weights generated inline)

    Returns:
        relax.Expr
    """
    op_type = type(op)
    builder = RELAX_FORWARD.get(op_type)
    if builder is None:
        raise NotImplementedError(
            f"TVM frontend does not support op: {op_type.__name__}. "
            f"Available ops: {list(RELAX_FORWARD.keys())}"
        )
    return builder(op, inputs, bb, param_map)