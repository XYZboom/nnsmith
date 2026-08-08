"""
nnsmith GraphIR normalizer — converts nnsmith's GraphIR to a uniform node-list format
that the PatternMatcher can consume.

Each normalized node is a dict:
  { 'op': str, 'attrs': dict, 'inputs': [{'shape':[...], 'dtype':str}, ...],
    'outputs': [{'shape':[...], 'dtype':str}, ...], 'raw_op': str }
"""

from typing import Any, Dict, List, Optional

# Op name mapping: nnsmith op names → pattern op names
NNSMITH_TO_PATTERN_OP = {
    # Torch backend
    'NCHWConv2d': 'CONV2D',
    'AvgPool2d': 'AVG_POOL2D',
    'MaxPool2d': 'MAX_POOL2D',
    'BatchNorm2d': 'BATCH_NORM',
    'Softmax': 'SOFTMAX',
    'ReduceMean': 'REDUCE_MEAN',
    'TorchReduceSum': 'REDUCE_SUM',
    'ReduceMax': 'REDUCE_MAX',
    'ReduceMin': 'REDUCE_MIN',
    'ReduceSum': 'REDUCE_SUM',
    'Sigmoid': 'SIGMOID',
    'ReLU': 'RELU',
    'GELU': 'GELU',
    'LeakyReLU': 'LEAKY_RELU',
    'PReLU': 'PReLU',
    'Floor': 'FLOOR',
    'Ceil': 'CEIL',
    'Round': 'ROUND',
    'Abs': 'ABS',
    'Neg': 'NEG',
    'Add': 'ADD',
    'Sub': 'SUBTRACT',
    'Mul': 'MULTIPLY',
    'Div': 'DIVIDE',
    'Max': 'MAXIMUM',
    'Min': 'MINIMUM',
    'Where': 'WHERE',
    'Transpose': 'TRANSPOSE',
    'Squeeze': 'SQUEEZE',
    'Clip': 'CLIP',
    'Sin': 'SIN',
    'Cos': 'COS',
    'Tan': 'TAN',
    'Atan': 'ATAN',
    'Sqrt': 'SQRT',
    'Log2': 'LOG2',
    'Pow': 'POWER',
    'PTMatMul': 'MATMUL',
    'MatMul': 'MATMUL',
    'Flatten': 'FLATTEN',
    'Concat1': 'CONCAT',
    'Concat2': 'CONCAT',
    'Concat3': 'CONCAT',
    'Concat4': 'CONCAT',
    'Concat5': 'CONCAT',
    'Slice': 'SLICE',
    'Reshape': 'RESHAPE',
    'ExpandLast1': 'EXPAND',
    'ExpandLast2': 'EXPAND',
    'ExpandLast3': 'EXPAND',
    'ExpandLast4': 'EXPAND',
    'Expand': 'EXPAND',
    'ArgMax': 'ARGMAX',
    'ArgMin': 'ARGMIN',
    'Equal': 'EQUAL',
    'Greater': 'GREATER',
    'Less': 'LESS',
    'And': 'AND',
    'Or': 'OR',
    'Xor': 'XOR',
    'ConstPad': 'CONST_PAD',
    'ReflectPad': 'REFLECT_PAD',
    'ReplicatePad': 'REPLICATE_PAD',
    'CastBool': 'CAST',
    'CastF32': 'CAST',
    'CastF64': 'CAST',
    'CastI32': 'CAST',
    'CastI64': 'CAST',
    'LinearInterp': 'LINEAR_INTERP',
    'BilinearInterp': 'BILINEAR_INTERP',
    'TrilinearInterp': 'TRILINEAR_INTERP',
    'NearestInterp': 'NEAREST_INTERP',
    'BicubicInterp': 'BICUBIC_INTERP',
    # ONNX ops (from ONNX backend)
    'Conv': 'CONV2D',
    'MaxPool': 'MAX_POOL2D',
    'AveragePool': 'AVG_POOL2D',
    'BatchNormalization': 'BATCH_NORM',
    'Relu': 'RELU',
    'Gemm': 'MATMUL',
    'ReduceProd': 'CUMPROD',
    'Tanh': 'TANH',
    'Split': 'SPLIT',
    'Pad': 'PAD',
    'Resize': 'RESIZE',
    'Shape': 'SHAPE',
    'Cast': 'CAST',
    'Constant': 'CONSTANT',
    'ConstantOfShape': 'CONSTANT',
    'Log': 'LOG',
    'Exp': 'EXP',
    'Asin': 'ASIN',
    'Acos': 'ACOS',
    'Sinh': 'SINH',
    'Cosh': 'COSH',
    'Erf': 'ERF',
    'LeakyRelu': 'LEAKY_RELU',
    'PRelu': 'PRELU',
    'Not': 'NOT',
    'Unsqueeze': 'UNSQUEEZE',
    'Gather': 'GATHER',
    'GatherElements': 'GATHER',
    'ScatterElements': 'SCATTER',
    'Softplus': 'SOFTPLUS',
    'Softsign': 'SOFTSIGN',
    'HardSigmoid': 'HARD_SIGMOID',
    'Elu': 'ELU',
    'Selu': 'SELU',
    'Mish': 'MISH',
    'HardSwish': 'HARD_SWISH',
    'LayerNormalization': 'LAYER_NORM',
    'InstanceNormalization': 'INSTANCE_NORM',
    'LpNormalization': 'LP_NORM',
    'LRN': 'LRN',
    'Mean': 'MEAN',
    'Sum': 'SUM',
    'CumSum': 'CUMPROD',
    'MatMulInteger': 'MATMUL',
    'QLinearMatMul': 'MATMUL',
    'Compress': 'COMPRESS',
    'OneHot': 'ONE_HOT',
    'TopK': 'TOPK',
    'NonMaxSuppression': 'NMS',
    'RoiAlign': 'ROI_ALIGN',
    'GRU': 'GRU',
    'LSTM': 'LSTM',
    'RNN': 'RNN',
}

# Attr name mapping: nnsmith extra_attrs → pattern attrs
NNSMITH_ATTR_MAP = {
    'kernel_h_size': 'kernel_size',
    'kernel_w_size': 'kernel_size',
    'stride_h': 'stride',
    'stride_w': 'stride',
    'padding_h': 'padding',
    'padding_w': 'padding',
    'dilation_h': 'dilation',
    'dilation_w': 'dilation',
    'reduce_dim': 'axis',
    'dim': 'axis',
    'in_channels': 'in_channels',
    'out_channels': 'out_channels',
    'nfeat': 'num_features',
}


def get_op_name(op) -> str:
    """Get nnsmith op name string. `op.name` is a classmethod, so call it.
    Strips 'core.'/'torch.' prefix and normalizes case."""
    name_attr = getattr(op, 'name', None)
    if callable(name_attr):
        try:
            raw = name_attr()
        except Exception:
            raw = type(op).__name__
    else:
        raw = type(op).__name__
    raw = str(raw)
    for prefix in ('core.', 'torch.'):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix):]
            break
    return raw


def _shape_to_list(shape) -> List[int]:
    """Convert a shape (tuple/list/tvm shape) to a list of concrete ints.
    Returns [] if shape is symbolic (not integer)."""
    if shape is None:
        return []
    try:
        return [int(d) for d in shape]
    except (TypeError, ValueError):
        # Symbolic dims (z3) — return placeholder
        return [d for d in shape]


def _dtype_to_str(dtype) -> str:
    """Convert a dtype to string. nnsmith DType objects have __repr__ like 'float32'."""
    s = str(dtype)
    # strip nnsmith DType wrapper if present
    if '.' in s and s.rstrip('0123456789').endswith('.'):
        s = s.split('.')[-1]
    return s


def normalize_graph(ir, backend: str = 'torch') -> List[dict]:
    """
    Normalize nnsmith GraphIR to a list of dicts:
    { 'op': str, 'attrs': dict, 'inputs': [{'shape':[...], 'dtype':str}], 
      'outputs': [{'shape':[...], 'dtype':str}] }
    """
    nodes = []
    # ir.vars is a dict: var_name -> AbsTensor
    vars_map = dict(ir.vars)

    for inst in ir.insts:
        op_name = get_op_name(inst.iexpr.op)
        # Skip Input and Constant
        if op_name in ('Input', 'Constant'):
            continue

        pattern_op = NNSMITH_TO_PATTERN_OP.get(op_name, op_name.upper())

        # Collect attrs from extra_attrs
        attrs = {}
        op = inst.iexpr.op
        if hasattr(op, 'extra_attrs') and op.extra_attrs:
            for k, v in op.extra_attrs.items():
                mapped_k = NNSMITH_ATTR_MAP.get(k, k)
                attrs[mapped_k] = v
        # Also collect from __dict__ for pool/conv attrs
        if hasattr(op, '__dict__'):
            for k in ['kernel_h_size', 'kernel_w_size', 'stride', 'padding',
                      'dilation_h', 'dilation_w', 'in_channels', 'out_channels',
                      'kernel_size', 'stride_h', 'stride_w', 'padding_h', 'padding_w']:
                if k in op.__dict__:
                    mapped_k = NNSMITH_ATTR_MAP.get(k, k)
                    attrs[mapped_k] = op.__dict__[k]

        # Input shapes/dtypes
        input_info = []
        for arg_name in inst.iexpr.args:
            if arg_name in vars_map:
                t = vars_map[arg_name]
                input_info.append({
                    'shape': _shape_to_list(t.shape),
                    'dtype': _dtype_to_str(t.dtype),
                })
            else:
                input_info.append({'shape': [], 'dtype': 'unknown'})

        # Output shapes/dtypes
        output_info = []
        for i in range(inst.n_output()):
            vname = inst.retval(i)
            if vname in vars_map:
                t = vars_map[vname]
                output_info.append({
                    'shape': _shape_to_list(t.shape),
                    'dtype': _dtype_to_str(t.dtype),
                })
            else:
                output_info.append({'shape': [], 'dtype': 'unknown'})

        nodes.append({
            'op': pattern_op,
            'attrs': attrs,
            'inputs': input_info,
            'outputs': output_info,
            'raw_op': op_name,
        })

    return nodes