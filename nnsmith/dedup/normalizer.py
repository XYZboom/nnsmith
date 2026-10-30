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


def _shape_to_list(shape, model=None) -> List[Optional[int]]:
    """Convert dimensions without consuming the generator's Z3 context state.

    Numeric pattern constraints fail closed on unassigned symbolic dimensions.
    A raw model is translated to a private context before evaluating shapes.
    """
    if shape is None:
        return []

    symbolic_dims = [dim for dim in shape if hasattr(dim, 'as_ast')]
    if model is not None and any(dim.ctx == model.ctx for dim in symbolic_dims):
        import z3
        model = model.translate(z3.Context())

    result = []
    for dim in shape:
        value = dim
        if hasattr(dim, 'as_ast'):
            if model is None:
                result.append(None)
                continue
            if dim.ctx != model.ctx:
                dim = dim.translate(model.ctx)
            value = model.eval(dim, model_completion=False)
        if hasattr(value, 'as_long'):
            result.append(int(value.as_long()))
        else:
            try:
                result.append(int(value))
            except (TypeError, ValueError):
                result.append(None)
    return result


def _dtype_to_str(dtype) -> str:
    """Convert a dtype to string. nnsmith DType objects have __repr__ like 'float32'."""
    s = str(dtype)
    # strip nnsmith DType wrapper if present
    if '.' in s and s.rstrip('0123456789').endswith('.'):
        s = s.split('.')[-1]
    return s


def enrich_reduction_attrs(node: dict) -> None:
    """Attach concrete reduction-axis metadata when it is provable."""
    if node.get('op') not in ('ARGMIN', 'ARGMAX', 'REDUCE_MIN', 'REDUCE_MAX'):
        return
    inputs = node.get('inputs', [])
    shape = inputs[0].get('shape', []) if inputs else []
    axis = node.get('attrs', {}).get('axis')
    try:
        axis = int(axis)
        rank = len(shape)
        if axis < 0:
            axis += rank
        axis_len = int(shape[axis])
    except (IndexError, TypeError, ValueError):
        return
    node['attrs']['axis_len'] = axis_len


def normalize_graph(ir, backend: str = 'torch', shape_model: Any = None) -> List[dict]:
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
                    'var_id': str(arg_name),
                    'shape': _shape_to_list(t.shape, model=shape_model),
                    'dtype': _dtype_to_str(t.dtype),
                })
            else:
                input_info.append({
                    'var_id': str(arg_name), 'shape': [], 'dtype': 'unknown'
                })

        # Output shapes/dtypes
        output_info = []
        for i in range(inst.n_output()):
            vname = inst.retval(i)
            if vname in vars_map:
                t = vars_map[vname]
                output_info.append({
                    'var_id': str(vname),
                    'shape': _shape_to_list(t.shape, model=shape_model),
                    'dtype': _dtype_to_str(t.dtype),
                })
            else:
                output_info.append({
                    'var_id': str(vname), 'shape': [], 'dtype': 'unknown'
                })

        node = {
            'inst_id': getattr(inst, 'identifier', None),
            'op': pattern_op,
            'attrs': attrs,
            'inputs': input_info,
            'outputs': output_info,
            'raw_op': op_name,
        }
        enrich_reduction_attrs(node)
        nodes.append(node)

    return nodes


def normalize_node_dict(
    candidate_op: Any,
    input_vars: List[str],
    otensors: List[Any],
    backend: str = 'torch',
    output_vars: Optional[List[str]] = None,
    shape_model: Any = None,
) -> dict:
    """Normalize an uncommitted candidate node during graph generation."""
    op_name = get_op_name(candidate_op)
    pattern_op = NNSMITH_TO_PATTERN_OP.get(op_name, op_name.upper())

    attrs = {}
    if hasattr(candidate_op, 'extra_attrs') and candidate_op.extra_attrs:
        for k, v in candidate_op.extra_attrs.items():
            mapped_k = NNSMITH_ATTR_MAP.get(k, k)
            attrs[mapped_k] = v
    if hasattr(candidate_op, '__dict__'):
        for k in ['kernel_h_size', 'kernel_w_size', 'stride', 'padding',
                  'dilation_h', 'dilation_w', 'in_channels', 'out_channels',
                  'kernel_size', 'stride_h', 'stride_w', 'padding_h', 'padding_w']:
            if k in candidate_op.__dict__:
                mapped_k = NNSMITH_ATTR_MAP.get(k, k)
                attrs[mapped_k] = candidate_op.__dict__[k]

    input_info = []
    for var in input_vars:
        input_info.append({
            'var_id': str(var),
            'shape': [],
            'dtype': 'unknown',
        })

    output_info = []
    for idx, ot in enumerate(otensors):
        shape = getattr(ot, 'shape', [])
        dtype = getattr(ot, 'dtype', 'unknown')
        ref = {
            'shape': _shape_to_list(shape, model=shape_model),
            'dtype': _dtype_to_str(dtype),
        }
        if output_vars is not None and idx < len(output_vars):
            ref['var_id'] = str(output_vars[idx])
        output_info.append(ref)

    return {
        'op': pattern_op,
        'attrs': attrs,
        'inputs': input_info,
        'outputs': output_info,
        'raw_op': op_name,
    }