"""
Constraint types for pattern matching.

Mirrors AIFuzzer's Kotlin implementation (DimMatcher, DtypeMatcher, AttrMatcher, etc.)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union


# =============================================================================
# DimMatcher — 维度匹配器
# =============================================================================

class DimMatcher:
    def matches(self, dim_value: Optional[int]) -> bool:
        raise NotImplementedError


class DimExact(DimMatcher):
    def __init__(self, value: int):
        self.value = value
    def matches(self, dv):
        return dv is not None and dv == self.value
    def __repr__(self):
        return f"={self.value}"


class DimNe(DimMatcher):
    def __init__(self, value: int):
        self.value = value
    def matches(self, dv):
        return dv is not None and dv != self.value
    def __repr__(self):
        return f"!={self.value}"


class DimGt(DimMatcher):
    def __init__(self, value: int):
        self.value = value
    def matches(self, dv):
        return dv is not None and dv > self.value
    def __repr__(self):
        return f">{self.value}"


class DimGte(DimMatcher):
    def __init__(self, value: int):
        self.value = value
    def matches(self, dv):
        return dv is not None and dv >= self.value
    def __repr__(self):
        return f">={self.value}"


class DimLt(DimMatcher):
    def __init__(self, value: int):
        self.value = value
    def matches(self, dv):
        return dv is not None and dv < self.value
    def __repr__(self):
        return f"<{self.value}"


class DimLte(DimMatcher):
    def __init__(self, value: int):
        self.value = value
    def matches(self, dv):
        return dv is not None and dv <= self.value
    def __repr__(self):
        return f"<={self.value}"


class DimInList(DimMatcher):
    def __init__(self, values: List[int]):
        self.values = values
    def matches(self, dv):
        return dv is not None and dv in self.values
    def __repr__(self):
        return f"in{self.values}"


class DimMod(DimMatcher):
    def __init__(self, divisor: int, remainder: int = 0):
        self.divisor = divisor
        self.remainder = remainder
    def matches(self, dv):
        return dv is not None and dv % self.divisor == self.remainder
    def __repr__(self):
        return f"%{self.divisor}=={self.remainder}"


class DimPow2(DimMatcher):
    def __init__(self, want_pow2: bool):
        self.want_pow2 = want_pow2
    def matches(self, dv):
        if dv is None:
            return False
        if dv <= 0:
            return not self.want_pow2
        is_pow2 = (dv & (dv - 1)) == 0
        return is_pow2 == self.want_pow2
    def __repr__(self):
        return "pow2" if self.want_pow2 else "!pow2"


class DimAny(DimMatcher):
    def matches(self, dv):
        return True
    def __repr__(self):
        return "*"


class DimAnd(DimMatcher):
    def __init__(self, matchers: List[DimMatcher]):
        self.matchers = matchers
    def matches(self, dv):
        return all(m.matches(dv) for m in self.matchers)
    def __repr__(self):
        return f"({' & '.join(repr(m) for m in self.matchers)})"


def parse_dim_matcher(obj: Any) -> DimMatcher:
    """Parse a DimMatcher from JSON. Mirrors Kotlin PatternParser."""
    if isinstance(obj, (int, float)):
        return DimExact(int(obj))
    if isinstance(obj, dict):
        keys = {k for k in obj if k in ('$eq', '$ne', '$gt', '$gte', '$lt', '$lte',
                                         '$in', '$mod', '$any', '$pow2')}
        if not keys:
            return DimAny()
        if len(keys) == 1:
            k = next(iter(keys))
            return _parse_single_dim(k, obj[k])
        return DimAnd([_parse_single_dim(k, obj[k]) for k in keys])
    return DimAny()


def _parse_single_dim(key: str, val: Any) -> DimMatcher:
    if key == '$eq':
        return DimExact(int(val))
    if key == '$ne':
        return DimNe(int(val))
    if key == '$gt':
        return DimGt(int(val))
    if key == '$gte':
        return DimGte(int(val))
    if key == '$lt':
        return DimLt(int(val))
    if key == '$lte':
        return DimLte(int(val))
    if key == '$in':
        return DimInList([int(v) for v in val])
    if key == '$pow2':
        return DimPow2(bool(val))
    if key == '$mod':
        if isinstance(val, int):
            return DimMod(val)
        return DimMod(int(val['d']), int(val.get('r', 0)))
    return DimAny()


# =============================================================================
# DtypeMatcher — 数据类型匹配器
# =============================================================================

class DtypeMatcher:
    def matches(self, name: str, bits: int) -> bool:
        raise NotImplementedError


class DtypeExact(DtypeMatcher):
    def __init__(self, name: str):
        self.name = name
    def matches(self, name, bits):
        return self.name == name
    def __repr__(self):
        return f"dtype={self.name}"


class DtypeInList(DtypeMatcher):
    def __init__(self, names: List[str]):
        self.names = names
    def matches(self, name, bits):
        return name in self.names
    def __repr__(self):
        return f"dtype in{self.names}"


class DtypeAny(DtypeMatcher):
    def matches(self, name, bits):
        return True
    def __repr__(self):
        return "dtype=*"


def parse_dtype_matcher(obj: Any) -> DtypeMatcher:
    if obj is None:
        return DtypeAny()
    if isinstance(obj, str):
        return DtypeExact(obj)
    if isinstance(obj, dict):
        if '$in' in obj:
            return DtypeInList([str(v) for v in obj['$in']])
    return DtypeAny()


# =============================================================================
# AttrMatcher — 属性匹配器
# =============================================================================

class AttrMatcher:
    def matches(self, actual: Any) -> bool:
        raise NotImplementedError


class AttrExactInt(AttrMatcher):
    def __init__(self, value: int):
        self.value = value
    def matches(self, actual):
        return isinstance(actual, int) and actual == self.value


class AttrExactString(AttrMatcher):
    def __init__(self, value: str):
        self.value = value
    def matches(self, actual):
        return isinstance(actual, str) and actual == self.value


class AttrExactIntList(AttrMatcher):
    def __init__(self, values: List[int]):
        self.values = values
    def matches(self, actual):
        if not isinstance(actual, (list, tuple)):
            return False
        if len(actual) != len(self.values):
            return False
        return all(isinstance(a, int) and a == v for a, v in zip(actual, self.values))


class AttrInList(AttrMatcher):
    def __init__(self, values: List[str]):
        self.values = values
    def matches(self, actual):
        return isinstance(actual, str) and actual in self.values


class AttrAny(AttrMatcher):
    def matches(self, actual):
        return True


class AttrNotInt(AttrMatcher):
    def __init__(self, value: int):
        self.value = value
    def matches(self, actual):
        return isinstance(actual, int) and actual != self.value


class AttrNotIntList(AttrMatcher):
    def __init__(self, values: List[int]):
        self.values = values
    def matches(self, actual):
        if not isinstance(actual, (list, tuple)):
            return False
        if len(actual) != len(self.values):
            return False
        return any(a != v for a, v in zip(actual, self.values))


def parse_attr_matcher(obj: Any) -> AttrMatcher:
    if isinstance(obj, str):
        return AttrExactString(obj)
    if isinstance(obj, (int, float)):
        return AttrExactInt(int(obj))
    if isinstance(obj, list):
        if all(isinstance(v, (int, float)) for v in obj):
            return AttrExactIntList([int(v) for v in obj])
        return AttrAny()
    if isinstance(obj, dict):
        if '$in' in obj:
            return AttrInList([str(v) for v in obj['$in']])
        if '$eq' in obj:
            eq = obj['$eq']
            if isinstance(eq, int):
                return AttrExactInt(eq)
            if isinstance(eq, str):
                return AttrExactString(eq)
            if isinstance(eq, list):
                if all(isinstance(v, (int, float)) for v in eq):
                    return AttrExactIntList([int(v) for v in eq])
            return AttrAny()
        if '$ne' in obj:
            ne = obj['$ne']
            if isinstance(ne, int):
                return AttrNotInt(ne)
            if isinstance(ne, list):
                if all(isinstance(v, (int, float)) for v in ne):
                    return AttrNotIntList([int(v) for v in ne])
            return AttrAny()
    return AttrAny()


# =============================================================================
# ExpressionConstraint — 跨维度表达式约束
# =============================================================================

class ExpressionConstraint:
    """约束跨维度的关系，如 product(dim[0], dim[2]) >= 65536"""

    def __init__(self, dim_indices: List[int], op: str, allowed_values: Set[int],
                 divisors: Optional[List[int]] = None,
                 exclude_when: Optional[List['ExpressionConstraint']] = None):
        self.dim_indices = dim_indices
        self.op = op
        self.allowed_values = allowed_values
        self.divisors = divisors
        self.exclude_when = exclude_when

    def evaluate(self, dims: List[Optional[int]]) -> Optional[int]:
        values = []
        for i, idx in enumerate(self.dim_indices):
            if idx >= len(dims):
                return None
            raw = dims[idx]
            if raw is None:
                return None
            dv = self.divisors[i] if self.divisors and i < len(self.divisors) else 1
            values.append(raw // dv)
        if self.op == 'mul':
            result = 1
            for v in values:
                result *= v
            return result
        elif self.op == 'add':
            return sum(values)
        elif self.op == 'sub':
            return values[0] - values[1] if len(values) == 2 else None
        elif self.op == 'mod':
            return values[0] % values[1] if len(values) == 2 and values[1] != 0 else None
        return None

    def matches(self, dims: List[Optional[int]]) -> bool:
        result = self.evaluate(dims)
        if result is None or result not in self.allowed_values:
            return False
        if self.exclude_when:
            for ex in self.exclude_when:
                ex_result = ex.evaluate(dims)
                if ex_result is not None and ex_result in ex.allowed_values:
                    return False
        return True

    def __repr__(self):
        return f"EC({self.dim_indices} {self.op} in {self.allowed_values})"


def parse_expression_constraints(arr: Any) -> List[ExpressionConstraint]:
    if not arr:
        return []
    result = []
    for obj in arr:
        dim_indices = obj['dimIndices']
        op = obj['op']
        allowed_values = set(obj['allowedValues'])
        divisors = None
        if 'divisors' in obj:
            divisors = obj['divisors']
        elif 'divisor' in obj:
            d = obj['divisor']
            divisors = [d] * len(dim_indices)
        exclude_when = None
        if 'excludeWhen' in obj:
            exclude_when = []
            for ex in obj['excludeWhen']:
                ex_dims = ex['dimIndices']
                exclude_when.append(ExpressionConstraint(
                    dim_indices=ex_dims, op=ex['op'],
                    allowed_values=set(ex['allowedValues']),
                    divisors=ex.get('divisors') or ([ex.get('divisor', 1)] * len(ex_dims)),
                ))
        result.append(ExpressionConstraint(
            dim_indices=dim_indices, op=op, allowed_values=allowed_values,
            divisors=divisors, exclude_when=exclude_when,
        ))
    return result


# =============================================================================
# ValueRangeMatcher — 值域匹配器
# =============================================================================

class ValueRangeMatcher:
    def matches(self, range_val) -> bool:
        raise NotImplementedError


class ValueRangeAny(ValueRangeMatcher):
    def matches(self, range_val):
        return True


class ValueRangeContainsZero(ValueRangeMatcher):
    def matches(self, range_val):
        return True  # 简化: 始终匹配


class ValueRangeNonNegative(ValueRangeMatcher):
    def matches(self, range_val):
        return True


class ValueRangeNonPositive(ValueRangeMatcher):
    def matches(self, range_val):
        return True


class ValueRangeEq(ValueRangeMatcher):
    def __init__(self, value: float):
        self.value = value
    def matches(self, range_val):
        return True


class ValueRangeLt(ValueRangeMatcher):
    def __init__(self, value: float):
        self.value = value
    def matches(self, range_val):
        return True


class ValueRangeGt(ValueRangeMatcher):
    def __init__(self, value: float):
        self.value = value
    def matches(self, range_val):
        return True


def parse_value_range_matcher(obj: Any) -> ValueRangeMatcher:
    if obj is None:
        return ValueRangeAny()
    if isinstance(obj, str):
        return {
            '$contains_zero': ValueRangeContainsZero(),
            '$non_negative': ValueRangeNonNegative(),
            '$non_positive': ValueRangeNonPositive(),
        }.get(obj, ValueRangeAny())
    if isinstance(obj, dict):
        if '$contains_zero' in obj:
            return ValueRangeContainsZero()
        if '$non_negative' in obj:
            return ValueRangeNonNegative()
        if '$non_positive' in obj:
            return ValueRangeNonPositive()
        if '$eq' in obj:
            return ValueRangeEq(float(obj['$eq']))
        if '$lt' in obj:
            return ValueRangeLt(float(obj['$lt']))
        if '$gt' in obj:
            return ValueRangeGt(float(obj['$gt']))
    return ValueRangeAny()