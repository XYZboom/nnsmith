from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from nnsmith.abstract.op import Constant, Input, Placeholder
from nnsmith.gir import GraphIR, InstIR
from nnsmith.materialize import BugReport, Model, Oracle, Stage, Symptom, TestCase


PropertyChecker = Callable[[GraphIR], bool]


@dataclass
class ReductionStep:
    stage: str
    before: int
    after: int
    detail: str


@dataclass
class ReductionResult:
    original_inst_count: int
    reduced_inst_count: int
    original_compute_count: int
    reduced_compute_count: int
    property_preserved: bool
    reduced_gir: GraphIR
    steps: List[ReductionStep] = field(default_factory=list)
    checks: int = 0
    cache_hits: int = 0

    @property
    def reduction_ratio(self) -> float:
        if self.original_compute_count == 0:
            return 0.0
        return 1.0 - self.reduced_compute_count / self.original_compute_count


class DDMin:
    """Classic delta debugging over a sequence of kept candidates."""

    def __init__(self, test: Callable[[List], bool]):
        self.test = test

    @staticmethod
    def _partitions(items: Sequence, n: int) -> List[List]:
        size, rem = divmod(len(items), n)
        ret = []
        start = 0
        for idx in range(n):
            width = size + (1 if idx < rem else 0)
            if width:
                ret.append(list(items[start : start + width]))
            start += width
        return ret

    def execute(self, items: Sequence) -> List:
        current = list(items)
        if not current:
            return current
        n = 2
        while len(current) >= 2:
            subsets = self._partitions(current, min(n, len(current)))
            reduced = False
            for subset in subsets:
                subset_ids = {id(x) for x in subset}
                complement = [x for x in current if id(x) not in subset_ids]
                if complement and self.test(complement):
                    current = complement
                    n = max(n - 1, 2)
                    reduced = True
                    break
            if reduced:
                continue
            for subset in subsets:
                if self.test(subset):
                    current = subset
                    n = max(n - 1, 2)
                    reduced = True
                    break
            if reduced:
                continue
            if n >= len(current):
                break
            n = min(len(current), n * 2)
        return current


def _is_compute(inst: InstIR) -> bool:
    return not isinstance(inst.iexpr.op, (Input, Constant, Placeholder))


def _rebuild_users(gir: GraphIR) -> None:
    for inst in gir.insts:
        inst.users = [set() for _ in range(inst.n_output())]
    defined = {ret: inst for inst in gir.insts for ret in inst.retvals()}
    for inst in gir.insts:
        for arg in set(inst.iexpr.args):
            producer = defined.get(arg)
            if producer is None:
                raise ValueError(f"Undefined GraphIR value: {arg}")
            _, ret_idx = InstIR.var_inst_idx(arg)
            producer.users[ret_idx].add(inst)
    gir.wellform_repair()
    gir.assert_wellform()


def _ancestor_inst_ids(gir: GraphIR, outputs: Iterable[str]) -> Set[int]:
    producer = {ret: inst for inst in gir.insts for ret in inst.retvals()}
    kept: Set[int] = set()
    stack = list(outputs)
    while stack:
        value = stack.pop()
        inst = producer.get(value)
        if inst is None or inst.identifier in kept:
            continue
        kept.add(inst.identifier)
        stack.extend(inst.iexpr.args)
    return kept


def _slice_to_outputs(gir: GraphIR, outputs: Sequence[str]) -> Optional[GraphIR]:
    if not outputs:
        return None
    available = {ret for inst in gir.insts for ret in inst.retvals()}
    if any(output not in available for output in outputs):
        return None
    ret = copy.deepcopy(gir)
    kept_ids = _ancestor_inst_ids(ret, outputs)
    ret.insts = [inst for inst in ret.insts if inst.identifier in kept_ids]
    kept_vars = {value for inst in ret.insts for value in inst.retvals()}
    ret.vars = {name: tensor for name, tensor in ret.vars.items() if name in kept_vars}
    try:
        _rebuild_users(ret)
    except Exception:
        return None
    return ret


def _compatible_replacement(gir: GraphIR, inst: InstIR, output: str) -> Optional[str]:
    out_type = gir.vars.get(output)
    for arg in inst.iexpr.args:
        in_type = gir.vars.get(arg)
        if out_type is None or in_type is None:
            continue
        if in_type.weak_compare(out_type) and out_type.weak_compare(in_type):
            return arg
    return None


def _apply_removed_nodes(
    gir: GraphIR,
    kept_compute_ids: Set[int],
    desired_outputs: Sequence[str],
) -> Optional[GraphIR]:
    ret = copy.deepcopy(gir)
    output_map = list(desired_outputs)
    removable = [
        inst
        for inst in ret.insts
        if _is_compute(inst) and inst.identifier not in kept_compute_ids
    ]
    for inst in reversed(removable):
        for output in inst.retvals():
            users = [user for user in ret.insts if output in user.iexpr.args]
            is_output = output in output_map
            if not users and not is_output:
                continue
            replacement = _compatible_replacement(ret, inst, output)
            if replacement is None:
                return None
            for user in users:
                user.iexpr.args = tuple(
                    replacement if arg == output else arg for arg in user.iexpr.args
                )
            output_map = [replacement if value == output else value for value in output_map]
        ret.insts.remove(inst)

    candidate = _slice_to_outputs(ret, list(dict.fromkeys(output_map)))
    return candidate


class GraphIRReducer:
    """AIFuzzer-style reduction for a single GraphIR.

    Stage 1 minimizes graph outputs and retains only their ancestor slices.
    Stage 2 runs DDMin over compute nodes. Removed nodes are dependency-repaired
    by wiring a shape/dtype-compatible input around the removed node.
    Stage 3 repeats single-node deletion to reach a 1-minimal fixed point.
    """

    def __init__(self, property_checker: PropertyChecker):
        self.property_checker = property_checker
        self._cache: Dict[str, bool] = {}
        self.checks = 0
        self.cache_hits = 0

    @staticmethod
    def _key(gir: GraphIR) -> str:
        return hashlib.sha256(pickle.dumps(gir, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()

    def _check(self, gir: Optional[GraphIR]) -> bool:
        if gir is None:
            return False
        key = self._key(gir)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.checks += 1
        try:
            result = bool(self.property_checker(gir))
        except Exception:
            result = False
        self._cache[key] = result
        return result

    def reduce(self, gir: GraphIR) -> ReductionResult:
        original = copy.deepcopy(gir)
        _rebuild_users(original)
        if not self._check(original):
            raise ValueError("The original GraphIR does not preserve the requested property")

        current = original
        steps: List[ReductionStep] = []

        leaves = current.leaf_var()
        if len(leaves) > 1:
            minimal_outputs = DDMin(
                lambda outputs: self._check(_slice_to_outputs(current, outputs))
            ).execute(leaves)
            sliced = _slice_to_outputs(current, minimal_outputs)
            if sliced is not None and self._check(sliced):
                before = current.n_compute_inst()
                current = sliced
                after = current.n_compute_inst()
                if after != before or len(minimal_outputs) != len(leaves):
                    steps.append(
                        ReductionStep(
                            "output-ddmin",
                            before,
                            after,
                            f"outputs {len(leaves)} -> {len(minimal_outputs)}",
                        )
                    )

        desired_outputs = current.leaf_var()
        compute_ids = [inst.identifier for inst in current.insts if _is_compute(inst)]
        if len(compute_ids) > 1:
            minimal_ids = DDMin(
                lambda kept: self._check(
                    _apply_removed_nodes(current, set(kept), desired_outputs)
                )
            ).execute(compute_ids)
            candidate = _apply_removed_nodes(current, set(minimal_ids), desired_outputs)
            if candidate is not None and self._check(candidate):
                before = current.n_compute_inst()
                current = candidate
                after = current.n_compute_inst()
                if after < before:
                    steps.append(
                        ReductionStep(
                            "node-ddmin",
                            before,
                            after,
                            f"kept compute IDs: {sorted(minimal_ids)}",
                        )
                    )

        changed = True
        while changed:
            changed = False
            desired_outputs = current.leaf_var()
            for inst in list(reversed(current.insts)):
                if not _is_compute(inst):
                    continue
                kept = {
                    other.identifier
                    for other in current.insts
                    if _is_compute(other) and other.identifier != inst.identifier
                }
                candidate = _apply_removed_nodes(current, kept, desired_outputs)
                if candidate is not None and self._check(candidate):
                    before = current.n_compute_inst()
                    current = candidate
                    steps.append(
                        ReductionStep(
                            "single-delete",
                            before,
                            current.n_compute_inst(),
                            f"removed {type(inst.iexpr.op).__name__}#{inst.identifier}",
                        )
                    )
                    changed = True
                    break

        if not self._check(current):
            current = original
            preserved = True
        else:
            preserved = True

        return ReductionResult(
            original_inst_count=gir.n_inst(),
            reduced_inst_count=current.n_inst(),
            original_compute_count=gir.n_compute_inst(),
            reduced_compute_count=current.n_compute_inst(),
            property_preserved=preserved,
            reduced_gir=current,
            steps=steps,
            checks=self.checks,
            cache_hits=self.cache_hits,
        )


_FAILING_VAR_RE = re.compile(r"(?:eager at|Failed at) (?P<var>v\d+_\d+)")
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _load_report_meta(report_dir: Path) -> Dict:
    with (report_dir / "meta.json").open() as stream:
        return json.load(stream)


def _normalize_error_line(line: str) -> str:
    line = _ANSI_RE.sub("", line).strip()
    line = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", line)
    line = re.sub(r"/[^\s:]+/torchinductor_[^\s:]+", "<torchinductor-temp>", line)
    return line


def derive_crash_signature(log: str) -> str:
    lines = [_normalize_error_line(line) for line in log.splitlines() if line.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if re.match(r"^[\w.]+(?:Error|Exception|Unsupported|AssertionError)(?::|$)", line):
            return line
    return lines[-1]


class ReportPropertyChecker:
    """Re-executes candidate GraphIRs and preserves the original bug class."""

    def __init__(
        self,
        report_dir: os.PathLike,
        backend_type: str = "pt2",
        target: str = "cpu",
        optmax: bool = True,
        equal_nan: bool = True,
        signature_regex: Optional[str] = None,
    ):
        self.report_dir = Path(report_dir)
        self.meta = _load_report_meta(self.report_dir)
        self.original_log = (self.report_dir / "err.log").read_text(errors="ignore")
        self.original_symptom = Symptom(self.meta["symptom"])
        self.original_stage = Stage(self.meta["stage"])
        self.failing_var = None
        failing_match = _FAILING_VAR_RE.search(self.original_log)
        if failing_match:
            self.failing_var = failing_match.group("var")
        self.crash_signature = derive_crash_signature(self.original_log)
        self.signature_regex = re.compile(signature_regex) if signature_regex else None
        self.equal_nan = equal_nan

        self.ModelType = Model.init("torch", target)
        self.factory = __import__(
            "nnsmith.backends", fromlist=["BackendFactory"]
        ).BackendFactory.init(
            backend_type, target=target, optmax=optmax, parse_name=True
        )
        self.original_oracle = Oracle.load(self.report_dir / Oracle.name())
        self.state_dict_path = (
            self.report_dir / f"{self.ModelType.name_prefix()}{self.ModelType.name_suffix()}"
        )
        self.last_bug: Optional[BugReport] = None

    def _build_testcase(self, gir: GraphIR) -> Optional[TestCase]:
        import torch

        model = self.ModelType.from_gir(gir)
        state = torch.load(self.state_dict_path, map_location="cpu", weights_only=True)
        model.torch_model.load_state_dict(state, strict=False)
        inputs = {}
        for name in model.input_like:
            if name not in self.original_oracle.input:
                return None
            inputs[name] = torch.from_numpy(np.asarray(self.original_oracle.input[name]))
        model.sat_inputs = inputs
        oracle = model.make_oracle()
        testcase = TestCase(model, oracle)
        testcase.ir = gir
        return testcase

    def _matches(self, bug: BugReport) -> bool:
        if bug.symptom != self.original_symptom or bug.stage != self.original_stage:
            return False
        log = bug.log or ""
        if self.signature_regex is not None:
            return self.signature_regex.search(log) is not None
        if self.original_symptom == Symptom.INCONSISTENCY:
            if self.failing_var is None:
                return True
            match = _FAILING_VAR_RE.search(log)
            return match is not None and match.group("var") == self.failing_var
        signature = derive_crash_signature(log)
        return bool(signature) and signature == self.crash_signature

    def __call__(self, gir: GraphIR) -> bool:
        import torch

        self.last_bug = None
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        testcase = self._build_testcase(gir)
        if testcase is None:
            return False
        with tempfile.TemporaryDirectory(prefix="nnsmith-reduce-") as cache_dir:
            previous_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
            try:
                result = self.factory.checked_compile_and_exec(testcase)
                if isinstance(result, BugReport):
                    bug = result
                else:
                    bug = self.factory.verify_results(
                        result, testcase, equal_nan=self.equal_nan
                    )
                if isinstance(bug, BugReport) and self._matches(bug):
                    self.last_bug = bug
                    return True
                return False
            finally:
                if previous_cache is None:
                    os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
                else:
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous_cache

    def dump_result(self, result: ReductionResult, output_dir: os.PathLike) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        reduced = result.reduced_gir
        if not self(reduced) or self.last_bug is None:
            raise RuntimeError("Final reduced GraphIR no longer reproduces the original bug class")
        testcase = self._build_testcase(reduced)
        assert testcase is not None
        testcase.dump(output)
        with (output / "gir.pkl").open("wb") as stream:
            pickle.dump(reduced, stream)
        (output / "err.log").write_text(self.last_bug.log or "")
        summary = {
            "original_inst_count": result.original_inst_count,
            "reduced_inst_count": result.reduced_inst_count,
            "original_compute_count": result.original_compute_count,
            "reduced_compute_count": result.reduced_compute_count,
            "reduction_ratio": result.reduction_ratio,
            "property_preserved": result.property_preserved,
            "checks": result.checks,
            "cache_hits": result.cache_hits,
            "top_level_class": (
                "Wrong-Code"
                if self.original_symptom == Symptom.INCONSISTENCY
                else "Crash"
            ),
            "symptom": self.original_symptom.value,
            "stage": self.original_stage.value,
            "failing_var": self.failing_var,
            "crash_signature": self.crash_signature,
            "steps": [asdict(step) for step in result.steps],
        }
        (output / "reduction_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        lines = [
            f"top_level_class: {summary['top_level_class']}",
            f"symptom/stage: {self.original_symptom.value}/{self.original_stage.value}",
            f"instructions: {result.original_inst_count} -> {result.reduced_inst_count}",
            f"compute nodes: {result.original_compute_count} -> {result.reduced_compute_count}",
            f"reduction ratio: {result.reduction_ratio:.1%}",
            f"property preserved: {result.property_preserved}",
            f"checks/cache hits: {result.checks}/{result.cache_hits}",
            "steps:",
        ]
        lines.extend(
            f"- {step.stage}: {step.before} -> {step.after}; {step.detail}"
            for step in result.steps
        )
        (output / "reduction_summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce NNSmith bug report using GraphIR DDMin")
    parser.add_argument("--report", type=str, required=True, help="Path to bug report directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory for reduced bug report")
    parser.add_argument("--backend", type=str, default="pt2", help="Backend type (default: pt2)")
    parser.add_argument("--target", type=str, default="cpu", help="Target device (default: cpu)")
    parser.add_argument("--optmax", action="store_true", default=True, help="Enable optmax (default: True)")
    parser.add_argument("--no-optmax", dest="optmax", action="store_false")
    parser.add_argument("--equal-nan", action="store_true", default=True, help="Equal NaN (default: True)")
    parser.add_argument("--no-equal-nan", dest="equal_nan", action="store_false")
    args = parser.parse_args()

    report_dir = Path(args.report)
    out_dir = Path(args.out)
    with (report_dir / "gir.pkl").open("rb") as f:
        gir = pickle.load(f)

    checker = ReportPropertyChecker(
        report_dir=report_dir,
        backend_type=args.backend,
        target=args.target,
        optmax=args.optmax,
        equal_nan=args.equal_nan,
    )
    reducer = GraphIRReducer(checker)
    res = reducer.reduce(gir)
    checker.dump_result(res, out_dir)
    print(
        f"Reduction finished: {res.original_compute_count} -> {res.reduced_compute_count} compute nodes (ratio: {res.reduction_ratio:.1%})"
    )
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()

