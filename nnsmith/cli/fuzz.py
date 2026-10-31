import json
import os
import random
import time
import traceback
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import FunctionType
from typing import Type

import hydra
from omegaconf import DictConfig

from nnsmith.abstract.extension import activate_ext
from nnsmith.backends.factory import BackendFactory
from nnsmith.cli.model_exec import verify_testcase
from nnsmith.error import InternalError
from nnsmith.filter import FILTERS
from nnsmith.graph_gen import model_gen
from nnsmith.logging import FUZZ_LOG
from nnsmith.macro import NNSMITH_BUG_PATTERN_TOKEN
from nnsmith.materialize import Model, TestCase
from nnsmith.narrow_spec import auto_opset
from nnsmith.util import (
    hijack_patch_requires,
    mkdir,
    op_filter,
    parse_timestr,
    set_seed,
)


def normalize_fuzz_seeds(seeds):
    """Normalize an optional explicit seed list from Hydra config."""
    if seeds is None:
        return None
    if isinstance(seeds, (int, str)):
        seeds = [seeds]
    normalized = [int(seed) for seed in seeds]
    if not normalized:
        raise ValueError("`fuzz.seeds` must contain at least one seed.")
    for seed in normalized:
        if not 0 <= seed < 2**32:
            raise ValueError(f"Invalid fuzz seed {seed}: expected 0 <= seed < 2**32.")
    return normalized


def load_fuzz_seeds(seed_file):
    """Load an ordered seed sequence from a JSON file."""
    path = Path(str(seed_file)).expanduser()
    if not path.is_file():
        raise ValueError(f"Fuzz seed file does not exist: {path}")

    with path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        seeds = data
    elif isinstance(data, dict) and isinstance(data.get("seeds"), list):
        seeds = data["seeds"]
    elif isinstance(data, dict) and isinstance(data.get("patterns"), list):
        seeds = []
        for pattern in data["patterns"]:
            pattern_seeds = pattern.get("seeds") if isinstance(pattern, dict) else None
            if not isinstance(pattern_seeds, list):
                raise ValueError(
                    f"Invalid pattern seed entry in {path}: expected a `seeds` list."
                )
            seeds.extend(pattern_seeds)
    else:
        raise ValueError(
            f"Unsupported fuzz seed file format in {path}: expected a JSON list, "
            "an object with `seeds`, or an object with grouped `patterns`."
        )

    return normalize_fuzz_seeds(seeds)


def resolve_fuzz_seeds(explicit_seeds, seed_file):
    """Resolve exactly one optional source of targeted fuzz seeds."""
    if explicit_seeds is not None and seed_file is not None:
        raise ValueError("Set only one of `fuzz.seeds` and `fuzz.seed_file`.")
    if seed_file is not None:
        return load_fuzz_seeds(seed_file)
    return normalize_fuzz_seeds(explicit_seeds)


class StatusCollect:
    def __init__(self, root, yes=True):
        self.root = Path(root)
        mkdir(self.root, yes=yes)
        self.n_testcases = 0
        self.n_bugs = 0
        self.n_fail_make_test = 0

    def get_next_bug_path(self):
        return self.root / f"bug-{NNSMITH_BUG_PATTERN_TOKEN}-{self.n_bugs}"


class FuzzingLoop:
    def __init__(
        self,
        cfg: DictConfig,
    ):
        self.cfg = cfg

        # FIXME(@ganler): well-form the fix or report to TF
        # Dirty fix for TFLite on CUDA-enabled systems.
        # If CUDA is not needed, disable them all.
        cmpwith = cfg["cmp"]["with"]
        if cfg["backend"]["type"] == "tflite" and (
            cmpwith is None or cmpwith["target"] != "cuda"
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        if (
            cfg["fuzz"]["crash_safe"]
            and cfg["backend"]["type"] == "xla"
            and cfg["backend"]["target"] == "cuda"
        ) or (
            cmpwith is not None
            and cmpwith["type"] == "xla"
            and cmpwith["target"] == "cuda"
        ):
            raise ValueError(
                "Please set `fuzz.crash_safe=false` for XLA on CUDA. "
                "Also see https://github.com/ise-uiuc/nnsmith/blob/main/doc/known-issues.md"
            )

        self.crash_safe = bool(cfg["fuzz"]["crash_safe"])
        self.test_timeout = cfg["fuzz"]["test_timeout"]
        if self.test_timeout is not None:
            if isinstance(self.test_timeout, str):
                self.test_timeout = parse_timestr(self.test_timeout)
            assert isinstance(
                self.test_timeout, int
            ), "`fuzz.test_timeout` must be an integer (or with `s` (default), `m`/`min`, or `h`/`hr`)."

        if not self.crash_safe and self.test_timeout is not None:
            # user enabled test_timeout but not crash_safe.
            FUZZ_LOG.warning(
                "`fuzz.crash_safe` is automatically enabled given `fuzz.test_timeout` is set."
            )

        self.filters = []
        # add filters.
        filter_types = (
            [cfg["filter"]["type"]]
            if isinstance(cfg["filter"]["type"], str)
            else cfg["filter"]["type"]
        )
        if filter_types:
            patches = (
                [cfg["filter"]["patch"]]
                if isinstance(cfg["filter"]["patch"], str)
                else cfg["filter"]["patch"]
            )
            for f in patches:
                assert os.path.isfile(
                    f
                ), "filter.patch must be a list of file locations."
                assert "@filter(" in open(f).read(), f"No filter found in the {f}."
                spec = spec_from_file_location("module.name", f)
                spec.loader.exec_module(module_from_spec(spec))
                FUZZ_LOG.info(f"Imported filter patch: {f}")
            for filter in filter_types:
                filter = str(filter)
                if filter not in FILTERS:
                    raise ValueError(
                        f"Filter {filter} not found. Available filters: {FILTERS.keys()}"
                    )
                fn_or_cls = FILTERS[filter]
                if isinstance(fn_or_cls, Type):
                    self.filters.append(fn_or_cls())
                elif isinstance(fn_or_cls, FunctionType):
                    self.filters.append(fn_or_cls)
                else:
                    raise InternalError(
                        f"Invalid filter type: {fn_or_cls} (aka {filter})"
                    )
                FUZZ_LOG.info(f"Filter enabled: {filter}")

        # Pattern dedup matcher.
        self.dedup_matcher = None
        self.n_dedup_rewrite = 0
        self.n_dedup_rewrite_fail = 0
        dedup_cfg = cfg.get("dedup")
        if dedup_cfg is not None and dedup_cfg.get("enabled", False):
            pattern_dir = dedup_cfg.get("pattern_dir")
            if pattern_dir:
                from nnsmith.dedup import DedupMatcher

                # Align with aiFuzzer: infer compiler and target if not specified
                compiler = dedup_cfg.get("compiler")
                if compiler is None:
                    b_type = str(cfg["backend"]["type"]).lower()
                    if b_type in ["torch", "torchjit", "pt2"]:
                        compiler = "pytorch"
                    elif "tvm" in b_type:
                        compiler = "tvm"
                    elif "onnx" in b_type:
                        compiler = "onnx"
                    else:
                        compiler = b_type

                target = dedup_cfg.get("target")
                if target is None:
                    target = cfg["backend"]["target"]

                self.dedup_matcher = DedupMatcher(
                    pattern_dir=pattern_dir,
                    compiler=compiler,
                    target=target,
                    backend=cfg["model"]["type"],
                )
                FUZZ_LOG.info(
                    f"Pattern dedup enabled: {len(self.dedup_matcher.patterns)} patterns "
                    f"(compiler={compiler}, target={target}) "
                    f"from {pattern_dir}"
                )

        self.status = StatusCollect(cfg["fuzz"]["root"])

        self.factory = BackendFactory.init(
            cfg["backend"]["type"],
            ad=cfg["ad"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
            parse_name=True,
        )

        model_cfg = self.cfg["model"]
        self.ModelType = Model.init(
            model_cfg["type"], backend_target=cfg["backend"]["target"]
        )
        self.ModelType.add_seed_setter()

        self.opset = op_filter(
            auto_opset(
                self.ModelType,
                self.factory,
                vulops=cfg["mgen"]["vulops"],
                grad=cfg["mgen"]["grad_check"],
            ),
            cfg["mgen"]["include"],
            cfg["mgen"]["exclude"],
        )

        hijack_patch_requires(cfg["mgen"]["patch_requires"])
        activate_ext(opset=self.opset, factory=self.factory)

        self.base_seed = cfg["fuzz"]["seed"]
        self.specific_seeds = resolve_fuzz_seeds(
            cfg["fuzz"].get("seeds", None),
            cfg["fuzz"].get("seed_file", None),
        )
        if self.specific_seeds is not None:
            seed = self.specific_seeds[0]
            FUZZ_LOG.info(
                f"Running {len(self.specific_seeds)} explicitly configured fuzz seed(s): "
                f"{self.specific_seeds}"
            )
        else:
            seed = self.base_seed if self.base_seed is not None else random.getrandbits(32)
        set_seed(seed)

        FUZZ_LOG.info(
            f"Test success info supressed -- only showing logs for failed tests"
        )

        # Time budget checking.
        self.timeout_s = self.cfg["fuzz"]["time"]
        if isinstance(self.timeout_s, str):
            self.timeout_s = parse_timestr(self.timeout_s)
        assert isinstance(
            self.timeout_s, int
        ), "`fuzz.time` must be an integer (with `s` (default), `m`/`min`, or `h`/`hr`)."

        self.max_testcases = self.cfg["fuzz"].get("max_testcases", None)
        if self.max_testcases is not None:
            self.max_testcases = int(self.max_testcases)
            FUZZ_LOG.info(f"Fuzzing will stop after {self.max_testcases} testcases.")

        self.save_test = cfg["fuzz"]["save_test"]
        if isinstance(self.save_test, str):  # path of root dir.
            FUZZ_LOG.info(f"Saving all intermediate testcases to {self.save_test}")
            mkdir(self.save_test)

    def make_concrete_gir(self, seed):
        mgen_cfg = self.cfg["mgen"]
        gen = model_gen(
            opset=self.opset,
            method=mgen_cfg["method"],
            seed=seed,
            max_elem_per_tensor=mgen_cfg["max_elem_per_tensor"],
            max_nodes=mgen_cfg["max_nodes"],
            timeout_ms=mgen_cfg["timeout_ms"],
            rank_choices=mgen_cfg["rank_choices"],
            dtype_choices=mgen_cfg["dtype_choices"],
        )
        return gen.make_concrete()

    def make_testcase(self, seed) -> TestCase:
        ir = self.make_concrete_gir(seed)

        if self.dedup_matcher is not None:
            matched_occurrences = self.dedup_matcher.match_occurrences(ir)
            if matched_occurrences:
                from nnsmith.dedup.rewrite import rewrite_gir
                from nnsmith.gir import InstIR

                pattern_counts = {}
                argreduce_input_producers = {}
                argreduce_input_context = {}
                reduce_input_context = {}
                clip_minmax_context = {}
                initial_occurrence_keys = set()
                for pattern, nodes in matched_occurrences:
                    pattern_counts[pattern.id] = pattern_counts.get(pattern.id, 0) + 1
                    initial_occurrence_keys.add(
                        (pattern.id, tuple(node.get('inst_id') for node in nodes))
                    )
                    if pattern.id == 'pt-inductor-int16-argminmax':
                        for node in nodes:
                            inst_id = node.get('inst_id')
                            inst = ir.find_inst_by_id(inst_id)
                            if (
                                inst is None
                                or type(inst.iexpr.op).__name__ not in ('ArgMin', 'ArgMax')
                                or len(inst.iexpr.args) != 1
                            ):
                                continue
                            producer_id, _ = InstIR.var_inst_idx(inst.iexpr.args[0])
                            producer = ir.find_inst_by_id(producer_id)
                            producer_name = (
                                type(producer.iexpr.op).__name__ if producer is not None else None
                            )
                            argreduce_input_producers[str(inst_id)] = producer_name
                            source_tensor = ir.vars.get(inst.iexpr.args[0])
                            reduce_dim = getattr(inst.iexpr.op, 'extra_attrs', {}).get('reduce_dim')
                            axis_len = None
                            if source_tensor is not None and reduce_dim is not None:
                                try:
                                    axis = int(reduce_dim)
                                    if axis < 0:
                                        axis += len(source_tensor.shape)
                                    if 0 <= axis < len(source_tensor.shape):
                                        axis_len = str(source_tensor.shape[axis])
                                except (TypeError, ValueError):
                                    pass
                            producer_chain = []
                            chain_inst = producer
                            for _ in range(6):
                                if chain_inst is None:
                                    break
                                chain_op = type(chain_inst.iexpr.op).__name__
                                output_tensor = (
                                    ir.vars.get(chain_inst.retval()) if chain_inst.n_output() else None
                                )
                                producer_chain.append({
                                    'op': chain_op,
                                    'dtype': str(output_tensor.dtype) if output_tensor is not None else None,
                                })
                                if chain_op in ('Input', 'Constant') or len(chain_inst.iexpr.args) != 1:
                                    break
                                parent_id, _ = InstIR.var_inst_idx(chain_inst.iexpr.args[0])
                                chain_inst = ir.find_inst_by_id(parent_id)
                            argreduce_input_context[str(inst_id)] = {
                                'producer': producer_name,
                                'producer_chain': producer_chain,
                                'dtype': str(source_tensor.dtype) if source_tensor is not None else None,
                                'shape': [str(dim) for dim in source_tensor.shape]
                                if source_tensor is not None else None,
                                'reduce_dim': str(reduce_dim),
                                'axis_len': axis_len,
                            }
                    elif pattern.id == 'pt-inductor-int-clip-minmax':
                        for node in nodes:
                            inst_id = node.get('inst_id')
                            inst = ir.find_inst_by_id(inst_id)
                            if inst is None:
                                continue
                            input_contexts = []
                            for input_var in inst.iexpr.args:
                                input_tensor = ir.vars.get(input_var)
                                producer_id, _ = InstIR.var_inst_idx(input_var)
                                chain_inst = ir.find_inst_by_id(producer_id)
                                producer_chain = []
                                for _ in range(4):
                                    if chain_inst is None:
                                        break
                                    chain_op = type(chain_inst.iexpr.op).__name__
                                    output_tensor = (
                                        ir.vars.get(chain_inst.retval())
                                        if chain_inst.n_output() else None
                                    )
                                    producer_chain.append({
                                        'op': chain_op,
                                        'dtype': str(output_tensor.dtype)
                                        if output_tensor is not None else None,
                                        'shape': [str(dim) for dim in output_tensor.shape]
                                        if output_tensor is not None else None,
                                    })
                                    if len(chain_inst.iexpr.args) != 1:
                                        break
                                    parent_id, _ = InstIR.var_inst_idx(chain_inst.iexpr.args[0])
                                    chain_inst = ir.find_inst_by_id(parent_id)
                                input_contexts.append({
                                    'var': str(input_var),
                                    'dtype': str(input_tensor.dtype)
                                    if input_tensor is not None else None,
                                    'shape': [str(dim) for dim in input_tensor.shape]
                                    if input_tensor is not None else None,
                                    'producer_chain': producer_chain,
                                })
                            output_tensor = (
                                ir.vars.get(inst.retval()) if inst.n_output() else None
                            )
                            clip_minmax_context[str(inst_id)] = {
                                'op': type(inst.iexpr.op).__name__,
                                'input_vars': [str(var) for var in inst.iexpr.args],
                                'input_vars_equal': len(inst.iexpr.args) == 2
                                and inst.iexpr.args[0] == inst.iexpr.args[1],
                                'inputs': input_contexts,
                                'output_dtype': str(output_tensor.dtype)
                                if output_tensor is not None else None,
                                'output_shape': [str(dim) for dim in output_tensor.shape]
                                if output_tensor is not None else None,
                                'attrs': {
                                    str(key): str(value)
                                    for key, value in getattr(inst.iexpr.op, 'extra_attrs', {}).items()
                                },
                            }
                    elif pattern.id in (
                        'pt-inductor-int16-abs-reduce-minmax',
                        'pt-inductor-int16-abs-squeeze-reduce-minmax',
                    ):
                        for node in nodes:
                            inst_id = node.get('inst_id')
                            inst = ir.find_inst_by_id(inst_id)
                            if (
                                inst is None
                                or type(inst.iexpr.op).__name__ not in ('ReduceMin', 'ReduceMax')
                                or len(inst.iexpr.args) != 1
                            ):
                                continue
                            source_var = inst.iexpr.args[0]
                            source_tensor = ir.vars.get(source_var)
                            producer_id, _ = InstIR.var_inst_idx(source_var)
                            producer = ir.find_inst_by_id(producer_id)
                            producer_chain = []
                            chain_inst = producer
                            for _ in range(6):
                                if chain_inst is None:
                                    break
                                chain_op = type(chain_inst.iexpr.op).__name__
                                output_tensor = (
                                    ir.vars.get(chain_inst.retval()) if chain_inst.n_output() else None
                                )
                                producer_chain.append({
                                    'op': chain_op,
                                    'dtype': str(output_tensor.dtype) if output_tensor is not None else None,
                                    'shape': [str(dim) for dim in output_tensor.shape]
                                    if output_tensor is not None else None,
                                })
                                if chain_op in ('Input', 'Constant') or len(chain_inst.iexpr.args) != 1:
                                    break
                                parent_id, _ = InstIR.var_inst_idx(chain_inst.iexpr.args[0])
                                chain_inst = ir.find_inst_by_id(parent_id)
                            reduce_dim = getattr(inst.iexpr.op, 'extra_attrs', {}).get('reduce_dim')
                            axis_len = None
                            if source_tensor is not None and reduce_dim is not None:
                                try:
                                    axis = int(reduce_dim)
                                    if axis < 0:
                                        axis += len(source_tensor.shape)
                                    if 0 <= axis < len(source_tensor.shape):
                                        axis_len = str(source_tensor.shape[axis])
                                except (TypeError, ValueError):
                                    pass
                            reduce_input_context[str(inst_id)] = {
                                'op': type(inst.iexpr.op).__name__,
                                'input_dtype': str(source_tensor.dtype) if source_tensor is not None else None,
                                'input_shape': [str(dim) for dim in source_tensor.shape]
                                if source_tensor is not None else None,
                                'reduce_dim': str(reduce_dim),
                                'axis_len': axis_len,
                                'producer_chain': producer_chain,
                            }
                original_op_by_id = {
                    inst.identifier: type(inst.iexpr.op).__name__ for inst in ir.insts
                }
                rewritten = rewrite_gir(ir, self.dedup_matcher)
                if rewritten is None:
                    rewritten_ops = {}
                    inserted_ops = {}
                else:
                    rewritten_op_by_id = {
                        inst.identifier: type(inst.iexpr.op).__name__ for inst in rewritten.insts
                    }
                    rewritten_ops = {
                        str(inst_id): f"{op_name}->{rewritten_op_by_id[inst_id]}"
                        for inst_id, op_name in original_op_by_id.items()
                        if inst_id in rewritten_op_by_id
                        and rewritten_op_by_id[inst_id] != op_name
                    }
                    inserted_ops = {
                        str(inst.identifier): type(inst.iexpr.op).__name__
                        for inst in rewritten.insts
                        if inst.identifier not in original_op_by_id
                    }
                if rewritten is None:
                    self.n_dedup_rewrite_fail += 1
                    FUZZ_LOG.warning(
                        f"Local pre-dedup: seed {seed} pattern_matches="
                        f"{json.dumps(pattern_counts, sort_keys=True)} rewrite=failed; "
                        f"argreduce_input_producers={json.dumps(argreduce_input_producers, sort_keys=True)}; "
                        f"argreduce_input_context={json.dumps(argreduce_input_context, sort_keys=True)}; "
                        f"reduce_input_context={json.dumps(reduce_input_context, sort_keys=True)}; "
                        f"clip_minmax_context={json.dumps(clip_minmax_context, sort_keys=True)}; "
                        "original GIR retained"
                    )
                else:
                    ir = rewritten
                    residual_occurrences = self.dedup_matcher.match_occurrences(ir)
                    residual_counts = {}
                    residual_occurrence_keys = set()
                    for pattern, nodes in residual_occurrences:
                        residual_counts[pattern.id] = residual_counts.get(pattern.id, 0) + 1
                        residual_occurrence_keys.add(
                            (pattern.id, tuple(node.get('inst_id') for node in nodes))
                        )
                    removed_occurrence_keys = [
                        {'pattern_id': pattern_id, 'inst_ids': list(inst_ids)}
                        for pattern_id, inst_ids in sorted(
                            initial_occurrence_keys - residual_occurrence_keys,
                            key=lambda item: (item[0], tuple(str(inst_id) for inst_id in item[1])),
                        )
                    ]
                    removed_occurrences = len(removed_occurrence_keys)
                    rewrite_status = 'partial' if residual_occurrences else 'applied'
                    self.n_dedup_rewrite += removed_occurrences
                    FUZZ_LOG.info(
                        f"Local pre-dedup: seed {seed} pattern_matches="
                        f"{json.dumps(pattern_counts, sort_keys=True)} rewrite={rewrite_status}; "
                        f"rewritten_occurrences={removed_occurrences}; "
                        f"rewritten_occurrence_keys={json.dumps(removed_occurrence_keys, sort_keys=True)}; "
                        f"remaining_pattern_matches={json.dumps(residual_counts, sort_keys=True)}; "
                        f"rewritten_ops={json.dumps(rewritten_ops, sort_keys=True)}; "
                        f"inserted_ops={json.dumps(inserted_ops, sort_keys=True)}; "
                        f"argreduce_input_producers={json.dumps(argreduce_input_producers, sort_keys=True)}; "
                        f"argreduce_input_context={json.dumps(argreduce_input_context, sort_keys=True)}; "
                        f"reduce_input_context={json.dumps(reduce_input_context, sort_keys=True)}; "
                        f"clip_minmax_context={json.dumps(clip_minmax_context, sort_keys=True)}; "
                        "same seed and original GIR, local rewrite only"
                    )

        model = self.ModelType.from_gir(ir)
        if self.cfg["debug"]["viz"]:
            model.attach_viz(ir)

        model.refine_weights()  # either random generated or gradient-based.
        model.set_grad_check(self.cfg["mgen"]["grad_check"])
        oracle = model.make_oracle()
        testcase = TestCase(model, oracle)
        testcase.ir = ir  # store GIR for dedup matching
        return testcase

    def validate_and_report(self, testcase: TestCase) -> bool:
        if not verify_testcase(
            self.cfg["cmp"],
            factory=self.factory,
            testcase=testcase,
            output_dir=self.status.get_next_bug_path(),
            filters=self.filters,
            crash_safe=self.crash_safe,
            timeout=self.test_timeout,
        ):
            self.status.n_bugs += 1
            return False
        return True

    def run(self):
        start_time = time.time()
        attempt_index = 0
        while time.time() - start_time < self.timeout_s:
            if self.specific_seeds is not None and attempt_index >= len(self.specific_seeds):
                break
            if self.max_testcases is not None and self.status.n_testcases >= self.max_testcases:
                break
            if self.specific_seeds is not None:
                seed = self.specific_seeds[attempt_index]
                # Make each targeted seed independently reproducible, including
                # weight/oracle generation after GraphIR construction.
                set_seed(seed)
            else:
                seed = random.getrandbits(32)
            attempt_index += 1
            if self.specific_seeds is not None:
                FUZZ_LOG.info(
                    f"Explicit seed start: index={attempt_index}/{len(self.specific_seeds)} seed={seed}"
                )
            else:
                FUZZ_LOG.debug(f"Making testcase with seed: {seed}")

            time_stat = {}

            gen_start = time.time()
            try:
                testcase = self.make_testcase(seed)
            except InternalError as e:
                raise e  # propagate internal errors
            except Exception:
                FUZZ_LOG.error(
                    f"`make_testcase` failed with seed {seed}. It can be NNSmith or Generator ({self.cfg['model']['type']}) bug."
                )
                FUZZ_LOG.error(traceback.format_exc())
                self.status.n_fail_make_test += 1
                if self.specific_seeds is not None:
                    FUZZ_LOG.info(
                        f"Explicit seed finished: index={attempt_index}/{len(self.specific_seeds)} seed={seed} status=make_testcase_failed"
                    )
                continue
            time_stat["gen"] = time.time() - gen_start

            eval_start = time.time()
            validation_succeeded = self.validate_and_report(testcase)
            if not validation_succeeded:
                FUZZ_LOG.warning(f"Failed model seed: {seed}")
            time_stat["eval"] = time.time() - eval_start

            if self.save_test:
                save_start = time.time()
                testcase_dir = os.path.join(
                    self.save_test, f"{time.time() - start_time:.3f}"
                )
                mkdir(testcase_dir)
                tmp, testcase.model.dotstring = testcase.model.dotstring, None
                testcase.dump(testcase_dir)
                testcase.model.dotstring = tmp
                # 保存 GIR 序列化（参考 aifuzzer BugCollector 保存 ir.jsonl）
                if hasattr(testcase, 'ir') and testcase.ir is not None:
                    try:
                        from nnsmith.gir_serializer import GirSerializer
                        with open(os.path.join(testcase_dir, "ir.jsonl"), "w") as f:
                            f.write(GirSerializer.to_jsonl(testcase.ir))
                    except Exception:
                        pass
                time_stat["save"] = time.time() - save_start

            FUZZ_LOG.info(
                f"Timing: { ''.join(f'{k}: {1000 * v:.1f}ms, ' for k, v in time_stat.items()) }"
            )
            if self.specific_seeds is not None:
                status = "completed" if validation_succeeded else "bug"
                FUZZ_LOG.info(
                    f"Explicit seed finished: index={attempt_index}/{len(self.specific_seeds)} seed={seed} status={status}"
                )
            self.status.n_testcases += 1
        FUZZ_LOG.info(f"Total {self.status.n_testcases} testcases generated.")
        FUZZ_LOG.info(f"Total {self.status.n_bugs} bugs found.")
        FUZZ_LOG.info(f"Total {self.status.n_fail_make_test} failed to make testcases.")
        if self.n_dedup_rewrite > 0 or self.n_dedup_rewrite_fail > 0:
            FUZZ_LOG.info(
                f"Total {self.n_dedup_rewrite} candidate nodes intercepted by pre-dedup."
            )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    FuzzingLoop(cfg).run()


if __name__ == "__main__":
    main()
