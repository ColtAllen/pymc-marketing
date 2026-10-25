#   Copyright 2022 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Benchmark the weighted-count sBG likelihood against the customer-level graph.

Three arms run on identical synthetic data:

``before``
    The ``pm.Censored`` graph from ``main`` before this change. Point ``--before-worktree``
    at a ``git worktree`` of that commit; it is put first on ``PYTHONPATH`` so the old
    ``pymc_marketing`` package shadows the installed one.
``after0``
    This branch with ``aggregate=False`` (single ``CensoredShiftedBetaGeometric`` code
    path, weights of one).
``after1``
    This branch with ``aggregate=True``.

Each (arm, size, scenario) combination runs in its own subprocess so peak RSS and
compile caches do not leak between runs. Results are JSON files; ``report`` turns them
into a markdown table with speedups and the ``after0`` vs ``before`` regression gate.

Examples
--------
.. code-block:: bash

    git worktree add /tmp/sbg-before $(git merge-base main HEAD)
    uv run python scripts/benchmarks/sbg_weighted_logp.py all \\
        --before-worktree /tmp/sbg-before --sizes 1e4 1e6 1e8 --out-dir sandbox/bench
    uv run python scripts/benchmarks/sbg_weighted_logp.py report --out-dir sandbox/bench

Not part of the test suite: it takes hours at 10^6 customers and needs ~10 GB of RAM at
10^8 (``after1`` only; the other arms are infeasible there, which is the point).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCENARIOS = {
    "S1": {"covariates": [], "label": "6 cohorts, no covariates"},
    "S2": {
        "covariates": ["channel", "tier"],
        "label": "6 cohorts, 2 categorical covariates",
    },
}
T_BY_COHORT = np.array([6, 7, 8, 9, 10, 12])
ALPHA_BY_COHORT = np.linspace(0.5, 1.2, len(T_BY_COHORT))
BETA_BY_COHORT = np.linspace(1.0, 3.0, len(T_BY_COHORT))
FIT_KWARGS = {"chains": 4, "tune": 500, "draws": 500}
N_LOGP_CALLS = 200


# --------------------------------------------------------------------------- data
def make_data(n: int, scenario: str, seed: int = 20261025):
    """Synthetic sBG customers with compact integer dtypes (10^8 rows fit in ~2 GB)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    cohort = rng.integers(0, len(T_BY_COHORT), size=n).astype("int16")
    T = T_BY_COHORT[cohort].astype("int16")
    alpha = ALPHA_BY_COHORT[cohort]
    beta = BETA_BY_COHORT[cohort]
    columns: dict[str, Any] = {}
    if SCENARIOS[scenario]["covariates"]:
        channel = rng.integers(0, 2, size=n).astype("int8")
        tier = rng.integers(0, 3, size=n).astype("int8")
        alpha = alpha * np.exp(-(0.3 * channel - 0.1 * tier))
        beta = beta * np.exp(-(-0.2 * channel + 0.15 * tier))
        columns["channel"], columns["tier"] = channel, tier
    p = rng.beta(alpha, beta)
    del alpha, beta
    recency = np.minimum(rng.geometric(p), T).astype("int16")
    del p
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1, dtype="int64"),
            "recency": recency,
            "T": T,
            # string labels (nutpie needs str/int64 coords) stored as int8 codes
            "cohort": pd.Categorical.from_codes(
                cohort.astype("int8"),
                categories=[f"c{i}" for i in range(len(T_BY_COHORT))],
            ),
            **columns,
        }
    )
    return frame


# ---------------------------------------------------------------------------- run
def _rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux
    return usage * scale / 1024**3


def _environment() -> dict[str, Any]:
    import arviz_stats
    import pymc
    import pytensor

    import pymc_marketing

    out = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "pymc": pymc.__version__,
        "pytensor": pytensor.__version__,
        "pytensor_linker": str(pytensor.config.linker),
        "arviz_stats": arviz_stats.__version__,
        "pymc_marketing_file": pymc_marketing.__file__,
    }
    try:
        import nutpie

        out["nutpie"] = nutpie.__version__
    except ImportError:  # pragma: no cover
        out["nutpie"] = None
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    """One arm x size x scenario in the current process."""
    import pymc as pm

    from pymc_marketing.clv import ShiftedBetaGeoModel

    n = int(float(args.n))
    result: dict[str, Any] = {
        "arm": args.arm,
        "n": n,
        "scenario": args.scenario,
        "sampler": args.sampler,
        "environment": _environment(),
    }
    covariates = SCENARIOS[args.scenario]["covariates"]

    t0 = time.perf_counter()
    data = make_data(n, args.scenario)
    result["data_time_s"] = time.perf_counter() - t0

    model_config: dict[str, Any] = {}
    if covariates:
        model_config["dropout_covariate_cols"] = covariates
    if args.arm == "after1":
        model_config["aggregate"] = True
    model = ShiftedBetaGeoModel(model_config=model_config)

    if args.arm == "after1":
        from pymc_marketing.clv._aggregation import aggregate_observations

        key_cols = ("cohort", "recency", "T", *covariates)
        t0 = time.perf_counter()
        agg = aggregate_observations(data, key_cols)
        result["aggregation_time_s"] = time.perf_counter() - t0
        result["n_unique"] = agg.n_unique
        if args.lazy_backend:
            import polars as pl

            t0 = time.perf_counter()
            aggregate_observations(pl.from_pandas(data).lazy(), key_cols)
            result["aggregation_time_polars_lazy_s"] = time.perf_counter() - t0
        del agg

    t0 = time.perf_counter()
    model.build_model(data=data)
    result["build_time_s"] = time.perf_counter() - t0

    # compiled logp + gradient at a fixed point
    t0 = time.perf_counter()
    from pymc.blocking import DictToArrayBijection

    fn = model.model.logp_dlogp_function()
    fn.set_extra_values({})
    point = model.model.initial_point()
    x = DictToArrayBijection.map(point)
    logp, grad = fn(x)
    result["compile_time_s"] = time.perf_counter() - t0
    result["logp_at_initial_point"] = float(logp)
    result["grad_norm_at_initial_point"] = float(np.linalg.norm(grad))
    for _ in range(10):
        fn(x)
    times = []
    for _ in range(N_LOGP_CALLS):
        t0 = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t0)
    result["logp_dlogp_median_ms"] = 1e3 * statistics.median(times)
    result["logp_dlogp_iqr_ms"] = 1e3 * (
        np.percentile(times, 75) - np.percentile(times, 25)
    )

    if args.fit:
        import arviz_stats as az

        wall = []
        ess_per_s = []
        for repeat in range(args.repeats):
            kwargs = dict(FIT_KWARGS, progressbar=False, random_seed=1000 + repeat)
            if args.sampler == "nutpie":
                kwargs["nuts_sampler"] = "nutpie"
            t0 = time.perf_counter()
            model.fit(data=data, method="mcmc", **kwargs)
            elapsed = time.perf_counter() - t0
            wall.append(elapsed)
            # cohort-level parameters present in every arm (per-customer alpha/beta
            # are unregistered under aggregation with covariates)
            ess_vars = (
                ["alpha_scale", "beta_scale"] if covariates else ["alpha", "beta"]
            )
            ess = az.ess(model.idata, var_names=ess_vars, method="bulk")
            ess_min = min(float(ess[v].min()) for v in ess.data_vars)
            ess_per_s.append(ess_min / elapsed)
        result["fit_wall_s"] = wall
        result["fit_wall_median_s"] = statistics.median(wall)
        result["ess_bulk_min_per_s"] = ess_per_s

        if args.loo:
            t0 = time.perf_counter()
            if args.arm == "after1":
                model.compute_log_likelihood()
                loo = model.loo()
            else:
                pm.compute_log_likelihood(
                    model.idata, model=model.model, progressbar=False
                )
                loo = az.loo(model.idata, var_name="dropout", pointwise=True)
            result["loo_time_s"] = time.perf_counter() - t0
            result["loo_elpd"] = float(loo.elpd)
            result["loo_n_data_points"] = int(loo.n_data_points)

    result["peak_rss_gb"] = _rss_gb()
    return result


# ---------------------------------------------------------------------------- all
def _result_path(out_dir: Path, arm: str, n: int, scenario: str, sampler: str) -> Path:
    return out_dir / f"{arm}_{scenario}_{n}_{sampler}.json"


def run_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(float(s)) for s in args.sizes]
    script = Path(__file__).resolve()
    for scenario in args.scenarios:
        for n in sizes:
            for arm in args.arms:
                if arm != "after1" and n > args.max_unaggregated:
                    continue
                samplers = ["nutpie"] + (["pymc"] if n <= args.pymc_sampler_max else [])
                for sampler in samplers:
                    path = _result_path(out_dir, arm, n, scenario, sampler)
                    if path.exists() and not args.force:
                        print(f"skip {path.name} (exists)")
                        continue
                    repeats = args.repeats if n <= args.repeat_max else 1
                    cmd = [
                        sys.executable,
                        str(script),
                        "run",
                        "--arm",
                        arm,
                        "--n",
                        str(n),
                        "--scenario",
                        scenario,
                        "--sampler",
                        sampler,
                        "--repeats",
                        str(repeats),
                        "--out",
                        str(path),
                    ]
                    if not args.no_fit:
                        cmd.append("--fit")
                    if n <= args.loo_max or arm == "after1":
                        cmd.append("--loo")
                    if arm == "after1" and n == max(sizes):
                        cmd.append("--lazy-backend")
                    env = dict(os.environ)
                    cwd = None
                    if arm == "before":
                        if not args.before_worktree:
                            raise SystemExit(
                                "--before-worktree is required for the before arm"
                            )
                        # The worktree goes first on sys.path (cwd and PYTHONPATH) so
                        # its pymc_marketing shadows the editable install; verified below.
                        cwd = str(Path(args.before_worktree).resolve())
                        env["PYTHONPATH"] = cwd + (
                            os.pathsep + env["PYTHONPATH"]
                            if env.get("PYTHONPATH")
                            else ""
                        )
                    print(">>>", " ".join(cmd[2:]), flush=True)
                    t0 = time.perf_counter()
                    proc = subprocess.run(cmd, env=env, cwd=cwd)  # noqa: S603
                    print(
                        f"<<< exit {proc.returncode} in {time.perf_counter() - t0:.0f}s",
                        flush=True,
                    )
                    if arm == "before" and path.exists():
                        used = json.loads(path.read_text())["environment"]
                        used = used["pymc_marketing_file"]
                        if cwd and not used.startswith(cwd):
                            raise SystemExit(
                                f"before arm imported {used}, not the worktree {cwd}"
                            )


# ------------------------------------------------------------------------- report
def report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    if not rows:
        raise SystemExit(f"no results in {out_dir}")

    def fmt(value, spec=".3g"):
        return "-" if value is None else format(value, spec)

    env = rows[0]["environment"]
    print("### Environment\n")
    for key in (
        "platform",
        "cpu",
        "python",
        "pymc",
        "pytensor",
        "pytensor_linker",
        "nutpie",
        "arviz_stats",
    ):
        print(f"- {key}: {env.get(key)}")
    print()

    print(
        "### Parity: logp at the initial point (identical across arms by construction)\n"
    )
    print("| scenario | n | arm | sampler | logp | grad norm |")
    print("|---|---|---|---|---|---|")
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["n"]), []).append(r)
        print(
            f"| {r['scenario']} | {r['n']:.0e} | {r['arm']} | {r['sampler']} | "
            f"{r['logp_at_initial_point']:.10f} | {r['grad_norm_at_initial_point']:.10f} |"
        )
    for key, group in groups.items():
        logps = {round(g["logp_at_initial_point"], 6) for g in group}
        if len(logps) > 1:
            print(f"\n**PARITY FAILURE** for {key}: {logps}")
    print()

    print("### Results\n")
    header = (
        "| scenario | n | arm | sampler | n_unique | aggregation s | build s | compile s | "
        "logp+dlogp ms (median) | fit wall s (median; all) | ESS_bulk/s (min var) | LOO s | peak RSS GB |"
    )
    print(header)
    print("|" + "---|" * 13)
    for r in rows:
        print(
            f"| {r['scenario']} | {r['n']:.0e} | {r['arm']} | {r['sampler']} | {fmt(r.get('n_unique'), 'd')} | "
            f"{fmt(r.get('aggregation_time_s'))}"
            + (
                f" (polars lazy {r['aggregation_time_polars_lazy_s']:.3g})"
                if "aggregation_time_polars_lazy_s" in r
                else ""
            )
            + f" | {r['build_time_s']:.3g} | {r['compile_time_s']:.3g} | {r['logp_dlogp_median_ms']:.3g} | "
            f"{fmt(r.get('fit_wall_median_s'))}"
            + (
                f" ({', '.join(f'{w:.0f}' for w in r['fit_wall_s'])})"
                if r.get("fit_wall_s")
                else ""
            )
            + f" | {fmt(statistics.median(r['ess_bulk_min_per_s']) if r.get('ess_bulk_min_per_s') else None)} | "
            f"{fmt(r.get('loo_time_s'))} | {r['peak_rss_gb']:.2f} |"
        )
    print()

    print(
        "### Speedups (after1 / before, nutpie) and regression gate (after0 vs before)\n"
    )
    print(
        "| scenario | n | logp+dlogp before/after1 | fit wall before/after1 "
        "| after0/before logp+dlogp | gate (<= 1.05) |"
    )
    print("|---|---|---|---|---|---|")
    by = {
        (r["scenario"], r["n"], r["arm"]): r for r in rows if r["sampler"] == "nutpie"
    }
    for (scenario, n), _ in sorted(groups.items()):
        before, after0, after1 = (
            by.get((scenario, n, a)) for a in ("before", "after0", "after1")
        )
        logp_speedup = (
            before["logp_dlogp_median_ms"] / after1["logp_dlogp_median_ms"]
            if before and after1
            else None
        )
        fit_speedup = (
            before["fit_wall_median_s"] / after1["fit_wall_median_s"]
            if before
            and after1
            and before.get("fit_wall_median_s")
            and after1.get("fit_wall_median_s")
            else None
        )
        ratio = (
            after0["logp_dlogp_median_ms"] / before["logp_dlogp_median_ms"]
            if before and after0
            else None
        )
        gate = "-" if ratio is None else ("PASS" if ratio <= 1.05 else "FAIL")
        print(
            f"| {scenario} | {n:.0e} | {fmt(logp_speedup)}x | {fmt(fit_speedup)}x | {fmt(ratio)} | {gate} |"
        )


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="one arm x size x scenario in this process")
    p_run.add_argument("--arm", choices=["before", "after0", "after1"], required=True)
    p_run.add_argument("--n", required=True, help="number of customers, e.g. 1e6")
    p_run.add_argument("--scenario", choices=list(SCENARIOS), default="S1")
    p_run.add_argument("--sampler", choices=["nutpie", "pymc"], default="nutpie")
    p_run.add_argument("--repeats", type=int, default=1)
    p_run.add_argument("--fit", action="store_true")
    p_run.add_argument("--loo", action="store_true")
    p_run.add_argument(
        "--lazy-backend",
        action="store_true",
        help="also time aggregation on a Polars LazyFrame",
    )
    p_run.add_argument("--out", type=Path)

    p_all = sub.add_parser("all", help="run every combination in subprocesses")
    p_all.add_argument("--out-dir", default="sandbox/bench")
    p_all.add_argument(
        "--before-worktree", help="git worktree of the merge-base with main"
    )
    p_all.add_argument("--sizes", nargs="+", default=["1e4", "1e6", "1e8"])
    p_all.add_argument(
        "--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS)
    )
    p_all.add_argument(
        "--arms",
        nargs="+",
        choices=["before", "after0", "after1"],
        default=["before", "after0", "after1"],
    )
    p_all.add_argument(
        "--repeats", type=int, default=3, help="fit repeats at sizes <= --repeat-max"
    )
    p_all.add_argument("--repeat-max", type=float, default=1e4)
    p_all.add_argument(
        "--max-unaggregated",
        type=float,
        default=1e6,
        help="largest n for before/after0",
    )
    p_all.add_argument(
        "--pymc-sampler-max",
        type=float,
        default=1e4,
        help="also run the PyMC NUTS sampler up to this n",
    )
    p_all.add_argument(
        "--loo-max",
        type=float,
        default=1e4,
        help="LOO on unaggregated arms up to this n",
    )
    p_all.add_argument("--no-fit", action="store_true")
    p_all.add_argument("--force", action="store_true")

    p_rep = sub.add_parser("report", help="markdown tables from the JSON results")
    p_rep.add_argument("--out-dir", default="sandbox/bench")

    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args)
        text = json.dumps(result, indent=2)
        if args.out:
            args.out.write_text(text)
        print(text)
    elif args.command == "all":
        run_all(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
