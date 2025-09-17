#!/usr/bin/env python
# scripts/aggregate_svar_metrics.py
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import tyro

# --------------------------- CLI ---------------------------


@dataclass
class Args:
    results_root: str = "results/svar"
    out_dir: str | None = None
    group_hint: str | None = None  # substring to filter GROUP names
    level: float = 0.95  # coverage level (for table filenames only)
    percent: bool = False  # render coverage as % in LaTeX
    decimals: int = 2  # numeric display precision in LaTeX
    verbose: bool = True


# ------------------------ Utilities ------------------------

_METHODS_ORDER = ("npe", "pnpe", "rnpe", "prnpe")


def _discover_methods(root: Path) -> list[str]:
    cand = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    # keep standard order, then append any extras
    ordered = [m for m in _METHODS_ORDER if m in cand]
    extras = [m for m in cand if m not in ordered]
    return ordered + sorted(extras)


def _pick_group_for_method(method_dir: Path, group_hint: str | None) -> str:
    groups = [p.name for p in method_dir.iterdir() if p.is_dir()]
    if group_hint:
        groups = [g for g in groups if group_hint in g]
    if not groups:
        raise RuntimeError(f"No GROUP directories found under {method_dir}")
    # choose the group with the most seeds; tiebreak by lexicographic max
    counts: dict[str, int] = {}
    for g in groups:
        gpath = method_dir / g
        n = sum(1 for p in gpath.iterdir() if p.is_dir() and p.name.startswith("seed-"))
        counts[g] = n
    max_n = max(counts.values())
    best = sorted([g for g, n in counts.items() if n == max_n])[-1]
    return best


_ts_pat = re.compile(r"^\d{8}-\d{6}$")


def _latest_timestamp_dir(seed_dir: Path) -> Path | None:
    ts = [p for p in seed_dir.iterdir() if p.is_dir() and _ts_pat.match(p.name)]
    if not ts:
        return None
    # YYYYMMDD-HHMMSS is lexicographically sortable
    return sorted(ts, key=lambda p: p.name)[-1]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return cast(dict[str, Any], json.load(f))


def _get_first(d: dict[str, Any], *candidates: str, default=None):  # type: ignore
    for k in candidates:
        if k in d:
            return d[k]
    return default


def _as_float_array(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim == 0:
        return np.asarray([float(arr)])
    return arr.astype(float)


def _coerce_bool_array(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim == 0:
        arr = np.asarray([arr])
    # accept bools or 0/1
    if arr.dtype == bool:
        return arr.astype(float)
    return (arr != 0).astype(float)


def _find_theta_labels(run_dir: Path, theta_dim: int) -> list[str]:
    # Try metrics.json, then artifacts.json, else default p1..pD
    labels: list[str] | None = None
    for fname in ("metrics.json", "artifacts.json"):
        f = run_dir / fname
        if f.exists():
            try:
                d = _read_json(f)
                labels = _get_first(d, "theta_labels", default=None)
                if labels:
                    break
                # sometimes nested under "spec"
                spec = d.get("spec")
                if isinstance(spec, dict):
                    labels = spec.get("theta_labels")
                    if labels:
                        break
            except Exception:
                pass
    if not labels or len(labels) != theta_dim:
        return [f"p{i+1}" for i in range(theta_dim)]
    return list(labels)


# ---------------------- Metrics parsing ---------------------


def _parse_metrics(md: dict[str, Any]) -> dict[str, Any]:
    """Return dict with keys: hpdi_cov, central_cov, bias, mse, logprob, ppd_median."""
    out: dict[str, Any] = {}

    # coverage vectors (per-parameter)
    hpdi_cov = _coerce_bool_array(
        _get_first(
            md,
            "coverage_hpdi",
            "hpdi_coverage",
            "hpdi_contains",
            "hpdi_cover",
            "contains_hpdi",
        )
    ) or _coerce_bool_array(_get_first(md.get("coverage", {}), "hpdi", default=None))
    central_cov = _coerce_bool_array(
        _get_first(
            md,
            "coverage_central",
            "central_coverage",
            "central_contains",
            "contains_central",
        )
    ) or _coerce_bool_array(_get_first(md.get("coverage", {}), "central", default=None))

    # bias and mse vectors
    bias = _as_float_array(_get_first(md, "bias", "param_bias", "theta_bias"))
    mse = _as_float_array(_get_first(md, "mse", "param_mse", "theta_mse"))

    # scalars
    logprob = _get_first(
        md,
        "log_prob_at_target",
        "logprob_at_target",
        "synlik_at_target",
        "log_q_at_theta_target",
        "log_density_at_target",
    )
    logprob = None if logprob is None else float(logprob)

    # posterior predictive distance (median)
    candidates = []
    for k, v in md.items():
        if isinstance(v, (int, float)):
            continue
        if isinstance(v, dict):
            # look for nested summaries
            # e.g., {"ppd": {"l2": {"median": ...}}}
            ppd = v
            if "ppd" in k:
                candidates.append(v)
        # flat keys like "ppd_distance_median", "ppd_l2_median"
        if isinstance(v, (int, float)) and "ppd" in k and "median" in k:
            out["ppd_median"] = float(v)
    if "ppd_median" not in out:
        # common flat keys
        v = _get_first(md, "ppd_distance_median", "ppd_l2_median", "ppd_median")
        if v is not None:
            out["ppd_median"] = float(v)
        else:
            # nested search
            def _search_median(d: dict[str, Any]) -> float | None:
                # depth-limited search for a "median" under something that smells like distance
                for kk, vv in d.items():
                    if isinstance(vv, dict):
                        m = _search_median(vv)
                        if m is not None:
                            return m
                    else:
                        if "median" in kk and isinstance(vv, (int, float)):
                            return float(vv)
                return None

            for cand in candidates:
                m = _search_median(cand)
                if m is not None:
                    out["ppd_median"] = m
                    break

    out["hpdi_cov"] = hpdi_cov
    out["central_cov"] = central_cov
    out["bias"] = bias
    out["mse"] = mse
    out["logprob"] = logprob
    out["ppd_median"] = out.get("ppd_median", None)
    return out


# ------------------- Aggregation pipeline -------------------


def _aggregate(args: Args) -> None:
    root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir or (root / "_aggregates")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = _discover_methods(root)
    if args.verbose:
        print(f"[discover] methods: {methods}")

    # For each method, pick a GROUP, then for each seed pick latest timestamp/run.
    chosen: dict[str, dict[int, Path]] = {}
    groups_used: dict[str, str] = {}
    for m in methods:
        mdir = root / m
        if not mdir.is_dir():
            continue
        try:
            g = _pick_group_for_method(mdir, args.group_hint)
        except RuntimeError:
            if args.verbose:
                print(f"[skip] no GROUP directories under {mdir}")
            continue
        groups_used[m] = g
        seeds = {}
        gdir = mdir / g
        for sdir in sorted(
            p for p in gdir.iterdir() if p.is_dir() and p.name.startswith("seed-")
        ):
            mobj = re.match(r"seed-(\d+)", sdir.name)
            if not mobj:
                continue
            seed = int(mobj.group(1))
            latest = _latest_timestamp_dir(sdir)
            if latest and (latest / "metrics.json").exists():
                seeds[seed] = latest
        if not seeds:
            print(f"[warn] no valid runs for method={m} group={g}")
        chosen[m] = seeds

    # Establish theta_dim and labels from the first available run
    theta_dim: int | None = None
    labels: list[str] | None = None

    # Accumulators
    cov_hpdi: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    cov_central: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    bias: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    mse: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    logprob: dict[str, list[float]] = {m: [] for m in methods}
    ppd_med: dict[str, list[float]] = {m: [] for m in methods}

    for m in methods:
        for seed, run_dir in sorted(chosen[m].items()):
            md_path = run_dir / "metrics.json"
            try:
                md = _read_json(md_path)
            except Exception as e:
                print(f"[warn] failed to read {md_path}: {e}")
                continue
            parsed = _parse_metrics(md)

            # theta_dim
            if theta_dim is None:
                # infer from any vector
                for key in ("hpdi_cov", "central_cov", "bias", "mse"):
                    arr = parsed.get(key)
                    if isinstance(arr, np.ndarray):
                        theta_dim = int(arr.size)
                        break
                if theta_dim is None:
                    raise RuntimeError(
                        "Could not infer theta_dim from metrics.json files."
                    )
                if labels is None:
                    labels = _find_theta_labels(run_dir, theta_dim)

            # push arrays if shapes match
            def _push(dst: dict[str, list[np.ndarray]], key: str):
                arr = parsed.get(key)
                if isinstance(arr, np.ndarray):
                    if arr.size != theta_dim:
                        print(
                            f"[warn] {key} shape {arr.size} != theta_dim {theta_dim} at {md_path}"
                        )
                    else:
                        dst[m].append(arr.astype(float))

            _push(cov_hpdi, "hpdi_cov")
            _push(cov_central, "central_cov")
            _push(bias, "bias")
            _push(mse, "mse")

            # scalars
            if isinstance(parsed.get("logprob"), (int, float)):
                logprob[m].append(float(parsed["logprob"]))
            if isinstance(parsed.get("ppd_median"), (int, float)):
                ppd_med[m].append(float(parsed["ppd_median"]))

    assert theta_dim is not None
    assert labels is not None

    # Compute aggregates per method
    def _stack_and_mean(d: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for m in methods:
            arrs = d[m]
            if arrs:
                A = np.vstack(arrs)  # (R, D)
                out[m] = A.mean(axis=0)
            else:
                out[m] = np.full(theta_dim, np.nan)
        return out

    cov_hpdi_mean = _stack_and_mean(cov_hpdi)
    cov_central_mean = _stack_and_mean(cov_central)
    bias_mean = _stack_and_mean(bias)
    mse_mean = _stack_and_mean(mse)

    def _scalar_summary(vals: list[float]) -> tuple[float, float, int]:
        if not vals:
            return (math.nan, math.nan, 0)
        x = np.asarray(vals, dtype=float)
        return (
            float(x.mean()),
            float(x.std(ddof=1)) if x.size > 1 else 0.0,
            int(x.size),
        )

    logprob_stats = {m: _scalar_summary(logprob[m]) for m in methods}
    ppd_stats = {m: _scalar_summary(ppd_med[m]) for m in methods}

    # Build DataFrames
    def _wide_df(per_method: dict[str, np.ndarray]) -> pd.DataFrame:
        cols = []
        data = []
        for m in methods:
            cols.append(m)
        for i in range(theta_dim):
            row = [per_method[m][i] for m in methods]
            data.append(row)
        df = pd.DataFrame(data, index=labels, columns=cols)
        return df

    df_cov_hpdi = _wide_df(cov_hpdi_mean)
    df_cov_central = _wide_df(cov_central_mean)
    df_bias = _wide_df(bias_mean)
    df_mse = _wide_df(mse_mean)

    df_logprob = pd.DataFrame(
        {
            "mean": [logprob_stats[m][0] for m in methods],
            "sd": [logprob_stats[m][1] for m in methods],
            "n": [logprob_stats[m][2] for m in methods],
        },
        index=methods,
    )
    df_ppd = pd.DataFrame(
        {
            "median_mean": [ppd_stats[m][0] for m in methods],
            "median_sd": [ppd_stats[m][1] for m in methods],
            "n": [ppd_stats[m][2] for m in methods],
        },
        index=methods,
    )

    # Write CSV
    lvl_tag = str(int(round(args.level * 100)))
    (out_dir / f"coverage_hpdi_{lvl_tag}.csv").write_text(df_cov_hpdi.to_csv())
    (out_dir / f"coverage_central_{lvl_tag}.csv").write_text(df_cov_central.to_csv())
    (out_dir / "bias.csv").write_text(df_bias.to_csv())
    (out_dir / "mse.csv").write_text(df_mse.to_csv())
    (out_dir / "logprob.csv").write_text(df_logprob.to_csv())
    (out_dir / "ppd_median.csv").write_text(df_ppd.to_csv())

    # Write LaTeX
    def _fmt_float(x: float) -> str:
        if np.isnan(x):
            return "--"
        if args.percent:
            return f"{100.0 * x:.{args.decimals}f}"
        return f"{x:.{args.decimals}f}"

    def _write_latex(
        df: pd.DataFrame,
        fname: str,
        percent: bool = False,
        caption: str = "",
        label: str = "",
    ):
        df_out = df.copy()
        if percent:
            df_out = df_out.applymap(
                lambda v: np.nan if pd.isna(v) else 100.0 * float(v)
            )
        float_format = lambda v: f"{v:.{args.decimals}f}"
        latex = df_out.to_latex(
            escape=False,
            index=True,
            float_format=float_format,
            column_format="l" + "c" * len(df.columns),
            caption=caption or None,
            label=label or None,
        )
        (out_dir / fname).write_text(latex)

    _write_latex(
        df_cov_hpdi,
        f"coverage_hpdi_{lvl_tag}.tex",
        percent=args.percent,
        caption=f"SVAR marginal HPDI {int(args.level*100)}\\% coverage.",
        label="tab:svar_cov_hpdi",
    )
    _write_latex(
        df_cov_central,
        f"coverage_central_{lvl_tag}.tex",
        percent=args.percent,
        caption=f"SVAR marginal central {int(args.level*100)}\\% coverage.",
        label="tab:svar_cov_central",
    )
    _write_latex(
        df_bias,
        "bias.tex",
        percent=False,
        caption="SVAR marginal bias (per parameter).",
        label="tab:svar_bias",
    )
    _write_latex(
        df_mse,
        "mse.tex",
        percent=False,
        caption="SVAR marginal mean squared error (per parameter).",
        label="tab:svar_mse",
    )
    _write_latex(
        df_logprob,
        "logprob_at_target.tex",
        percent=False,
        caption="Log probability at (pseudo-)true parameter: mean$\\pm$sd over replicates.",
        label="tab:svar_logprob",
    )
    _write_latex(
        df_ppd,
        "ppd_distance_median.tex",
        percent=False,
        caption="Posterior predictive distance (median) summary: mean$\\pm$sd over replicates.",
        label="tab:svar_ppd",
    )

    # Summary manifest
    manifest = {
        "results_root": str(root),
        "out_dir": str(out_dir),
        "groups_used": groups_used,
        "replicates_per_method": {m: len(chosen[m]) for m in methods},
        "theta_dim": int(theta_dim),
        "param_labels": labels,
        "files": sorted([p.name for p in out_dir.iterdir() if p.is_file()]),
    }
    (out_dir / "summary.json").write_text(json.dumps(manifest, indent=2))
    if args.verbose:
        print(json.dumps(manifest, indent=2))


def main() -> None:
    args = tyro.cli(Args)
    _aggregate(args)


if __name__ == "__main__":
    main()
