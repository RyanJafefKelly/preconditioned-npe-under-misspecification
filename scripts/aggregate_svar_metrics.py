#!/usr/bin/env python
# scripts/aggregate_svar_metrics.py
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
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
    debug_runs: int = 0  # print debug for first N runs per method

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


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict/list structures to dotted key paths."""
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(d, list):
        # Keep lists; often numeric vectors
        out[prefix] = d
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


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
        return [f"p{i + 1}" for i in range(theta_dim)]
    return list(labels)


def _first_array_from_npz(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "samples" in data.files:
            return np.asarray(data["samples"])
        # Fallback: first array entry
        for k in data.files:
            arr = np.asarray(data[k])
            if arr.ndim >= 1:
                return arr
    return None


def _get_theta_target(
    run_dir: Path, flat_md: dict[str, Any], theta_dim: int
) -> np.ndarray | None:
    # Look in metrics.json first (various spellings)
    for key in flat_md:
        lk = key.lower()
        if any(
            val = flat_md[key]
            if any(
                tk in lk
                for tk in (
                    "theta_target",
                    "theta.dagger",
                    "theta_dagger",
                    "target_theta",
                    "theta*",
                )
            )
        ):
            if isinstance(val, (list, tuple)):
                arr = np.asarray(val, dtype=float).reshape(-1)
            elif isinstance(val, str):
                try:
                    arr = np.asarray([float(t) for t in re.split(r"[,\s]+", val.strip()) if t], dtype=float)
                except Exception:
                    continue
            else:
                continue
            if arr.size == theta_dim:
                return arr
    # Fallback: config.json → run_cfg.theta_true
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            jd = _read_json(cfg)
            flat = _flatten(jd)
            for key, val in flat.items():
                lk = key.lower()
                if "run_cfg.theta_true" in lk.replace(" ", "") and isinstance(
                    val, (list, tuple)
                ):
                    arr = np.asarray(val, dtype=float).reshape(-1)
                    if arr.size == theta_dim:
                        return arr
        except Exception:
            pass
    return None


def _hpdi_contains(x: np.ndarray, target: float, level: float) -> float:
    x = np.sort(np.asarray(x, dtype=float).reshape(-1))
    n = x.size
    if n == 0:
        return np.nan
    k = int(np.ceil(level * n))
    widths = x[k - 1 :] - x[: n - k + 1]
    j = int(np.argmin(widths))
    lo = x[j]
    hi = x[j + k - 1]
    return float(lo <= target <= hi)


def _central_contains(x: np.ndarray, target: float, level: float) -> float:
    a = (1.0 - level) / 2.0
    lo = float(np.quantile(x, a))
    hi = float(np.quantile(x, 1.0 - a))
    return float(lo <= target <= hi)


def _coverage_and_mse_from_samples(
    samples: np.ndarray, theta_target: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    samples: (N, D), theta_target: (D,)
    returns: hpdi_cov (D,), central_cov (D,), mse (D,)
    """
    if samples.ndim != 2:
        raise ValueError("samples must be (N, D)")
    N, D = samples.shape
    hpdi = np.zeros(D, dtype=float)
    cent = np.zeros(D, dtype=float)
    mse = ((samples - theta_target[None, :]) ** 2).mean(axis=0)
    for d in range(D):
        col = samples[:, d]
        hpdi[d] = _hpdi_contains(col, float(theta_target[d]), level)
        cent[d] = _central_contains(col, float(theta_target[d]), level)
    return hpdi, cent, mse


# ---------------------- Metrics parsing ---------------------


def _parse_metrics(md: dict[str, Any]) -> dict[str, Any]:
    """Schema-agnostic parse. Returns: hpdi_cov, central_cov, bias, mse, logprob, ppd_median."""
    out: Dict[str, Any] = {}
    flat = _flatten(md)
    flat_l = {k.lower(): v for k, v in flat.items()}

    def _find_vec(
        keys_any: Iterable[str], keys_all: Iterable[str] = (), length_at_least: int = 1
    ) -> np.ndarray | None:
        for k, v in flat_l.items():
            if not isinstance(v, (list, tuple, np.ndarray)):
                continue
            s = k
            if any(a in s for a in keys_any) and all(a in s for a in keys_all):
                arr = np.asarray(v)
                if arr.size >= length_at_least and arr.ndim >= 1:
                    return arr.astype(float).reshape(-1)
        return None

    def _find_scalar(keys_all: Iterable[str]) -> Optional[float]:
        best = None
        for k, v in flat_l.items():
            if isinstance(v, (int, float)) and all(a in k for a in keys_all):
                best = float(v)
        return best

    def _find_array(keys_any: Iterable[str], keys_all: Iterable[str] = ()) -> np.ndarray | None:
        for k, v in flat_l.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                s = k
                if all(a in s for a in keys_all) and all(a in s for a in keys_any):
                    arr = np.asarray(v, dtype=float).reshape(-1)
                    if arr.size:
                        return arr
        return None


    # coverage
    hpdi_cov = (
        _coerce_bool_array(_find_vec(keys_any=("hpdi",), keys_all=("cover",)))
        or _coerce_bool_array(_find_vec(keys_any=("hpdi",), keys_all=("contains",)))
        or _coerce_bool_array(_find_vec(keys_any=("coverage", "hpdi"), keys_all=()))
    )
    central_cov = (
        _coerce_bool_array(_find_vec(keys_any=("central",), keys_all=("cover",)))
        or _coerce_bool_array(_find_vec(keys_any=("central",), keys_all=("contains",)))
        or _coerce_bool_array(_find_vec(keys_any=("coverage", "central"), keys_all=()))
    )

    # bias and MSE vectors
    bias = _as_float_array(_find_vec(keys_any=("bias",), keys_all=()))
    mse = _as_float_array(
        _find_vec(keys_any=("mse", "mean_squared", "squared_error"), keys_all=())
    )

    # log-probability at target
    logprob = None
    # prefer target-specific keys if present
    for keys in (
        ("log", "prob", "target"),
        ("log", "lik", "target"),
        ("logp", "target"),
        ("lp", "target"),
        ("synlik", "target"),
        ("log", "density", "target"),
    ):
        v = _find_scalar(keys_all=keys)
        if v is not None:
            logprob = v
            break
    if logprob is None:
        logprob = _find_scalar(keys_all=("log", "prob"))

    # posterior predictive distance median
    ppd_median = None
    for keys in (("ppd", "median"), ("ppd", "l2", "median"), ("ppc", "median")):
        v = _find_scalar(keys_all=keys)
        if v is not None:
            ppd_median = v
            break
    if ppd_median is None:
        # derive median from arrays if present
        arr = (
            _find_array(keys_any=("ppd", "l2"), keys_all=())
            or _find_array(keys_any=("ppd", "dist"), keys_all=())
            or _find_array(keys_any=("ppc", "l2"), keys_all=())
        )
        if arr is not None and arr.size:
            ppd_median = float(np.median(arr))

    out["hpdi_cov"] = hpdi_cov
    out["central_cov"] = central_cov
    out["bias"] = bias
    out["mse"] = mse
    out["logprob"] = logprob
    out["ppd_median"] = ppd_median
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
        dbg_left = args.debug_runs
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

            # Fallback: compute coverage/MSE from saved posterior samples if missing
            need_cov = not isinstance(
                parsed.get("hpdi_cov"), np.ndarray
            ) or not isinstance(parsed.get("central_cov"), np.ndarray)
            need_mse = not isinstance(parsed.get("mse"), np.ndarray)
            if need_cov or need_mse:
                # try to load samples
                samps = _first_array_from_npz(run_dir / "posterior_samples.npz")
                if samps is None:
                    samps = _first_array_from_npz(
                        run_dir / "posterior_samples_robust.npz"
                    )
                if (
                    samps is not None
                    and samps.ndim == 2
                    and samps.shape[1] == theta_dim
                ):
                    tgt = _get_theta_target(run_dir, _flatten(md), theta_dim)
                    if tgt is not None:
                        hpdi_v, cent_v, mse_v = _coverage_and_mse_from_samples(
                            samps, tgt, args.level
                        )
                        if need_cov:
                            parsed["hpdi_cov"] = hpdi_v
                            parsed["central_cov"] = cent_v
                        if need_mse:
                            parsed["mse"] = mse_v
                    elif args.verbose:
                        print(f"[info] target not found for {run_dir}")
                else:
                    if args.verbose:
                        print(f"[info] samples not found/bad shape in {run_dir}")

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
            if args.debug_runs > 0 and dbg_left > 0:
                dbg_left -= 1
                print(f"[debug] {m} seed={seed} dir={run_dir.name}")
                print(f"        hpdi_cov={parsed.get('hpdi_cov')}")
                print(f"        central_cov={parsed.get('central_cov')}")
                print(f"        bias={parsed.get('bias')}")
                print(f"        mse={parsed.get('mse')}")
                print(f"        logprob={parsed.get('logprob')}")
                print(f"        ppd_median={parsed.get('ppd_median')}")

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

    def _escape_tex(s: str) -> str:
        # Minimal TeX escaping for table text
        if s is None:
            return ""
        rep = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        out = str(s)
        for k, v in rep.items():
            out = out.replace(k, v)
        return out

    def _df_to_latex_simple(
        df: pd.DataFrame, *, percent: bool, decimals: int, caption: str, label: str
    ) -> str:
        cols = [str(c) for c in df.columns]
        colspec = "l" + "c" * len(cols)
        lines: list[str] = []
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{" + colspec + r"}")
        lines.append(r"\hline")
        header = [""] + [_escape_tex(c) for c in cols]
        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\hline")
        for r, idx in enumerate(df.index):
            row = [_escape_tex(str(idx))]
            for c in cols:
                v = df.iloc[r][c]
                if pd.isna(v):
                    row.append(r"--")
                else:
                    x = float(v)
                    if percent:
                        x = 100.0 * x
                    # int-like → no decimals
                    if abs(x - round(x)) < 1e-12:
                        row.append(f"{int(round(x))}")
                    else:
                        row.append(f"{x:.{decimals}f}")
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        if caption:
            lines.append(r"\caption{" + _escape_tex(caption) + r"}")
        if label:
            lines.append(r"\label{" + _escape_tex(label) + r"}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def _write_latex(
        df: pd.DataFrame,
        fname: str,
        percent: bool = False,
        caption: str = "",
        label: str = "",
    ):
        latex = _df_to_latex_simple(
            df, percent=percent, decimals=args.decimals, caption=caption, label=label
        )
        (out_dir / fname).write_text(latex)

    _write_latex(
        df_cov_hpdi,
        f"coverage_hpdi_{lvl_tag}.tex",
        percent=args.percent,
        caption=f"SVAR marginal HPDI {int(args.level * 100)}\\% coverage.",
        label="tab:svar_cov_hpdi",
    )
    _write_latex(
        df_cov_central,
        f"coverage_central_{lvl_tag}.tex",
        percent=args.percent,
        caption=f"SVAR marginal central {int(args.level * 100)}\\% coverage.",
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
