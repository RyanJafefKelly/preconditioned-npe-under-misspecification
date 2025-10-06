#!/usr/bin/env python
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

# ---------- CLI ----------


@dataclass
class Args:
    results_root: str = "results/svar"  # e.g., results/svar or results/gnk
    out_dir: str | None = None
    group_hint: str | None = None  # substring to pick the GROUP dir
    level: float = 0.95
    percent: bool = False  # render coverage as percentages in LaTeX
    decimals: int = 2
    verbose: bool = True
    debug_runs: int = 0
    param_labels: tuple[str, ...] | None = None  # override row labels, e.g. ("A","B","g","k")


# ---------- Utilities ----------

# _METHODS_ORDER = ("npe", "pnpe", "rnpe", "prnpe", "npe_rs", "pnpe_rs")
_METHODS_ORDER = ("npe", "pnpe", "rnpe", "prnpe", "rf_abc_npe", "rf_abc_rnpe")
_ALLOWED = set(_METHODS_ORDER)
_TS_PAT = re.compile(r"^\d{8}-\d{6}$")


def _example_name_from_root(root: Path) -> str:
    # results/<example>
    try:
        return root.name
    except Exception:
        return "experiment"


def _discover_methods(root: Path) -> list[str]:
    cand = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name in _ALLOWED]
    missing = [m for m in _METHODS_ORDER if m not in cand]
    if missing:
        print(f"[warn] missing method dirs: {missing}")
    return [m for m in _METHODS_ORDER if m in cand]


def _pick_group_for_method(method_dir: Path, group_hint: str | None) -> str:
    groups = [p.name for p in method_dir.iterdir() if p.is_dir()]
    if group_hint:
        groups = [g for g in groups if group_hint in g]
    if not groups:
        raise RuntimeError(f"No GROUP directories under {method_dir}")
    counts: dict[str, int] = {}
    for g in groups:
        gpath = method_dir / g
        n = sum(1 for p in gpath.iterdir() if p.is_dir() and p.name.startswith("seed-"))
        counts[g] = n
    max_n = max(counts.values())
    return sorted([g for g, n in counts.items() if n == max_n])[-1]


def _latest_timestamp_dir(seed_dir: Path) -> Path | None:
    ts = [p for p in seed_dir.iterdir() if p.is_dir() and _TS_PAT.match(p.name)]
    return sorted(ts, key=lambda p: p.name)[-1] if ts else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return cast(dict[str, Any], json.load(f))


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(d, list):
        out[prefix] = d
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = d
    return out


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
    return (arr.astype(bool)).astype(float)


def _first_array_from_npz(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "samples" in data.files:
            return np.asarray(data["samples"])
        for k in data.files:
            arr = np.asarray(data[k])
            if arr.ndim >= 1:
                return arr
    return None


def _find_theta_labels(run_dir: Path, theta_dim: int, override: tuple[str, ...] | None) -> list[str]:
    if override and len(override) == theta_dim:
        return list(override)
    for fname in ("metrics.json", "artifacts.json"):
        f = run_dir / fname
        if f.exists():
            try:
                d = _read_json(f)
                labels = d.get("theta_labels") or (d.get("spec") or {}).get("theta_labels")
                if labels and len(labels) == theta_dim:
                    return list(labels)
            except Exception:
                pass
    return [f"p{i + 1}" for i in range(theta_dim)]


# ---------- Target + coverage helpers ----------


def _get_theta_target(run_dir: Path, flat_md: dict[str, Any], theta_dim: int) -> np.ndarray | None:
    for key, val in flat_md.items():
        lk = key.lower()
        if any(
            tk in lk
            for tk in (
                "theta_target",
                "theta.dagger",
                "theta_dagger",
                "target_theta",
                "theta*",
            )
        ):
            if isinstance(val, (list | tuple)):
                arr = np.asarray(val, dtype=float).reshape(-1)
            elif isinstance(val, str):
                try:
                    arr = np.asarray(
                        [float(t) for t in re.split(r"[,\s]+", val.strip()) if t],
                        dtype=float,
                    )
                except Exception:
                    continue
            else:
                continue
            if arr.size == theta_dim:
                return arr
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            jd = _read_json(cfg)
            flat = _flatten(jd)
            for k, v in flat.items():
                lk = k.lower()
                if "run_cfg.theta_true" in lk.replace(" ", "") and isinstance(v, (list | tuple)):
                    arr = np.asarray(v, dtype=float).reshape(-1)
                    if arr.size == theta_dim:
                        return arr
        except Exception:
            pass
    return None


def _hpdi_contains(x: np.ndarray, target: float, level: float) -> float:
    x = np.sort(np.asarray(x, dtype=float).reshape(-1))
    n = x.size
    if n == 0:
        return float("nan")
    k = int(np.ceil(level * n))
    widths = x[k - 1 :] - x[: n - k + 1]
    j = int(np.argmin(widths))
    lo, hi = x[j], x[j + k - 1]
    return float(lo <= target <= hi)


def _central_contains(x: np.ndarray, target: float, level: float) -> float:
    a = (1.0 - level) / 2.0
    lo, hi = float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))
    return float(lo <= target <= hi)


def _coverage_and_mse_from_samples(
    samples: np.ndarray, theta_target: np.ndarray, level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, D = samples.shape
    hpdi = np.zeros(D, dtype=float)
    cent = np.zeros(D, dtype=float)
    mse = ((samples - theta_target[None, :]) ** 2).mean(axis=0)
    for d in range(D):
        col = samples[:, d]
        hpdi[d] = _hpdi_contains(col, float(theta_target[d]), level)
        cent[d] = _central_contains(col, float(theta_target[d]), level)
    return hpdi, cent, mse


# ---------- Parse metrics.json (SVAR+GNK) ----------


def _parse_metrics(md: dict[str, Any]) -> dict[str, Any]:
    """Return: hpdi_cov, central_cov, bias, mse, logprob, ppd_median."""
    out: dict[str, Any] = {}
    flat = _flatten(md)
    flat_l = {k.lower(): v for k, v in flat.items()}

    def _vec(keys: list[str]) -> np.ndarray | None:
        for key in keys:
            v = flat_l.get(key)
            if isinstance(v, (list | tuple | np.ndarray)):
                return np.asarray(v, dtype=float).reshape(-1)
        return None

    def _scalar(keys: list[str]) -> float | None:
        for key in keys:
            v = flat_l.get(key)
            if isinstance(v, (int | float)):
                return float(v)
        return None

    # Coverage (booleans in your files)
    hpdi_cov = _coerce_bool_array(_vec(["hit_hpdi"]))
    central_cov = _coerce_bool_array(_vec(["hit_central"]))

    # Bias and MSE
    bias = _as_float_array(_vec(["bias"]))
    mse = _as_float_array(_vec(["post_mse", "mse"]))

    # Log probability at θ_target
    logprob = _scalar(["post_logpdf_at_theta"]) or _scalar(["post_logpdf_quantiles.q50"])

    # PPD median
    ppd_median = _scalar(["ppd_q50", "ppd_median"])

    out.update(
        dict(
            hpdi_cov=hpdi_cov,
            central_cov=central_cov,
            bias=bias,
            mse=mse,
            logprob=logprob,
            ppd_median=ppd_median,
        )
    )
    return out


# ---------- Aggregation ----------


def _aggregate(args: Args) -> None:
    root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir or (root / "_aggregates")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ex_name = _example_name_from_root(root)

    methods = _discover_methods(root)
    if args.verbose:
        print(f"[discover] methods: {methods}")

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
        gdir = mdir / g
        seeds: dict[int, Path] = {}
        for sdir in sorted(p for p in gdir.iterdir() if p.is_dir() and p.name.startswith("seed-")):
            mobj = re.match(r"seed-(\d+)", sdir.name)
            if not mobj:
                continue
            latest = _latest_timestamp_dir(sdir)
            if latest and (latest / "metrics.json").exists():
                seeds[int(mobj.group(1))] = latest
        chosen[m] = seeds

    # Infer theta_dim and labels
    theta_dim: int | None = None
    labels: list[str] | None = None

    cov_hpdi: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    cov_central: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    bias: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    mse: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    logprob: dict[str, list[float]] = {m: [] for m in methods}
    ppd_med: dict[str, list[float]] = {m: [] for m in methods}

    def _append_array(
        dst: dict[str, list[np.ndarray]],
        *,
        method: str,
        parsed: dict[str, Any],
        key: str,
        theta_dim: int,
        md_path: Path,
    ) -> None:
        arr = parsed.get(key)
        if isinstance(arr, np.ndarray):
            if arr.size != theta_dim:
                print(f"[warn] {key} shape {arr.size} != theta_dim {theta_dim} at {md_path}")
            else:
                dst[method].append(arr.astype(float))

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

            if theta_dim is None:
                for key in ("hpdi_cov", "central_cov", "bias", "mse"):
                    arr = parsed.get(key)
                    if isinstance(arr, np.ndarray):
                        theta_dim = int(arr.size)
                        break
                if theta_dim is None:
                    raise RuntimeError("Could not infer theta_dim from metrics.json files.")
                labels = _find_theta_labels(run_dir, theta_dim, args.param_labels)

            # Fallbacks via samples if needed
            need_cov = not isinstance(parsed.get("hpdi_cov"), np.ndarray) or not isinstance(
                parsed.get("central_cov"), np.ndarray
            )
            need_mse = not isinstance(parsed.get("mse"), np.ndarray)
            if need_cov or need_mse:
                samps = _first_array_from_npz(run_dir / "posterior_samples.npz")
                if samps is None:
                    samps = _first_array_from_npz(run_dir / "posterior_samples_robust.npz")
                if samps is not None and samps.ndim == 2 and theta_dim is not None and samps.shape[1] == theta_dim:
                    tgt = _get_theta_target(run_dir, _flatten(md), theta_dim)
                    if tgt is not None:
                        hpdi_v, cent_v, mse_v = _coverage_and_mse_from_samples(samps, tgt, args.level)
                        if need_cov:
                            parsed["hpdi_cov"], parsed["central_cov"] = hpdi_v, cent_v
                        if need_mse:
                            parsed["mse"] = mse_v
                    elif args.verbose:
                        print(f"[info] target not found for {run_dir}")
                elif args.verbose:
                    print(f"[info] samples not found/bad shape in {run_dir}")

            assert theta_dim is not None
            _append_array(
                cov_hpdi,
                method=m,
                parsed=parsed,
                key="hpdi_cov",
                theta_dim=theta_dim,
                md_path=md_path,
            )
            _append_array(
                cov_central,
                method=m,
                parsed=parsed,
                key="central_cov",
                theta_dim=theta_dim,
                md_path=md_path,
            )
            _append_array(
                bias,
                method=m,
                parsed=parsed,
                key="bias",
                theta_dim=theta_dim,
                md_path=md_path,
            )
            _append_array(
                mse,
                method=m,
                parsed=parsed,
                key="mse",
                theta_dim=theta_dim,
                md_path=md_path,
            )

            if isinstance(parsed.get("logprob"), (int | float)):
                logprob[m].append(float(parsed["logprob"]))
            if isinstance(parsed.get("ppd_median"), (int | float)):
                ppd_med[m].append(float(parsed["ppd_median"]))

            if args.debug_runs > 0 and dbg_left > 0:
                dbg_left -= 1
                print(f"[debug] {m} seed={seed} dir={run_dir.name}")
                for k in (
                    "hpdi_cov",
                    "central_cov",
                    "bias",
                    "mse",
                    "logprob",
                    "ppd_median",
                ):
                    print(f"        {k}={parsed.get(k)}")

    assert theta_dim is not None and labels is not None

    def _stack_mean(d: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for m in methods:
            arrs = d[m]
            out[m] = np.vstack(arrs).mean(axis=0) if arrs else np.full(theta_dim, np.nan)
        return out

    cov_hpdi_mean = _stack_mean(cov_hpdi)
    cov_central_mean = _stack_mean(cov_central)
    bias_mean = _stack_mean(bias)
    mse_mean = _stack_mean(mse)

    def _scalar_stats(vals: list[float]) -> tuple[float, float, int]:
        if not vals:
            return (math.nan, math.nan, 0)
        x = np.asarray(vals, float)
        return float(x.mean()), float(x.std(ddof=1) if x.size > 1 else 0.0), int(x.size)

    logprob_stats = {m: _scalar_stats(logprob[m]) for m in methods}
    ppd_stats = {m: _scalar_stats(ppd_med[m]) for m in methods}

    def _wide(per_method: dict[str, np.ndarray]) -> pd.DataFrame:
        rows = [[per_method[m][i] for m in methods] for i in range(theta_dim)]
        return pd.DataFrame(rows, index=labels, columns=methods)

    df_cov_hpdi = _wide(cov_hpdi_mean)
    df_cov_central = _wide(cov_central_mean)
    df_bias = _wide(bias_mean)
    df_mse = _wide(mse_mean)
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

    lvl_tag = str(int(round(args.level * 100)))
    out_dir.joinpath(f"coverage_hpdi_{lvl_tag}.csv").write_text(df_cov_hpdi.to_csv())
    out_dir.joinpath(f"coverage_central_{lvl_tag}.csv").write_text(df_cov_central.to_csv())
    out_dir.joinpath("bias.csv").write_text(df_bias.to_csv())
    out_dir.joinpath("mse.csv").write_text(df_mse.to_csv())
    out_dir.joinpath("logprob.csv").write_text(df_logprob.to_csv())
    out_dir.joinpath("ppd_median.csv").write_text(df_ppd.to_csv())

    # Minimal LaTeX (no Jinja2)
    def _escape_tex(s: str) -> str:
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

    def _df_to_latex(df: pd.DataFrame, *, percent: bool, decimals: int, caption: str, label: str) -> str:
        cols = [str(c) for c in df.columns]
        colspec = "l" + "c" * len(cols)
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\begin{tabular}{" + colspec + r"}",
            r"\hline",
        ]
        lines.append(" & ".join([""] + [_escape_tex(c) for c in cols]) + r" \\")
        lines.append(r"\hline")
        for r, idx in enumerate(df.index):
            row = [_escape_tex(str(idx))]
            for c in cols:
                v = df.iloc[r][c]
                if pd.isna(v):
                    row.append(r"--")
                else:
                    x = float(v)
                    x = 100.0 * x if percent else x
                    row.append(f"{int(round(x))}" if abs(x - round(x)) < 1e-12 else f"{x:.{decimals}f}")
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\hline", r"\end{tabular}"]
        if caption:
            lines.append(r"\caption{" + _escape_tex(caption) + r"}")
        if label:
            lines.append(r"\label{" + _escape_tex(label) + r"}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def _write(df: pd.DataFrame, name: str, caption: str, label: str, percent: bool = False) -> None:
        tex = _df_to_latex(df, percent=percent, decimals=args.decimals, caption=caption, label=label)
        out_dir.joinpath(name).write_text(tex)

    _write(
        df_cov_hpdi,
        f"coverage_hpdi_{lvl_tag}.tex",
        f"{ex_name.upper()} marginal HPDI {int(args.level * 100)}\\% coverage.",
        f"tab:{ex_name}_cov_hpdi",
        percent=args.percent,
    )
    _write(
        df_cov_central,
        f"coverage_central_{lvl_tag}.tex",
        f"{ex_name.upper()} marginal central {int(args.level * 100)}\\% coverage.",
        f"tab:{ex_name}_cov_central",
        percent=args.percent,
    )
    _write(
        df_bias,
        "bias.tex",
        f"{ex_name.upper()} marginal bias (per parameter).",
        f"tab:{ex_name}_bias",
    )
    _write(
        df_mse,
        "mse.tex",
        f"{ex_name.upper()} marginal mean squared error (per parameter).",
        f"tab:{ex_name}_mse",
    )
    _write(
        df_logprob,
        "logprob_at_target.tex",
        "Log probability at (pseudo-)true parameter: mean$\\pm$sd.",
        f"tab:{ex_name}_logprob",
    )
    _write(
        df_ppd,
        "ppd_distance_median.tex",
        "Posterior predictive distance (median): mean$\\pm$sd.",
        f"tab:{ex_name}_ppd",
    )

    manifest = {
        "results_root": str(root),
        "out_dir": str(out_dir),
        "example": ex_name,
        "groups_used": groups_used,
        "replicates_per_method": {m: len(chosen[m]) for m in methods},
        "theta_dim": int(theta_dim),
        "param_labels": labels,
        "files": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
    }
    out_dir.joinpath("summary.json").write_text(json.dumps(manifest, indent=2))
    if args.verbose:
        print(json.dumps(manifest, indent=2))


def main() -> None:
    args = tyro.cli(Args)
    _aggregate(args)


if __name__ == "__main__":
    main()
