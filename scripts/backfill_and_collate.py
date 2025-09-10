from __future__ import annotations

import json
from dataclasses import dataclass  # add this import
from pathlib import Path

import numpy as np
import tyro


def np_central_ci(x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.quantile(x, [alpha / 2, 1 - alpha / 2], axis=0)
    return q[0], q[1]


def np_hpdi(x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    # x: (K,D) float
    K, D = x.shape
    m = int(np.ceil((1.0 - alpha) * K))
    lo = np.empty(D)
    hi = np.empty(D)
    xs = np.sort(x, axis=0)  # (K,D)
    widths = xs[m - 1 :, :] - xs[: K - m + 1, :]  # (K-m+1, D)
    idx = np.argmin(widths, axis=0)  # (D,)
    for j in range(D):
        i = idx[j]
        lo[j] = xs[i, j]
        hi[j] = xs[i + m - 1, j]
    return lo, hi


def np_compute_rep_metrics(
    samples: np.ndarray, theta_target: np.ndarray, level: float = 0.95
) -> dict[str, object]:
    # samples: (K,D) float; theta_target: (D,)
    alpha = 1.0 - level
    mu = samples.mean(0)
    sd = samples.std(0, ddof=0)
    bias = mu - theta_target

    lo_c, hi_c = np_central_ci(samples, alpha)
    hit_c = (theta_target >= lo_c) & (theta_target <= hi_c)
    lo_h, hi_h = np_hpdi(samples, alpha)
    hit_h = (theta_target >= lo_h) & (theta_target <= hi_h)

    return {
        "n_post": int(samples.shape[0]),
        "level": float(level),
        "post_mean": mu.tolist(),
        "post_sd": sd.tolist(),
        "bias": bias.tolist(),
        "ci_central": {"lo": lo_c.tolist(), "hi": hi_c.tolist()},
        "width_central": (hi_c - lo_c).tolist(),
        "hit_central": hit_c.astype(int).tolist(),
        "ci_hpdi": {"lo": lo_h.tolist(), "hi": hi_h.tolist()},
        "width_hpdi": (hi_h - lo_h).tolist(),
        "hit_hpdi": hit_h.astype(int).tolist(),
    }


def _key_for(dirp: Path) -> tuple[str, str, str]:
    """(method, group, seed) from results/gnk/<method>/<group>/seed-XX/<DATE>/"""
    parts = dirp.parts
    i = parts.index("gnk")
    return parts[i + 1].lower(), parts[i + 2], parts[i + 3]


@dataclass
class Args:
    results_root: str = "results/gnk"
    theta_target: tuple[float, float, float, float] = (2.3663, 4.1757, 1.7850, 0.1001)
    methods: tuple[str, ...] = ("npe", "rnpe", "pnpe", "prnpe")
    level: float = 0.95
    out_json: str = "results/gnk/coverage_summary.json"
    out_tex: str = "results/gnk/coverage_table.tex"
    param_labels: tuple[str, ...] = ("A", "B", "g", "k")
    decimals_cov: int = 2
    decimals_bias: int = 3

    filter_n_obs: int | None = None
    filter_n_sims: int | None = None
    filter_q_precond: float | None = None  # optional


def _samples_in(dirp: Path) -> np.ndarray | None:
    for name in ("posterior_samples_robust.npz", "posterior_samples.npz"):
        p = dirp / name
        if p.is_file():
            with np.load(p) as z:
                arr = z["samples"]
            # force float64 for stable quantiles; fallback to float32 if needed
            return arr.astype(np.float64, copy=False)
    return None


def _read_run_params(dirp: Path) -> tuple[int | None, int | None, float | None]:
    """Return (n_obs, n_sims, q_precond) from config.json, else from GROUP tokens."""
    # 1) config.json
    cfg = dirp / "config.json"
    if cfg.is_file():
        try:
            with open(cfg) as f:
                j = json.load(f)
            run = j.get("run", {})
            n_sims = int(run.get("n_sims")) if "n_sims" in run else None
            q_precond = float(run.get("q_precond")) if "q_precond" in run else None
            sim_kwargs = run.get("sim_kwargs", {}) or {}
            n_obs = int(sim_kwargs.get("n_obs")) if "n_obs" in sim_kwargs else None  # type: ignore
            return n_obs, n_sims, q_precond
        except Exception:
            pass
    # 2) parse from GROUP folder name: .../<method>/<GROUP>/seed-XX/<DATE>
    parts = dirp.parts
    try:
        i = parts.index("gnk")
        group = parts[i + 2]
        n_obs = n_sims = None
        q = None
        for tok in group.split("-"):
            if tok.startswith("n_obs_"):
                try:
                    n_obs = int(tok[len("n_obs_") :])
                except Exception:
                    pass
            elif tok.startswith("n_sims_"):
                try:
                    n_sims = int(tok[len("n_sims_") :])
                except Exception:
                    pass
            elif tok.startswith("q_"):
                try:
                    q = float(tok[len("q_") :])
                except Exception:
                    pass
        return n_obs, n_sims, q
    except Exception:
        return None, None, None


def _method_from_dir(dirp: Path) -> str:
    # expected: results/gnk/<method>/...
    parts = dirp.parts
    try:
        i = parts.index("gnk")
        return parts[i + 1]
    except Exception:
        return "unknown"


def _fmt_cell(cov, bias, sdbias, avgsd, dc, db):  # type: ignore
    return f"{cov:.{dc}f} / {bias:+.{db}f} ± {sdbias:.{db}f} / {avgsd:.{db}f}"


def main(a: Args) -> None:
    root = Path(a.results_root)
    # 0) find level for caption (optional)
    level = 0.95
    for mp in root.rglob("metrics.json"):
        try:
            level = float(json.load(open(mp))["level"])
            break
        except Exception:
            pass

    # 1) Latest-only selection by metrics.json
    latest: dict[tuple[str, str, str], Path] = {}
    for mp in root.rglob("metrics.json"):
        d = mp.parent  # .../seed-XX/<DATE>
        try:
            key = _key_for(d)
        except Exception:
            continue
        if key not in latest or d.name > latest[key].name:
            latest[key] = d

    # 2) Aggregate per method
    methods = tuple(m.lower() for m in a.methods)
    by_m = {m: {"bias": [], "psd": [], "hit": []} for m in methods}

    for d in latest.values():
        meth, _, _ = _key_for(d)
        if meth not in by_m:
            continue
        mj = json.load(open(d / "metrics.json"))
        by_m[meth]["bias"].append(np.array(mj["bias"], float))
        by_m[meth]["psd"].append(np.array(mj["post_sd"], float))
        hits = np.array(mj.get("hit_hpdi", mj.get("hit_central")), float)
        by_m[meth]["hit"].append(hits)

    summary: dict[str, dict[str, list[float] | int]] = {}
    for m in methods:
        if not by_m[m]["hit"]:
            continue
        B = np.stack(by_m[m]["bias"])
        S = np.stack(by_m[m]["psd"])
        H = np.stack(by_m[m]["hit"])
        R = B.shape[0]
        cov = H.mean(0)
        se = np.sqrt(cov * (1 - cov) / R)
        summary[m] = {
            "R": R,
            "coverage": cov.tolist(),
            "coverage_se": se.tolist(),
            "bias_mean": B.mean(0).tolist(),
            "bias_sd": B.std(0, ddof=1).tolist(),
            "avg_post_sd": S.mean(0).tolist(),
        }

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(summary, indent=2))

    # 3) LaTeX
    def _fmt_cell(cov, bias, sdbias, avgsd, dc, db):
        return f"{cov:.{dc}f} / {bias:+.{db}f} ± {sdbias:.{db}f} / {avgsd:.{db}f}"

    lines = []
    lines.append("\\begin{table}[!htb]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{@{}l" + "c" * len(a.param_labels) + "@{}}\\toprule")
    lines.append(
        "Method & " + " & ".join(f"${p}$" for p in a.param_labels) + " \\\\ \\midrule"
    )
    for m in methods:
        if m not in summary:
            continue
        s = summary[m]
        cells = [
            _fmt_cell(
                s["coverage"][j],
                s["bias_mean"][j],
                s["bias_sd"][j],
                s["avg_post_sd"][j],
                a.decimals_cov,
                a.decimals_bias,
            )
            for j in range(len(a.param_labels))
        ]
        lines.append(m.upper() + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    pct = int(round(level * 100))
    lines.append(
        f"\\caption{{Coverage (Cov), bias (Bias), SD of bias, and average posterior SD (AvgSD) for g-and-k. "
        f"Cells show Cov / Bias $\\pm$ SD / AvgSD. Nominal {pct}\\% HPDI.}}"
    )
    lines.append("\\label{tab:gnk_coverage}")
    lines.append("\\end{table}")
    Path(a.out_tex).write_text("\n".join(lines))


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
