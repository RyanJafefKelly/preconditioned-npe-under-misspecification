from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro


@dataclass
class Args:
    results_root: str = "results/gnk"
    group_contains: str = ""  # optional substring to filter GROUP paths
    prefer_hpdi: bool = True
    methods: tuple[str, ...] = ("npe", "rnpe", "pnpe", "prnpe")
    param_labels: tuple[str, ...] = ("A", "B", "g", "k")
    level: float = 0.95
    decimals_cov: int = 2
    decimals_bias: int = 3
    decimals_sd: int = 3
    # outputs
    out_json: str = "results/gnk/metrics_summary.json"
    out_cov_tex: str = "results/gnk/coverage_table.tex"
    out_mse_tex: str = "results/gnk/mse_table.tex"
    out_ppd_tex: str = "results/gnk/ppd_table.tex"
    out_synlik_tex: str = "results/gnk/synlik_table.tex"


def _collect(root: Path, group_contains: str, methods: list[str]) -> Any:
    rows = []
    for p in root.rglob("metrics.json"):
        if group_contains and group_contains not in str(p.parent):
            continue
        with open(p) as f:
            m = json.load(f)
        meth = (m.get("method") or "unknown").lower()
        if meth not in methods:
            continue
        # pick hit flags
        if "hit_hpdi" in m:
            hit = np.array(m["hit_hpdi"], float)
        elif "hit_central" in m:
            hit = np.array(m["hit_central"], float)
        else:
            hit = np.full_like(np.array(m["post_mean"], float), np.nan, dtype=float)
        rows.append(
            dict(
                method=meth,
                bias=np.array(m["bias"], float),
                post_sd=np.array(m["post_sd"], float),
                se_mean=np.array(m.get("se_mean", []), float),
                post_mse=np.array(m.get("post_mse", []), float),
                hit=hit,
                ppd_mean=float(m.get("ppd_mean", np.nan)),
                ppd_sd=float(m.get("ppd_sd", np.nan)),
                synlik=float(m.get("synlik_logpdf_at_theta", np.nan)),
            )
        )
    return rows


def _agg(rows: list[dict], methods: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for meth in methods:
        R_bias, R_sd, R_hit, R_se, R_pmse = [], [], [], [], []
        R_ppd, R_ppd_sd, R_sl = [], [], []
        for r in rows:
            if r["method"] != meth:
                continue
            R_bias.append(r["bias"])
            R_sd.append(r["post_sd"])
            R_hit.append(r["hit"])
            if r["se_mean"].size:
                R_se.append(r["se_mean"])
            if r["post_mse"].size:
                R_pmse.append(r["post_mse"])
            R_ppd.append(r["ppd_mean"])
            R_ppd_sd.append(r["ppd_sd"])
            R_sl.append(r["synlik"])
        if not R_bias:
            continue
        B = np.stack(R_bias)  # (R, D)
        S = np.stack(R_sd)  # (R, D)
        H = np.stack(R_hit)  # (R, D)
        cov = np.nanmean(H, axis=0)
        se_cov = np.sqrt(cov * (1 - cov) / H.shape[0])
        se_mean = np.stack(R_se) if R_se else np.square(B)  # fallback
        mse = np.nanmean(se_mean, axis=0)
        rmse = np.sqrt(mse)
        pmse = np.nanmean(np.stack(R_pmse), axis=0) if R_pmse else np.nan * mse
        out[meth] = {
            "R": B.shape[0],
            "coverage": cov.tolist(),
            "coverage_se": se_cov.tolist(),
            "bias_mean": B.mean(0).tolist(),
            "bias_sd": B.std(0, ddof=1).tolist(),
            "avg_post_sd": S.mean(0).tolist(),
            "mse_mean": mse.tolist(),
            "rmse_mean": rmse.tolist(),
            "post_mse_mean": pmse.tolist(),
            "ppd_mean": float(np.nanmean(R_ppd)),
            "ppd_sd_over_reps": float(np.nanstd(R_ppd, ddof=1)),
            "synlik_mean": float(np.nanmean(R_sl)),
            "synlik_sd": float(np.nanstd(R_sl, ddof=1)),
        }
    return out


def _write_cov_tex(path: Path, summary, methods, labels, level, dc, db):
    lines = []
    lines += [
        "\\begin{table}[!htb]",
        "\\centering",
        "\\begin{tabular}{@{}l" + "c" * len(labels) + "@{}}\\toprule",
        "Method & " + " & ".join(f"${lab}$" for lab in labels) + " \\\\ \\midrule",
    ]
    for m in methods:
        if m not in summary:
            continue
        s = summary[m]
        cells = []
        for j in range(len(labels)):
            cov = s["coverage"][j]
            bias = s["bias_mean"][j]
            sdbias = s["bias_sd"][j]
            avgsd = s["avg_post_sd"][j]
            cells.append(f"{cov:.{dc}f} / {bias:+.{db}f} ± {sdbias:.{db}f} / {avgsd:.{db}f}")
        lines.append(m.upper() + " & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{Coverage (Cov), bias of posterior mean (Bias), SD of bias, and average posterior SD (AvgSD). Nominal {int(level * 100)}\\% intervals.}}",
        "\\label{tab:gnk_coverage}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines))


def _write_mse_tex(path: Path, summary, methods, labels, db):
    lines = []
    lines += [
        "\\begin{table}[!htb]",
        "\\centering",
        "\\begin{tabular}{@{}l" + "c" * len(labels) + "@{}}\\toprule",
        "Method & " + " & ".join(f"${lab}$" for lab in labels) + " \\\\ \\midrule",
    ]
    for m in methods:
        if m not in summary:
            continue
        s = summary[m]
        cells = []
        for j in range(len(labels)):
            mse = s["mse_mean"][j]
            rmse = s["rmse_mean"][j]
            cells.append(f"{mse:.{db}f} / {rmse:.{db}f}")
        lines.append(m.upper() + " & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{MSE and RMSE of posterior mean per parameter over replicates.}",
        "\\label{tab:gnk_mse}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines))


def _write_ppd_tex(path: Path, summary, methods):
    lines = []
    lines += [
        "\\begin{table}[!htb]",
        "\\centering",
        "\\begin{tabular}{@{}lc@{}}\\toprule",
        "Method & PPD mean $\\pm$ SD \\\\ \\midrule",
    ]
    for m in methods:
        if m not in summary:
            continue
        s = summary[m]
        lines.append(f"{m.upper()} & {s['ppd_mean']:.3f} $\\pm$ {s['ppd_sd_over_reps']:.3f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Posterior predictive distance between summary vectors and the observed summaries.}",
        "\\label{tab:gnk_ppd}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines))


def _write_synlik_tex(path: Path, summary, methods):
    lines = []
    lines += [
        "\\begin{table}[!htb]",
        "\\centering",
        "\\begin{tabular}{@{}lc@{}}\\toprule",
        "Method & Synthetic log-likelihood at $\\theta^{\\dagger}$ \\\\ \\midrule",
    ]
    for m in methods:
        if m not in summary:
            continue
        s = summary[m]
        lines.append(f"{m.upper()} & {s['synlik_mean']:.3f} $\\pm$ {s['synlik_sd']:.3f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Gaussian synthetic log-likelihood of observed summaries at the target parameter.}",
        "\\label{tab:gnk_synlik}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines))


def main(a: Args) -> None:
    rows = _collect(Path(a.results_root), a.group_contains, list(a.methods))
    summary = _agg(rows, list(a.methods))
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(summary, indent=2))
    _write_cov_tex(
        Path(a.out_cov_tex),
        summary,
        list(a.methods),
        list(a.param_labels),
        a.level,
        a.decimals_cov,
        a.decimals_bias,
    )
    _write_mse_tex(
        Path(a.out_mse_tex),
        summary,
        list(a.methods),
        list(a.param_labels),
        a.decimals_bias,
    )
    _write_ppd_tex(Path(a.out_ppd_tex), summary, list(a.methods))
    _write_synlik_tex(Path(a.out_synlik_tex), summary, list(a.methods))


if __name__ == "__main__":
    tyro.cli(main)
