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
    group_contains: str = ""  # optional substring to filter the GROUP path
    prefer_hpdi: bool = True
    out_json: str = "results/gnk/coverage_summary.json"
    out_tex: str = "results/gnk/coverage_table.tex"
    methods: tuple[str, ...] = ("npe", "rnpe", "pnpe", "prnpe")
    param_labels: tuple[str, ...] = ("A", "B", "g", "k")
    level: float = 0.95
    decimals_cov: int = 2
    decimals_bias: int = 3
    decimals_sd: int = 3


def _collect(root: Path, group_contains: str, methods: list[str]) -> Any:
    rows = []
    for p in root.rglob("metrics.json"):
        if group_contains and group_contains not in str(p.parent):
            continue
        with open(p) as f:
            m = json.load(f)
        meth = m.get("method") or "unknown"
        if meth not in methods:
            continue
        rows.append(
            (
                meth,
                np.array(m["bias"], float),
                np.array(m["post_sd"], float),
                np.array(m.get("hit_hpdi", m.get("hit_central")), float),
            )
        )
    return rows


def _agg(rows: Any, methods: Any) -> Any:
    out: dict[str, dict[str, list[float]]] = {}
    for meth in methods:
        B, S, H = [], [], []
        for m, b, s, h in rows:
            if m == meth:
                B.append(b)
                S.append(s)
                H.append(h)
        if not B:
            continue
        B = np.stack(B)
        S = np.stack(S)
        H = np.stack(H)
        R = B.shape[0]
        cov = H.mean(axis=0)
        se = np.sqrt(cov * (1 - cov) / R)
        out[meth] = {
            "R": R,
            "coverage": cov.tolist(),
            "coverage_se": se.tolist(),
            "bias_mean": B.mean(0).tolist(),
            "bias_sd": B.std(0, ddof=1).tolist(),
            "avg_post_sd": S.mean(0).tolist(),
        }
    return out


def _fmt_cell(cov, bias, sdbias, avgsd, dc, db):  # type: ignore
    cov_s = f"{cov:.{dc}f}"
    return f"{cov_s} / {bias:+.{db}f} ± {sdbias:.{db}f} / {avgsd:.{db}f}"


def _write_tex(path: Path, summary, methods, labels, level, dc, db):  # type: ignore
    lines = []
    lines.append("\\begin{table}[!htb]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{@{}l" + "c" * len(labels) + "@{}}\\toprule")
    lines.append("Method & " + " & ".join(f"${label_name}$" for label_name in labels) + " \\\\ \\midrule")
    for m in methods:
        if m not in summary:
            continue
        s = summary[m]
        cells = []
        for j in range(len(labels)):
            cells.append(
                _fmt_cell(  # type: ignore
                    s["coverage"][j],
                    s["bias_mean"][j],
                    s["bias_sd"][j],
                    s["avg_post_sd"][j],
                    dc,
                    db,
                )
            )
        lines.append(m.upper() + " & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    pct = int(level * 100)
    cap = (
        f"Coverage (Cov), bias of posterior mean (Bias), SD of bias, and average posterior SD (AvgSD) for g-and-k. "
        f"Cells show Cov / Bias $\\pm$ SD / AvgSD. Nominal {pct}\\% HPDI; R varies by method."
    )
    lines.append(f"\\caption{{{cap}}}")
    lines.append("\\label{tab:gnk_coverage}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines))


def main(a: Args) -> None:
    rows = _collect(Path(a.results_root), a.group_contains, list(a.methods))
    summary = _agg(rows, list(a.methods))
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_tex).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(summary, indent=2))
    _write_tex(
        Path(a.out_tex),
        summary,
        list(a.methods),
        list(a.param_labels),
        a.level,
        a.decimals_cov,
        a.decimals_bias,
    )


if __name__ == "__main__":
    tyro.cli(main)
