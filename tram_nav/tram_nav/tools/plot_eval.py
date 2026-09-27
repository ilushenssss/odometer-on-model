#!/usr/bin/env python3
"""Plot the evaluator output (eval.csv + summary.json) of a bag replay.

    ros2 run tram_nav plot_eval --dir tram_nav_eval [--track track.csv]

Writes ``eval_trajectory.png`` and ``eval_errors.png`` into the same directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

C_REF, C_EST, GRID, TEXT2, SURF = "#3d3d3a", "#2a78d6", "#e4e3df", "#52514e", "#fcfcfb"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="tram_nav_eval")
    ap.add_argument("--track", default="")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                         "ytick.color": TEXT2, "legend.frameon": False, "axes.titleweight": "bold"})
    with open(os.path.join(args.dir, "eval.csv")) as f:
        rows = list(csv.DictReader(f))
    a = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    summ = {}
    sp = os.path.join(args.dir, "summary.json")
    if os.path.isfile(sp):
        with open(sp) as f:
            summ = json.load(f)
    t = a["t"] - a["t"][0]
    ref = summ.get("reference", "gnss")
    ref_ru = "GNSS-эталон" if ref == "gnss" else "истина"

    fig, ax = plt.subplots(figsize=(8, 6))
    if args.track:
        from tram_nav.core.track import TrackMap
        tr = TrackMap.from_csv(args.track)
        ax.plot(tr.x, tr.y, color=GRID, lw=6, zorder=0, label="карта линии")
    ax.plot(a["x_ref"], a["y_ref"], ".", ms=2, color=C_REF, label=ref_ru)
    ax.plot(a["x_est"], a["y_est"], "-", lw=1.2, color=C_EST, label="навигатор (/result/position)")
    ax.set_aspect("equal")
    ax.set_xlabel("x (восток), м")
    ax.set_ylabel("y (север), м")
    ax.set_title(f"Траектория: СКО 2D {summ.get('rms_2d_m', float('nan')):.1f} м, "
                 f"макс. {summ.get('max_2d_m', float('nan')):.1f} м")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(args.dir, "eval_trajectory.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    axs[0].plot(t, a["v_ref"] * 3.6, color=C_REF, lw=1.8, label=ref_ru)
    axs[0].plot(t, a["v_est"] * 3.6, color=C_EST, lw=1.2, label="навигатор (/result/velocity)")
    axs[0].set_ylabel("скорость, км/ч")
    axs[0].legend(fontsize=8, loc="upper right")
    axs[0].set_title("Сравнение с эталоном")
    axs[1].plot(t, a["e_v"], color=C_EST, lw=0.9)
    axs[1].set_ylabel("ошибка скорости, м/с")
    axs[1].set_title(f"СКО скорости {summ.get('rms_v_mps', float('nan')):.3f} м/с")
    axs[2].plot(t, a["e_2d"], color=C_EST, lw=1.2, label="ошибка 2D")
    axs[2].plot(t, np.abs(a["e_along"]), color=C_REF, lw=0.8, label="|ошибка вдоль пути|")
    axs[2].set_ylabel("ошибка положения, м")
    axs[2].set_xlabel("время от начала записи, с")
    axs[2].legend(fontsize=8, loc="upper left")
    perf = ""
    if "latency_ms_mean" in summ:
        perf = (f"; задержка {summ['latency_ms_mean']:.2f} мс, частота {summ['output_rate_hz']:.1f} Гц, "
                f"CPU {summ['cpu_pct_mean']:.1f} %, память {summ['rss_mb_max']:.0f} МБ")
    axs[2].set_title(f"Ошибка положения, p95 {summ.get('p95_2d_m', float('nan')):.1f} м{perf}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, "eval_errors.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"figures written to {args.dir}")


if __name__ == "__main__":
    main()
