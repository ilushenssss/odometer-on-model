#!/usr/bin/env python3
"""Run all evaluation scenarios and produce the figures / tables of the README.

    python3 scripts/generate_figures.py            # all scenarios + Monte Carlo
    python3 scripts/generate_figures.py --quick    # fewer Monte Carlo runs

Outputs: docs/images/*.png and docs/results.md / docs/results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tram_nav"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tram_nav.core import physics as ph  # noqa: E402
from tram_nav.core.evaluation import run_scenario  # noqa: E402
from tram_nav.core.params import WEATHER, NavigatorParams, TramParams  # noqa: E402
from tram_nav.core.scenarios import SCENARIOS, get_scenario  # noqa: E402
from tram_nav.core.track import generate_demo_track  # noqa: E402

IMG = os.path.join(ROOT, "docs", "images")
os.makedirs(IMG, exist_ok=True)

# --- palette (validated reference categorical order; truth = neutral ink) ----
C_TRUTH = "#3d3d3a"
C_PROP = "#2a78d6"     # proposed navigator
C_ODO = "#eb6834"      # plain odometry
C_MODEL = "#1baf7a"    # open-loop model
C_BAND = "#2a78d6"
SURFACE = "#fcfcfb"
GRID = "#e4e3df"
TEXT2 = "#52514e"
STATUS = {"OK": "#d9d8d3", "SLIP": "#eda100", "SUSPECT": "#e87ba4", "FAULT": "#e34948",
          "DROPOUT": "#4a3aa7", "STUCK": "#e34948", "INVALID": "#e34948"}
CODE2STATUS = {0: "OK", 1: "SLIP", 2: "SUSPECT", 3: "FAULT", 4: "DROPOUT", 5: "STUCK", 6: "INVALID"}
LABELS = {"proposed": "предложенный навигатор", "odometry": "обычная одометрия", "model": "модель без обратной связи"}
WEATHER_RU = {"dry": "сухо", "wet": "мокро", "leaves": "листья", "ice": "лёд"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.6,
})

TRACK = generate_demo_track()
SC_TITLES = {
    "nominal": "Номинальный режим (сухие рельсы)", "wet": "Мокрые рельсы", "autumn": "Осень: листья на рельсах",
    "ice": "Обледенелые рельсы", "sensor_faults": "Отказы одометрии + 300 с полной потери",
    "blackout": "Мокро, тяжёлый вагон, 300 с полной потери одометрии",
    "param_change": "Меняющийся объект: пассажиры, отказ преобразователя, +15 % сопротивления",
    "combined": "Всё сразу",
    "demo": "Демо для жюри: мокро → листья, отказы, 100 с без одометрии",
    "leaves_blackout": "Трудный случай: потеря одометрии при отправлении на листьях",
}


def _run(args):
    name, seed, nav_kw = args
    r = run_scenario(get_scenario(name, seed), track=TRACK, nav=NavigatorParams(**nav_kw), log_every=5)
    m = r.metrics()
    m["trajectory"] = r.trajectory(TRACK)
    return name, seed, json.dumps(nav_kw), r.data, m, r.step_times, r.navigator.rbf.w.tolist() \
        if r.navigator.rbf is not None else None


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.relpath(path, ROOT))


# ---------------------------------------------------------------------------
def fig_physics():
    p = TramParams()
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    v = np.linspace(0, 22, 300)
    ax = axs[0]
    ax.plot(v * 3.6, [ph.traction_envelope(x, p) / 1e3 for x in v], color=C_PROP, label="тяга (u = 1)")
    ax.plot(v * 3.6, [p.brake_force_max * ph.ed_share(x, p) / 1e3 for x in v], color=C_ODO,
            label="доля ЭДТ в служебном торможении")
    ax.plot(v * 3.6, [ph.resistance_force(x, p.mass_nominal, p) / 1e3 for x in v], color=C_MODEL,
            label="сопротивление движению (50 т)")
    ax.set_xlabel("скорость, км/ч")
    ax.set_ylabel("сила, кН")
    ax.set_title("Тяга, торможение, сопротивление")
    ax.legend(fontsize=8)
    ax = axs[1]
    lam = np.linspace(0, 0.35, 400)
    for (name, ap), col in zip(WEATHER.items(), [C_PROP, C_ODO, C_MODEL, "#4a3aa7"]):
        pk = ph.mu_peak(5.0, ap)
        ax.plot(lam * 100, [ph.creep_curve(x, pk, ap) for x in lam], color=col, label=WEATHER_RU.get(name, name))
    ax.set_xlabel("проскальзывание λ, %")
    ax.set_ylabel("коэффициент сцепления μ")
    ax.set_title("Кривая крипа (неустойчивая падающая ветвь)")
    ax.legend(fontsize=8, title="состояние рельсов", title_fontsize=8)
    ax = axs[2]
    s = np.linspace(0, TRACK.length, 1200)
    ax.plot(s, [1000 * math.tan(TRACK.grade_at(x)) for x in s], color=C_PROP)
    ax.axhline(0, color=GRID)
    for st in TRACK.stations:
        ax.axvline(st, color=GRID, lw=2, zorder=0)
    ax.set_xlabel("путь s, м (серые линии — станции)")
    ax.set_ylabel("уклон, ‰")
    ax.set_title("Профиль уклонов линии (из карты)")
    fig.tight_layout()
    save(fig, "physics.png")


def fig_track():
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    s = np.linspace(0, TRACK.length, 2000)
    xy = np.array([TRACK.pose(x)[:2] for x in s])
    ax.plot(xy[:, 0], xy[:, 1], color=C_TRUTH, lw=2.2)
    for k, st in enumerate(TRACK.stations):
        x, y, _ = TRACK.pose(st)
        ax.plot(x, y, "o", ms=8, color=C_PROP, mec=SURFACE, mew=2)
        ax.annotate(f"О{k}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8, color=TEXT2)
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(f"Демонстрационная линия: {TRACK.length / 1000:.1f} км, {len(TRACK.stations)} остановок, кривые R 25–300 м")
    save(fig, "track.png")


def status_strip(ax, d):
    t = d["t"]
    for i in range(3):
        codes = d[f"st{i}"].astype(int)
        start = 0
        for k in range(1, len(t) + 1):
            if k == len(t) or codes[k] != codes[start]:
                st = CODE2STATUS.get(codes[start], "OK")
                ax.barh(i, t[min(k, len(t) - 1)] - t[start], left=t[start], height=0.7,
                        color=STATUS[st], linewidth=0)
                start = k
    ax.set_yticks([0, 1, 2], ["ось 0 (моторная)", "ось 2 (моторная)", "ось 3 (немоторная)"], fontsize=8)
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS[k]) for k in ["OK", "SLIP", "SUSPECT", "FAULT", "DROPOUT"]]
    ax.legend(handles, ["исправен", "буксование / юз", "подозрение", "отказ / залипание", "нет данных"], ncol=5, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.35))


def fig_scenario(name, d, m):
    t = d["t"]
    fig, axs = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 1.6, 2.0, 0.9]})
    ax = axs[0]
    ax.plot(t, d["odometry_v"] * 3.6, color=C_ODO, lw=1.0, label=LABELS["odometry"])
    ax.plot(t, d["v"] * 3.6, color=C_TRUTH, lw=2.2, label="истина")
    ax.plot(t, d["proposed_v"] * 3.6, color=C_PROP, lw=1.4, label=LABELS["proposed"])
    ax.set_ylabel("скорость, км/ч")
    ax.set_ylim(-2, max(70, float(np.nanmax(d["v"])) * 3.6 + 8))
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.set_title(f"{SC_TITLES.get(name, name)} — сценарий «{name}»")
    blind = d["mode"] == 2
    for a in axs[:3]:
        _shade(a, t, blind)
    ax = axs[1]
    ax.plot(t, d["odometry_v"] - d["v"], color=C_ODO, lw=0.9, label=LABELS["odometry"])
    ax.plot(t, d["proposed_v"] - d["v"], color=C_PROP, lw=1.2, label=LABELS["proposed"])
    ax.set_ylabel("ошибка скорости, м/с")
    lim = max(0.6, min(3.0, 1.3 * float(np.nanpercentile(np.abs(d["odometry_v"] - d["v"]), 99.5))))
    ax.set_ylim(-lim, lim)
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    ax = axs[2]
    es = d["proposed_s"] - d["s"]
    ax.fill_between(t, -3 * d["sigma_s"], 3 * d["sigma_s"], color=C_BAND, alpha=0.13, lw=0,
                    label="±3σ, сообщаемая навигатором")
    ax.plot(t, d["odometry_s"] - d["s"], color=C_ODO, lw=1.0, label=LABELS["odometry"])
    ax.plot(t, d["model_s"] - d["s"], color=C_MODEL, lw=1.0, label=LABELS["model"])
    ax.plot(t, es, color=C_PROP, lw=1.5, label=LABELS["proposed"])
    lim = max(20.0, 1.4 * float(np.max(np.abs(es))), 3.2 * float(np.percentile(d["sigma_s"], 50)))
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("ошибка положения, м")
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    status_strip(axs[3], d)
    axs[3].set_xlabel("время, с" + ("   (штриховка: режим BLIND — нет пригодной одометрии)" if blind.any() else ""))
    fig.tight_layout()
    save(fig, f"scenario_{name}.png")


def _shade(ax, t, mask):
    if not mask.any():
        return
    start = None
    for k in range(len(t)):
        if mask[k] and start is None:
            start = t[k]
        if (not mask[k] or k == len(t) - 1) and start is not None:
            ax.axvspan(start, t[k], facecolor="none", edgecolor="#9a9994", hatch="///", lw=0, zorder=0)
            start = None


def fig_adaptation(res):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    d = res["param_change"]["data"]
    ax = axs[0]
    ax.plot(d["t"], d["mass"] / 1e3, color=C_TRUTH, lw=2.2, label="истинная масса")
    ax.plot(d["t"], d["mass"] / d["eff"] / 1e3, color=C_TRUTH, lw=1.2, ls="--",
            label="истинная масса / КПД тяги")
    ax.plot(d["t"], d["mass_est"] / 1e3, color=C_PROP, label="оценка эффективной массы (m₀ / η)")
    ax.axvline(300, color=C_ODO, lw=1, ls="--")
    ax.annotate("отказ тягового преобразователя\n(−33 % силы тяги)", (300, 35), xytext=(8, 10),
                textcoords="offset points", fontsize=8, color=TEXT2)
    ax.set_ylim(35, 95)
    ax.set_xlabel("время, с")
    ax.set_ylabel("масса, т")
    ax.set_title("Идентификация массы / тяги (η)")
    ax.legend(fontsize=8, loc="upper left")
    d = res["autumn"]["data"]
    ax = axs[1]
    ax.plot(d["t"], d["mu_true"], color=C_TRUTH, lw=2.2, label="истинное пиковое сцепление (погода)")
    ax.plot(d["t"], d["mu_hat"], color=C_PROP, label="оценка μ̂ (по подтверждённым буксованиям)")
    ax.set_xlabel("время, с")
    ax.set_ylabel("μ")
    ax.set_title("Оценка сцепления — сценарий «autumn»")
    ax.legend(fontsize=8)
    ax = axs[2]
    for nm, col in [("nominal", C_PROP), ("param_change", C_ODO), ("autumn", C_MODEL)]:
        d = res[nm]["data"]
        ax.plot(d["t"], d["sigma_s"], color=col, label=nm)
    ax.set_xlabel("время, с")
    ax.set_ylabel("сообщаемая σ_s, м")
    ax.set_title("Неопределённость положения (сброс на привязках)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "adaptation.png")


def fig_rbf(w):
    if w is None:
        return
    from tram_nav.core.rbf import RBFApproximator
    npar = NavigatorParams()
    net = RBFApproximator(TramParams().v_max, npar.rbf_v_centers, npar.rbf_u_centers)
    net.w = np.asarray(w)
    p = TramParams()
    v = np.linspace(0, 20, 200)
    # true unmodelled residual: 15 % extra resistance (mass ~ 58 t at start, varies)
    true = [-0.15 * ph.resistance_force(x, 55_000, p) / (55_000 * (1 + p.rot_mass_factor)) for x in v]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(v * 3.6, true, color=C_TRUTH, lw=2.2, label="истинная ошибка сопротивления (+15 %)")
    ax.plot(v * 3.6, [net.predict(x, 0.0) for x in v], color=C_PROP,
            label="выучено RBF (поглощает и потерю тяги, и ошибку массы)")
    ax.axhline(0, color=GRID)
    ax.set_xlabel("скорость, км/ч")
    ax.set_ylabel("остаточное удельное усилие, м/с²")
    ax.set_title("Опциональный RBF-аппроксиматор, сценарий «param_change»")
    ax.legend(fontsize=8)
    save(fig, "rbf.png")


def fig_summary(res, names):
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
    x = np.arange(len(names))
    wdt = 0.26
    for j, (k, col) in enumerate([("proposed", C_PROP), ("odometry", C_ODO), ("model", C_MODEL)]):
        vals = [res[n]["metrics"][f"{k}_max_es"] for n in names]
        axs[0].bar(x + (j - 1) * wdt, vals, wdt - 0.03, color=col, label=LABELS[k])
        vals = [res[n]["metrics"][f"{k}_rmse_v"] for n in names]
        axs[1].bar(x + (j - 1) * wdt, vals, wdt - 0.03, color=col, label=LABELS[k])
    for ax, lab, tt in [(axs[0], "макс. |ошибка положения|, м (лог.)", "Худшая ошибка положения за прогон"),
                        (axs[1], "СКО скорости, м/с (лог.)", "СКО скорости за прогон")]:
        ax.set_yscale("log")
        ax.set_xticks(x, names, rotation=20, fontsize=8)
        ax.set_ylabel(lab)
        ax.set_title(tt)
        ax.grid(axis="x", visible=False)
    axs[0].legend(fontsize=8)
    for i, n in enumerate(names):
        v = res[n]["metrics"]["proposed_max_es"]
        axs[0].annotate(f"{v:.0f}", (i - wdt, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=7, color=TEXT2)
    fig.tight_layout()
    save(fig, "summary.png")


def fig_montecarlo(mc):
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, key, lab in [(axs[0], "rel_es_pct", "макс. ошибка положения / путь, %"),
                         (axs[1], "rmse_v", "СКО скорости, м/с")]:
        data = [[m[f"{k}_{key}"] for m in mc] for k in ("proposed", "odometry", "model")]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True)
        for patch, col in zip(bp["boxes"], [C_PROP, C_ODO, C_MODEL]):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
            patch.set_edgecolor(col)
        for med in bp["medians"]:
            med.set_color(C_TRUTH)
        ax.set_xticks([1, 2, 3], ["предложенный", "одометрия", "модель без ОС"])
        ax.set_yscale("log")
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Монте-Карло: {len(mc)} случайных прогонов (сценарий, зерно, износ колёс, поле сцепления)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "montecarlo.png")


def fig_trajectory_map(d, name):
    """Map of the reconstructed trajectory coloured by the 2-D error + zoom on the blackout."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("err", ["#cfe0f6", "#2a78d6", "#0d2f5c"])
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
    s = np.linspace(0, TRACK.length, 2000)
    xy = np.array([TRACK.pose(x)[:2] for x in s])
    e2d = np.hypot(d["px"] - d["x"], d["py"] - d["y"])
    ax = axs[0]
    ax.plot(xy[:, 0], xy[:, 1], color=GRID, lw=6, zorder=0, solid_capstyle="round")
    k = slice(None, None, 10)
    sc = ax.scatter(d["px"][k], d["py"][k], c=e2d[k], cmap=cmap, s=9, vmin=0, vmax=max(10.0, e2d.max()),
                    zorder=2, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01)
    cb.set_label("ошибка положения 2D, м")
    for j, st in enumerate(TRACK.stations):
        x, y, _ = TRACK.pose(st)
        ax.plot(x, y, "s", ms=5, color=C_TRUTH, zorder=3)
    xo, yo, _ = TRACK.pose(float(d["odometry_s"][-1]))
    ax.plot(d["x"][-1], d["y"][-1], "o", ms=10, color=C_TRUTH, mec=SURFACE, mew=2, zorder=4,
            label="истинное конечное положение")
    ax.plot(d["px"][-1], d["py"][-1], "o", ms=7, color=C_PROP, mec=SURFACE, mew=1.5, zorder=5,
            label=f"навигатор: ошибка {e2d[-1]:.1f} м")
    far = math.hypot(xo - d["x"][-1], yo - d["y"][-1]) > 300.0
    ax.plot([] if far else [xo], [] if far else [yo], "D", ms=7, color=C_ODO, mec=SURFACE, mew=1.5, zorder=5,
            label=f"одометрия: ошибка {math.hypot(xo - d['x'][-1], yo - d['y'][-1]):.0f} м"
                  + (" (за пределами рисунка)" if far else ""))
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(f"Восстановленная траектория, сценарий «{name}»")
    ax.legend(fontsize=8, loc="lower left")
    # zoom: +-20 s around the largest error inside the blind phase (or overall)
    ax = axs[1]
    blind = d["mode"] == 2
    cand = np.nonzero(blind)[0] if blind.any() else np.arange(len(e2d))
    ic = int(cand[np.argmax(e2d[cand])])
    dtl = float(np.median(np.diff(d["t"])))
    half = int(20.0 / dtl)
    sel = np.arange(max(0, ic - half), min(len(e2d), ic + half), max(1, int(2.0 / dtl)))
    ax.plot(xy[:, 0], xy[:, 1], color=GRID, lw=8, zorder=0, solid_capstyle="round")
    for i in sel:
        ax.plot([d["x"][i], d["px"][i]], [d["y"][i], d["py"][i]], color=C_PROP, lw=0.8, alpha=0.5)
    ax.plot(d["x"][sel], d["y"][sel], "o", ms=6, color=C_TRUTH, label="истина (каждые 2 с)")
    ax.plot(d["px"][sel], d["py"][sel], "o", ms=4.5, color=C_PROP, label="навигатор")
    xs = np.concatenate([d["x"][sel], d["px"][sel]])
    ys = np.concatenate([d["y"][sel], d["py"][sel]])
    pad = 25.0
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    i0, i1 = sel[0], sel[-1]
    ax.set_aspect("equal")
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ttl = "Лупа: без одометрии" if blind[ic] else "Лупа"
    ax.set_title(f"{ttl}, t = {d['t'][i0]:.0f}–{d['t'][i1]:.0f} с, макс. ошибка {e2d[ic]:.1f} м")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save(fig, f"trajectory_map_{name}.png")


def fig_trajectory_errors(res, names):
    fig, axs = plt.subplots(len(names), 1, figsize=(12, 3.2 * len(names)), sharex=False)
    axs = np.atleast_1d(axs)
    for ax, n in zip(axs, names):
        d = res[n]["data"]
        t = d["t"]
        e_p = np.hypot(d["px"] - d["x"], d["py"] - d["y"])
        for key, col, lw in [("odometry", C_ODO, 1.0), ("model", C_MODEL, 1.0)]:
            tr = np.array([TRACK.pose(float(v))[:2] for v in d[f"{key}_s"]])
            ax.plot(t, np.hypot(tr[:, 0] - d["x"], tr[:, 1] - d["y"]) + 0.01, color=col, lw=lw, label=LABELS[key])
        ax.plot(t, e_p + 0.01, color=C_PROP, lw=1.5, label=LABELS["proposed"])
        ax.fill_between(t, 0.01, 3 * d["sigma_s"], color=C_BAND, alpha=0.12, lw=0, label="3σ навигатора")
        _shade(ax, t, d["mode"] == 2)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 5000)
        ax.set_ylabel("ошибка 2D, м (лог.)")
        ax.set_title(f"Ошибка положения на плоскости — {SC_TITLES.get(n, n)}")
        ax.legend(fontsize=8, ncol=4, loc="upper left")
    axs[-1].set_xlabel("время, с   (штриховка: режим BLIND)")
    fig.tight_layout()
    save(fig, "trajectory_errors.png")


def fig_timing(times):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    us = np.asarray(times) * 1e6
    ax.hist(us, bins=np.logspace(np.log10(max(us.min(), 20)), np.log10(us.max()), 60), color=C_PROP)
    ax.set_xscale("log")
    for q, lab in [(50, "медиана"), (99, "p99"), (99.9, "p99.9")]:
        val = np.percentile(us, q)
        ax.axvline(val, color=C_TRUTH, lw=1, ls="--")
        ax.annotate(f"{lab} {val:.0f} мкс", (val, ax.get_ylim()[1] * 0.9), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=TEXT2, rotation=90, va="top")
    ax.set_xlabel("время шага навигатора, мкс (чистый Python, одно ядро)")
    ax.set_ylabel("количество шагов")
    ax.set_title(f"Бюджет реального времени: 20 000 мкс на цикл 50 Гц, {len(us)} шагов")
    save(fig, "timing.png")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    names = list(SCENARIOS)
    mc_names = [n for n in names if n not in ("demo", "leaves_blackout")]
    n_mc = 8 if args.quick else 24
    rng = np.random.default_rng(2024)
    mc_jobs = [(str(rng.choice(mc_names)), int(rng.integers(100, 10_000)), {}) for _ in range(n_mc)]
    jobs = [(n, 1, {}) for n in names] + [("param_change", 1, {"rbf_enable": True}),
                                          ("blackout", 1, {"rbf_enable": True}),
                                          ("nominal", 1, {"station_snap": False}),
                                          ("autumn", 1, {"station_snap": False})] + mc_jobs
    print(f"running {len(jobs)} simulations on {args.jobs} processes ...")
    fig_physics()
    fig_track()
    res, abl, mc, times = {}, {}, [], []
    rbf_w = None
    with Pool(args.jobs) as pool:
        for k, (name, seed, kw, data, metrics, st, w) in enumerate(pool.imap(_run, jobs)):
            if k < len(names):
                res[name] = dict(data=data, metrics=metrics, rbf_w=w)
                times.extend(st.tolist())
                print(f"  {name:14s} max|e_s| {metrics['proposed_max_es']:7.1f} m   "
                      f"rmse_v {metrics['proposed_rmse_v']:.3f} m/s")
            elif k < len(names) + 4:
                metrics.pop("trajectory", None)
                abl[f"{name} {kw}"] = metrics
                if name == "param_change":
                    rbf_w = w
            else:
                metrics["scenario"], metrics["seed"] = name, seed
                metrics.pop("trajectory", None)
                mc.append(metrics)
    for n in names:
        fig_scenario(n, res[n]["data"], res[n]["metrics"])
    fig_adaptation(res)
    fig_rbf(rbf_w)
    fig_summary(res, names)
    fig_montecarlo(mc)
    fig_timing(times)
    for n in ("sensor_faults", "combined"):
        fig_trajectory_map(res[n]["data"], n)
    fig_trajectory_errors(res, ["sensor_faults", "combined", "autumn"])

    # ---------------- tables ----------------
    out = {"scenarios": {n: res[n]["metrics"] for n in names}, "ablation": abl, "montecarlo": mc}
    with open(os.path.join(ROOT, "docs", "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    lines = ["# Численные результаты", "",
             "Генерируются скриптом `scripts/generate_figures.py`.", "",
             "## Сценарии (seed 1)", "",
             "| сценарий | путь, км | макс. ошибка положения: предложенный / одометрия / модель, м | "
             "ошибка, % пути (предложенный) | СКО скорости: предложенный / одометрия / модель, м/с | "
             "внутри ±3σ, % |", "|---|---|---|---|---|---|"]
    for n in names:
        m = res[n]["metrics"]
        lines.append(f"| `{n}` | {m['distance'] / 1000:.2f} | **{m['proposed_max_es']:.1f}** / "
                     f"{m['odometry_max_es']:.1f} / {m['model_max_es']:.1f} | {m['proposed_rel_es_pct']:.2f} | "
                     f"**{m['proposed_rmse_v']:.3f}** / {m['odometry_rmse_v']:.3f} / {m['model_rmse_v']:.3f} | "
                     f"{m['within_3sigma_pct']:.0f} |")
    lines += ["", "## Траектория на плоскости (x, y) по выходу навигатора", "",
              "| сценарий | СКО 2D, м | p95 2D, м | макс. 2D, м | макс. 2D одометрии, м | "
              "ошибка курса p95, ° | макс. отклонение от рельсов, м | внутри 3σ, % |",
              "|---|---|---|---|---|---|---|---|"]
    for n in names:
        tp, to = res[n]["metrics"]["trajectory"]["proposed"], res[n]["metrics"]["trajectory"]["odometry"]
        lines.append(f"| `{n}` | **{tp['rms_2d']:.1f}** | {tp['p95_2d']:.1f} | {tp['max_2d']:.1f} | "
                     f"{to['max_2d']:.1f} | {tp['heading_p95_deg']:.1f} | {tp['max_off_track']:.3f} | "
                     f"{100 * tp['within_3sigma']:.0f} |")
    lines += ["", "## Абляция (seed 1)", "", "| прогон | макс. ошибка положения, м | СКО скорости, м/с |", "|---|---|---|"]
    for k, m in abl.items():
        lines.append(f"| {k} | {m['proposed_max_es']:.1f} | {m['proposed_rmse_v']:.3f} |")
    lines += ["", "## Прогоны Монте-Карло", "",
              "| сценарий | зерно | макс. ошибка положения, м (% пути) | СКО скорости, м/с |", "|---|---|---|---|"]
    for m in sorted(mc, key=lambda q: (q["scenario"], q["seed"])):
        lines.append(f"| {m['scenario']} | {m['seed']} | {m['proposed_max_es']:.1f} ({m['proposed_rel_es_pct']:.2f} %) "
                     f"| {m['proposed_rmse_v']:.3f} |")
    lines.append("")
    for key, ru in (("proposed", "предложенный"), ("odometry", "одометрия"), ("model", "модель без ОС")):
        e = np.array([m[f"{key}_rel_es_pct"] for m in mc])
        v = np.array([m[f"{key}_rmse_v"] for m in mc])
        lines.append(f"* Монте-Карло, {ru}: ошибка положения медиана {np.median(e):.2f} %, p90 "
                     f"{np.percentile(e, 90):.2f} %, макс. {e.max():.2f} %; СКО скорости медиана {np.median(v):.3f}, "
                     f"макс. {v.max():.3f} м/с")
    t = np.asarray(times) * 1e6
    lines.append(f"\n* Время шага: среднее {t.mean():.0f} мкс, медиана {np.median(t):.0f} мкс, p99 "
                 f"{np.percentile(t, 99):.0f} мкс, макс. {t.max():.0f} мкс ({len(t)} шагов)")
    with open(os.path.join(ROOT, "docs", "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
