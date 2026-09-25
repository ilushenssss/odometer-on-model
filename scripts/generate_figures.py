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
LABELS = {"proposed": "proposed (adaptive model + FDI)", "odometry": "plain odometry", "model": "open-loop model"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.6,
})

TRACK = generate_demo_track()
SC_TITLES = {
    "nominal": "Nominal (dry rail)", "wet": "Wet rail", "autumn": "Autumn: leaves on the rails",
    "ice": "Icy rail", "sensor_faults": "Odometry faults + 300 s blackout",
    "blackout": "Wet rail, heavy tram, 300 s total odometry loss",
    "param_change": "Changing plant: passengers, lost converter, +15 % resistance",
    "combined": "Everything at once",
}


def _run(args):
    name, seed, nav_kw = args
    r = run_scenario(get_scenario(name, seed), track=TRACK, nav=NavigatorParams(**nav_kw), log_every=5)
    return name, seed, json.dumps(nav_kw), r.data, r.metrics(), r.step_times, r.navigator.rbf.w.tolist() \
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
    ax.plot(v * 3.6, [ph.traction_envelope(x, p) / 1e3 for x in v], color=C_PROP, label="traction (u = 1)")
    ax.plot(v * 3.6, [p.brake_force_max * ph.ed_share(x, p) / 1e3 for x in v], color=C_ODO,
            label="ED brake share of service brake")
    ax.plot(v * 3.6, [ph.resistance_force(x, p.mass_nominal, p) / 1e3 for x in v], color=C_MODEL,
            label="motion resistance (50 t)")
    ax.set_xlabel("speed [km/h]")
    ax.set_ylabel("force [kN]")
    ax.set_title("Traction / braking / resistance")
    ax.legend(fontsize=8)
    ax = axs[1]
    lam = np.linspace(0, 0.35, 400)
    for (name, ap), col in zip(WEATHER.items(), [C_PROP, C_ODO, C_MODEL, "#4a3aa7"]):
        pk = ph.mu_peak(5.0, ap)
        ax.plot(lam * 100, [ph.creep_curve(x, pk, ap) for x in lam], color=col, label=name)
    ax.set_xlabel("slip ratio λ [%]")
    ax.set_ylabel("adhesion coefficient μ")
    ax.set_title("Creep-force curve (unstable falling branch)")
    ax.legend(fontsize=8, title="rail condition", title_fontsize=8)
    ax = axs[2]
    s = np.linspace(0, TRACK.length, 1200)
    ax.plot(s, [1000 * math.tan(TRACK.grade_at(x)) for x in s], color=C_PROP)
    ax.axhline(0, color=GRID)
    for st in TRACK.stations:
        ax.axvline(st, color=GRID, lw=2, zorder=0)
    ax.set_xlabel("along-track distance s [m] (grey lines = stations)")
    ax.set_ylabel("grade [‰]")
    ax.set_title("Demo line: grade profile (known from the map)")
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
        ax.annotate(f"S{k}", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8, color=TEXT2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Demo tram line: {TRACK.length / 1000:.1f} km, {len(TRACK.stations)} stations, curves R 25-300 m")
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
    ax.set_yticks([0, 1, 2], ["axle 0 (motor)", "axle 2 (motor)", "axle 3 (trailer)"], fontsize=8)
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS[k]) for k in ["OK", "SLIP", "SUSPECT", "FAULT", "DROPOUT"]]
    ax.legend(handles, ["OK", "slip / slide", "suspect", "fault / stuck", "dropout"], ncol=5, fontsize=8,
              loc="upper center", bbox_to_anchor=(0.5, -0.35))


def fig_scenario(name, d, m):
    t = d["t"]
    fig, axs = plt.subplots(4, 1, figsize=(12, 10.5), sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 1.6, 2.0, 0.9]})
    ax = axs[0]
    ax.plot(t, d["odometry_v"] * 3.6, color=C_ODO, lw=1.0, label=LABELS["odometry"])
    ax.plot(t, d["v"] * 3.6, color=C_TRUTH, lw=2.2, label="ground truth")
    ax.plot(t, d["proposed_v"] * 3.6, color=C_PROP, lw=1.4, label=LABELS["proposed"])
    ax.set_ylabel("speed [km/h]")
    ax.set_ylim(-2, max(70, float(np.nanmax(d["v"])) * 3.6 + 8))
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.set_title(f"{SC_TITLES.get(name, name)}  -  scenario '{name}'")
    blind = d["mode"] == 2
    for a in axs[:3]:
        _shade(a, t, blind)
    ax = axs[1]
    ax.plot(t, d["odometry_v"] - d["v"], color=C_ODO, lw=0.9, label=LABELS["odometry"])
    ax.plot(t, d["proposed_v"] - d["v"], color=C_PROP, lw=1.2, label=LABELS["proposed"])
    ax.set_ylabel("speed error [m/s]")
    lim = max(0.6, min(3.0, 1.3 * float(np.nanpercentile(np.abs(d["odometry_v"] - d["v"]), 99.5))))
    ax.set_ylim(-lim, lim)
    ax.legend(ncol=2, fontsize=8, loc="upper right")
    ax = axs[2]
    es = d["proposed_s"] - d["s"]
    ax.fill_between(t, -3 * d["sigma_s"], 3 * d["sigma_s"], color=C_BAND, alpha=0.13, lw=0,
                    label="±3σ reported by the navigator")
    ax.plot(t, d["odometry_s"] - d["s"], color=C_ODO, lw=1.0, label=LABELS["odometry"])
    ax.plot(t, d["model_s"] - d["s"], color=C_MODEL, lw=1.0, label=LABELS["model"])
    ax.plot(t, es, color=C_PROP, lw=1.5, label=LABELS["proposed"])
    lim = max(20.0, 1.4 * float(np.max(np.abs(es))), 3.2 * float(np.percentile(d["sigma_s"], 50)))
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("position error [m]")
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    status_strip(axs[3], d)
    axs[3].set_xlabel("time [s]" + ("   (hatched: BLIND - no usable odometry)" if blind.any() else ""))
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
    ax.plot(d["t"], d["mass"] / 1e3, color=C_TRUTH, lw=2.2, label="true mass")
    ax.plot(d["t"], d["mass"] / d["eff"] / 1e3, color=C_TRUTH, lw=1.2, ls="--",
            label="true mass / traction efficiency")
    ax.plot(d["t"], d["mass_est"] / 1e3, color=C_PROP, label="estimated effective mass (m_nom / η)")
    ax.axvline(300, color=C_ODO, lw=1, ls="--")
    ax.annotate("traction converter lost\n(-33 % tractive effort)", (300, 35), xytext=(8, 10),
                textcoords="offset points", fontsize=8, color=TEXT2)
    ax.set_ylim(35, 95)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mass [t]")
    ax.set_title("Mass / traction identification (η)")
    ax.legend(fontsize=8, loc="upper left")
    d = res["autumn"]["data"]
    ax = axs[1]
    ax.plot(d["t"], d["mu_true"], color=C_TRUTH, lw=2.2, label="true peak adhesion (weather)")
    ax.plot(d["t"], d["mu_hat"], color=C_PROP, label="estimated μ̂ (from slip onsets)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("μ")
    ax.set_title("Adhesion estimate - scenario 'autumn'")
    ax.legend(fontsize=8)
    ax = axs[2]
    for nm, col in [("nominal", C_PROP), ("param_change", C_ODO), ("autumn", C_MODEL)]:
        d = res[nm]["data"]
        ax.plot(d["t"], d["sigma_s"], color=col, label=nm)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("reported σ_s [m]")
    ax.set_title("Position uncertainty (drops at station fixes)")
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
    ax.plot(v * 3.6, true, color=C_TRUTH, lw=2.2, label="true resistance error only (+15 %)")
    ax.plot(v * 3.6, [net.predict(x, 0.0) for x in v], color=C_PROP,
            label="learned by RBF (absorbs traction loss / mass error too)")
    ax.axhline(0, color=GRID)
    ax.set_xlabel("speed [km/h]")
    ax.set_ylabel("residual specific force [m/s²]")
    ax.set_title("Optional RBF residual approximator, scenario 'param_change'")
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
    for ax, lab, tt in [(axs[0], "max |position error| [m] (log)", "Worst position error per run"),
                        (axs[1], "speed RMSE [m/s] (log)", "Speed RMSE per run")]:
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
    for ax, key, lab in [(axs[0], "rel_es_pct", "max position error / distance [%]"),
                         (axs[1], "rmse_v", "speed RMSE [m/s]")]:
        data = [[m[f"{k}_{key}"] for m in mc] for k in ("proposed", "odometry", "model")]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True)
        for patch, col in zip(bp["boxes"], [C_PROP, C_ODO, C_MODEL]):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
            patch.set_edgecolor(col)
        for med in bp["medians"]:
            med.set_color(C_TRUTH)
        ax.set_xticks([1, 2, 3], ["proposed", "odometry", "open-loop model"])
        ax.set_yscale("log")
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    fig.suptitle(f"Monte Carlo: {len(mc)} random runs (random scenario, seed, wheel wear, adhesion field)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "montecarlo.png")


def fig_timing(times):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    us = np.asarray(times) * 1e6
    ax.hist(us, bins=np.logspace(np.log10(max(us.min(), 20)), np.log10(us.max()), 60), color=C_PROP)
    ax.set_xscale("log")
    for q, lab in [(50, "median"), (99, "p99"), (99.9, "p99.9")]:
        val = np.percentile(us, q)
        ax.axvline(val, color=C_TRUTH, lw=1, ls="--")
        ax.annotate(f"{lab} {val:.0f} µs", (val, ax.get_ylim()[1] * 0.9), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=TEXT2, rotation=90, va="top")
    ax.set_xlabel("navigator step time [µs] (pure Python, one core)")
    ax.set_ylabel("count")
    ax.set_title(f"Real-time budget: 20 000 µs per 50 Hz cycle, {len(us)} steps")
    save(fig, "timing.png")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()
    names = list(SCENARIOS)
    n_mc = 8 if args.quick else 24
    rng = np.random.default_rng(2024)
    mc_jobs = [(str(rng.choice(names)), int(rng.integers(100, 10_000)), {}) for _ in range(n_mc)]
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
                abl[f"{name} {kw}"] = metrics
                if name == "param_change":
                    rbf_w = w
            else:
                metrics["scenario"], metrics["seed"] = name, seed
                mc.append(metrics)
    for n in names:
        fig_scenario(n, res[n]["data"], res[n]["metrics"])
    fig_adaptation(res)
    fig_rbf(rbf_w)
    fig_summary(res, names)
    fig_montecarlo(mc)
    fig_timing(times)

    # ---------------- tables ----------------
    out = {"scenarios": {n: res[n]["metrics"] for n in names}, "ablation": abl, "montecarlo": mc}
    with open(os.path.join(ROOT, "docs", "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    lines = ["| scenario | distance, km | max pos. error: proposed / odometry / model, m | "
             "pos. error, % of distance (proposed) | speed RMSE: proposed / odometry / model, m/s | "
             "within ±3σ, % |", "|---|---|---|---|---|---|"]
    for n in names:
        m = res[n]["metrics"]
        lines.append(f"| `{n}` | {m['distance'] / 1000:.2f} | **{m['proposed_max_es']:.1f}** / "
                     f"{m['odometry_max_es']:.1f} / {m['model_max_es']:.1f} | {m['proposed_rel_es_pct']:.2f} | "
                     f"**{m['proposed_rmse_v']:.3f}** / {m['odometry_rmse_v']:.3f} / {m['model_rmse_v']:.3f} | "
                     f"{m['within_3sigma_pct']:.0f} |")
    lines += ["", "Ablation (same seed):", "", "| run | max pos. error, m | speed RMSE, m/s |", "|---|---|---|"]
    for k, m in abl.items():
        lines.append(f"| {k} | {m['proposed_max_es']:.1f} | {m['proposed_rmse_v']:.3f} |")
    lines += ["", "Monte Carlo runs:", "", "| scenario | seed | max pos. error, m (% of distance) | speed RMSE, m/s |",
              "|---|---|---|---|"]
    for m in sorted(mc, key=lambda q: (q["scenario"], q["seed"])):
        lines.append(f"| {m['scenario']} | {m['seed']} | {m['proposed_max_es']:.1f} ({m['proposed_rel_es_pct']:.2f} %) "
                     f"| {m['proposed_rmse_v']:.3f} |")
    for key in ("proposed", "odometry", "model"):
        e = np.array([m[f"{key}_rel_es_pct"] for m in mc])
        v = np.array([m[f"{key}_rmse_v"] for m in mc])
        lines.append(f"\nMonte Carlo `{key}`: position error median {np.median(e):.2f} %, p90 "
                     f"{np.percentile(e, 90):.2f} %, max {e.max():.2f} %; speed RMSE median {np.median(v):.3f}, "
                     f"max {v.max():.3f} m/s")
    t = np.asarray(times) * 1e6
    lines.append(f"\nStep time: mean {t.mean():.0f} µs, median {np.median(t):.0f} µs, p99 "
                 f"{np.percentile(t, 99):.0f} µs, max {t.max():.0f} µs ({len(t)} steps)")
    with open(os.path.join(ROOT, "docs", "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
