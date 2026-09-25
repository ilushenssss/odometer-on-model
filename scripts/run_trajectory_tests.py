#!/usr/bin/env python3
"""Отдельный запуск тестов построения траектории и отчёт.

    python3 scripts/run_trajectory_tests.py
    # в окружении ROS 2 Humble дополнительно выполняется сквозной тест через топики:
    source /opt/ros/humble/setup.bash && python3 scripts/run_trajectory_tests.py

Результат: вывод pytest в консоль и отчёт docs/trajectory_tests.md
(таблица тестов + метрики восстановленной траектории).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG = os.path.join(ROOT, "tram_nav")
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "test"))

TITLES = {
    "test_distance_to_track_straight_line": "расстояние до рельсов (прямая, продолжение за концами)",
    "test_projection_lies_on_rails_everywhere": "проекция s → (x, y) лежит на рельсах по всей линии",
    "test_projection_heading_matches_track_tangent": "курс совпадает с касательной к пути",
    "test_projection_is_arc_length_parameterised": "1 м по s = 1 м на плоскости (параметризация длиной дуги)",
    "test_metrics_identity_and_chord_property": "метрики: тождество и «хорда ≤ дуги»",
    "test_trajectory_accuracy": "точность траектории (СКО и максимум 2D)",
    "test_trajectory_better_than_odometry": "траектория точнее обычной одометрии",
    "test_trajectory_stays_on_rails": "все точки траектории на рельсах",
    "test_trajectory_heading": "ошибка курса",
    "test_trajectory_continuous_while_moving": "непрерывность при движении, скачки только с уменьшением ошибки",
    "test_trajectory_error_within_reported_uncertainty": "ошибка внутри сообщаемой 3σ",
    "test_trajectory_follows_rails_during_blackout": "счисление по рельсам при полной потере одометрии",
    "test_trajectory_on_a_different_line": "другая геометрия линии (петля с S-образной кривой)",
    "test_ros_trajectory_end_to_end": "сквозной тест ROS 2: /clock → топики → nav_msgs/Odometry → траектория",
}


def run_pytest(targets, xml_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = PKG + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider", f"--junitxml={xml_path}", *targets]
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=PKG, env=env)


def parse(xml_path):
    rows = []
    for tc in ET.parse(xml_path).getroot().iter("testcase"):
        name = tc.get("name")
        status = "пройден"
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "**НЕ ПРОЙДЕН**"
        elif tc.find("skipped") is not None:
            status = "пропущен"
        rows.append((name, status, float(tc.get("time", 0.0))))
    return rows


def metrics_table():
    import numpy as np  # noqa: F401
    from test_trajectory import CASES, TRACK, _loop_track, _run
    from tram_nav.core.params import ScenarioConfig
    lines = ["| прогон | путь, м | СКО 2D, м | p95 2D, м | макс. 2D, м | макс. 2D одометрии, м | "
             "курс p95, ° | макс. отклонение от рельсов, м | внутри 3σ, % |",
             "|---|---|---|---|---|---|---|---|---|"]
    runs = [(name, sc, TRACK) for name, sc, _, _ in CASES]
    runs.append(("loop (другая линия)", ScenarioConfig(name="loop", duration=420, weather_profile=[(0.0, "wet")]),
                 _loop_track()))
    for name, sc, track in runs:
        res, tm = _run(sc, track)
        p, o = tm["proposed"], tm["odometry"]
        dist = res.data["s"][-1] - res.data["s"][0]
        lines.append(f"| {name} | {dist:.0f} | **{p['rms_2d']:.1f}** | {p['p95_2d']:.1f} | {p['max_2d']:.1f} | "
                     f"{o['max_2d']:.1f} | {p['heading_p95_deg']:.1f} | {p['max_off_track']:.4f} | "
                     f"{100 * p['within_3sigma']:.0f} |")
        print("  ", lines[-1], flush=True)
    return lines


def main():
    t0 = time.time()
    xml = os.path.join("/tmp", "tram_nav_trajectory_tests.xml")
    targets = ["test/test_trajectory.py"]
    try:
        import rclpy  # noqa: F401
        targets.append("test/test_ros_nodes.py::test_ros_trajectory_end_to_end")
        ros = True
    except ImportError:
        ros = False
        print("ROS 2 (rclpy) не найден - сквозной ROS-тест пропущен", flush=True)
    rc = run_pytest(targets, xml)
    rows = parse(xml)
    print("\nМетрики траектории:", flush=True)
    mt = metrics_table()
    n_ok = sum(1 for r in rows if r[1] == "пройден")
    out = ["# Тесты построения траектории", "",
           "Отчёт создан скриптом `scripts/run_trajectory_tests.py`. Навигатор оценивает путь $s$, "
           "траектория на плоскости получается через карту $s \\rightarrow (x, y, \\psi)$ и сравнивается "
           "с истинной траекторией симулятора.", "",
           f"**Итог: {n_ok} из {len(rows)} тестов пройдено** "
           f"({'с' if ros else 'без'} сквозного теста ROS 2), время {time.time() - t0:.0f} с.", "",
           "| тест | что проверяется | результат | время, с |", "|---|---|---|---|"]
    for name, status, dt in rows:
        base = name.split("[")[0]
        case = name[len(base):].strip("[]")
        what = TITLES.get(base, "") + (f" — `{case}`" if case else "")
        out.append(f"| `{base}` | {what} | {status} | {dt:.1f} |")
    out += ["", "## Метрики восстановленной траектории", "",
            "`nominal`, `autumn`, `param_change`: 400 с; `faults_blackout`: отказы датчиков и 150 с полной "
            "потери одометрии (220–370 с); `loop`: другая геометрия линии, мокрые рельсы.", ""] + mt
    path = os.path.join(ROOT, "docs", "trajectory_tests.md")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nотчёт: {os.path.relpath(path, ROOT)}; {n_ok}/{len(rows)} тестов пройдено")
    return rc


if __name__ == "__main__":
    sys.exit(main())
