# Инструкция для жюри: проверка решения на rosbag

Пакет `tram_nav` (ROS 2 Humble, Python, `ament_python`). Вход — ручка контроллера и скорости колёс.
Выход — `/result/velocity` и `/result/position`. GNSS используется **только** как эталон для оценки
точности (и, при необходимости, один раз для построения карты линии).

## 0. Коротко

```bash
# сборка
mkdir -p ~/tram_ws/src && cd ~/tram_ws/src
git clone https://github.com/ilushenssss/odometer-on-model.git
cd ~/tram_ws && source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install --packages-select tram_nav
source install/setup.bash

# демо: воспроизвести прилагаемый bag, оценить против GNSS, построить графики
cd ~/tram_ws/src/odometer-on-model/data
ros2 launch tram_nav bag_replay.launch.py bag:=demo_bag track_csv:=demo_track_from_gnss.csv out_dir:=/tmp/tram_eval
ros2 run tram_nav plot_eval --dir /tmp/tram_eval --track demo_track_from_gnss.csv
cat /tmp/tram_eval/summary.json
```

## 1. Требования и сборка

* Ubuntu 22.04 + ROS 2 Humble (`ros-humble-desktop` или `ros-humble-ros-base`), `ros-humble-rosbag2*`
  (обычно уже есть), `python3-numpy`; для графиков — `python3-matplotlib`.
* Сборка — `colcon build` (см. выше). Проверка:

```bash
ros2 pkg executables tram_nav
# tram_nav evaluator | make_demo_bag | navigator | plot_eval | simulator | track_from_gnss
```

Тесты (необязательно, ~3 мин): `colcon test --packages-select tram_nav && colcon test-result --verbose` — 69 тестов.

## 2. Что лежит в репозитории для проверки

| файл | что это |
|---|---|
| [`data/demo_bag/`](../data/demo_bag) | rosbag2 (sqlite3, 9 МБ, 7 мин): ручка, 3 датчика колёс, GNSS-эталон 10 Гц, истина симулятора 10 Гц |
| [`data/demo_track_from_gnss.csv`](../data/demo_track_from_gnss.csv) | карта линии, построенная **из GNSS этого же bag** (`track_from_gnss`) |
| [`data/demo_track_true.csv`](../data/demo_track_true.csv) | точная карта (для сравнения) |
| [`tram_nav/config/jury.yaml`](../tram_nav/config/jury.yaml) | параметры навигатора для воспроизведения bag (имена/типы/единицы входных топиков, трамвай) |

Топики демо-bag (`ros2 bag info data/demo_bag`):

| топик | тип | частота | назначение |
|---|---|---|---|
| `/tram/controller_handle` | `std_msgs/Float64` | 50 Гц | вход: ручка $u \in [-1, 1]$ |
| `/tram/wheel_speeds` | `sensor_msgs/JointState` | 50 Гц | вход: `velocity[]` [рад/с] осей `axle_0, axle_2, axle_3`; `NaN` — нет данных |
| `/tram/gnss` | `sensor_msgs/NavSatFix` | 10 Гц | **только эталон** (шум 1,5 м, медленный дрейф) |
| `/tram/ground_truth` | `nav_msgs/Odometry` | 10 Гц | истина симулятора (для отладки) |

Сценарий демо: мокро → листья → мокро, залипание датчика, выбросы, экстренное торможение, смена
пассажиров, **100 с полной потери одометрии** (290–390 с). Полный 15-минутный bag любого сценария:
`ros2 run tram_nav make_demo_bag --scenario combined --out combined_bag`.

## 3. Воспроизведение rosbag

### 3.1 Одной командой (навигатор + оценщик + `ros2 bag play`)

```bash
ros2 launch tram_nav bag_replay.launch.py \
    bag:=<путь к bag> track_csv:=<карта.csv> \
    [params:=<свой yaml>] [rate:=1.0] [reference:=gnss] [gnss_topic:=/tram/gnss] [out_dir:=tram_nav_eval]
```

Launch запускает `tram_navigator` и `tram_nav_evaluator` с `use_sim_time:=true`, через 3 с выполняет
`ros2 bag play <bag> --clock 100 -r <rate>` и после окончания bag завершает всё и сохраняет результаты.

### 3.2 Вручную (три терминала)

```bash
# 1 — навигатор
ros2 run tram_nav navigator --ros-args --params-file <share>/tram_nav/config/jury.yaml \
     -p use_sim_time:=true -p track_csv:=<карта.csv>
# 2 — оценщик (необязательно)
ros2 run tram_nav evaluator --ros-args -p use_sim_time:=true -p track_csv:=<карта.csv> \
     -p reference:=gnss -p gnss_topic:=/tram/gnss -p csv_path:=eval.csv -p summary_path:=summary.json
# 3 — данные
ros2 bag play <bag> --clock 100
```

`<share>` = `$(ros2 pkg prefix tram_nav)/share`.

### 3.3 Свой bag с другими топиками

Посмотрите `ros2 bag info <bag>` и поправьте блок inputs в копии `jury.yaml`, затем передайте её как
`params:=my.yaml`:

| что в bag | параметры |
|---|---|
| ручка `std_msgs/Int16` от 0 (торможение) до 1000 (тяга), выбег = 500 | `handle_topic: /my/handle`, `handle_type: int16`, `handle_min: 0`, `handle_neutral: 500`, `handle_max: 1000` |
| скорости колёс в `JointState` с другими именами | `wheel_joint_names: [left_front, right_rear, trailer]`, `sensor_powered: [true, true, false]` |
| скорости в `Float32MultiArray` | `wheel_type: float32_array` |
| по топику на датчик `Float64`, об/мин | `wheel_type: float64`, `wheel_topics: [/w1, /w2, /w3]`, `wheel_units: rpm` |
| окружные скорости, км/ч | `wheel_units: km_h` |
| трамвай стартует не с начала карты | `initial_latlon: [55.75, 37.61]` (или `initial_position: <s, м>`) |
| другая модель трамвая | `tram.mass_nominal`, `tram.wheel_radius`, `tram.traction_*`, `tram.brake_*`, `tram.n_axles`, `tram.powered_axles` |

Полный список параметров — [PARAMETERS.md](PARAMETERS.md).

### 3.4 Карта линии

Если карты нет, постройте её из GNSS-трека эталонного рейса (в том же или отдельном bag, лучше в
хорошую погоду):

```bash
ros2 run tram_nav track_from_gnss --bag <bag> --topic /tram/gnss --out track.csv
# => lat,lon,alt,station  (шаг 2 м, стоянки ≥ 10 с — платформы)
```

На демо-bag: линия 2,88 км, боковое отклонение от истинной линии в среднем 1,0 м (максимум 3,9 м),
уклоны по шумной высоте GNSS — с точностью около 7 ‰ (окно сглаживания 250 м). Платформы ищутся по
стоянкам ≥ 10 с (`--min-dwell`): в демо одна долгая остановка после экстренного торможения ошибочно
принята за платформу. Проверьте столбец `station` в полученном CSV и при необходимости поправьте его
вручную или постройте карту по штатному рейсу.

## 4. Какие топики ожидать на выходе

```bash
ros2 topic list | grep -E "result|odometry|diagnostics"
ros2 topic echo /result/velocity --once
ros2 topic hz /result/position
```

| топик | тип | частота | содержание |
|---|---|---|---|
| **`/result/velocity`** | `geometry_msgs/TwistStamped` | = частота одометрии (`event`) или 50 Гц (`timer`) | `twist.linear.x` — скорость вдоль пути [м/с]; `twist.angular.z` — скорость рыскания $v\kappa$ [рад/с] |
| **`/result/position`** | `geometry_msgs/PoseStamped` | то же | `pose.position.x/y` — положение в системе карты (x — восток, y — север) [м]; `orientation` — курс |
| `/result/position_geo` | `sensor_msgs/NavSatFix` | то же | широта/долгота/ковариация (`status = NO_FIX`: это не спутниковое решение) |
| `/result/perf` | `std_msgs/Float64MultiArray` | 1 Гц | `[step_ms, latency_ms, rate_hz, cpu_pct, rss_mb]` |
| `/odometry` | `nav_msgs/Odometry` | как `/result/*` | поза + скорость **с ковариациями** ($\sigma$ вдоль пути, $\sigma_v$) |
| `/state` | `std_msgs/Float64MultiArray` | как `/result/*` | `s, v, a, σs, σv, η, масса, d, μ̂, rbf, k, режим, t_blind, n_used, n_fixes, x, y, ψ` |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 1 Гц | режим (`NOMINAL/DEGRADED/BLIND/STANDSTILL`), статус каждого датчика, задержка, частота, CPU, память |
| `/eval/errors` | `std_msgs/Float64MultiArray` | 5 Гц | оценщик: `[e_2d, e_along, e_v, rms_2d, max_2d, rms_v, n]` |
| `/tf` | `map → base_link` | | |

Типы `/result/*` меняются параметрами `result_velocity_type` (`twist | vector3 | float64`) и
`result_position_type` (`pose | point | pose_cov`), имена — `result_velocity_topic`,
`result_position_topic`. В режиме `event` метка времени выхода равна метке времени входного
сообщения одометрии, поэтому оценку удобно сопоставлять с GNSS по времени.

## 5. Логи, метрики, задержка

**Консоль навигатора** (при старте — конфигурация входов/выходов; при завершении — сводка):

```
tram navigator: mode=event, rate=50 Hz, wheels=joint_state ['axle_0', 'axle_2', 'axle_3'] [rad_s], ...
outputs 21001, step 0.316 ms (max 2.80), latency 0.96 ms (max 3.35), rate 50.0 Hz, CPU 13.7 %, RSS 72 MB, overruns 0
```

**Консоль оценщика** — каждые 10 с:

```
[gnss] n=20892: 2D RMS 12.07 m, max 47.7 m, speed RMS 0.704 m/s, latency 1.05 ms, rate 50.0 Hz, CPU 15.0 %, RSS 72 MB
```

**Файлы оценщика** (`out_dir`):

* `summary.json` — СКО/p95/максимум ошибки 2D и вдоль пути, СКО скорости, задержка, частота, CPU,
  память. Перезаписывается каждые 10 с и при завершении;
* `eval.csv` — каждый сопоставленный отсчёт: `t, x_ref, y_ref, x_est, y_est, e_2d, e_along, v_ref, v_est, e_v`;
* графики: `ros2 run tram_nav plot_eval --dir <out_dir> --track <карта>` → `eval_trajectory.png`,
  `eval_errors.png`.

**Задержка и ресурсы в реальном времени:**

```bash
ros2 topic echo /result/perf          # [step_ms, latency_ms, rate_hz, cpu_pct, rss_mb]
ros2 topic echo /diagnostics          # latency_mean_ms, latency_max_ms, step_mean_ms, output_rate_hz, cpu_pct, rss_mb
ros2 topic hz /result/velocity
ros2 topic delay /result/velocity     # задержка относительно header.stamp (= метке входа в режиме event)
ros2 run rqt_runtime_monitor rqt_runtime_monitor   # /diagnostics в GUI (если установлен)
```

Определения:
* `step_ms` — время вычисления одного шага фильтра;
* `latency_ms` — время от **приёма** сообщения одометрии нодой до **публикации** оценки, учитывающей
  его (в режиме `event` — время вычисления и публикации; в режиме `timer` добавляется ожидание до
  тика, в среднем половина периода);
* `rate_hz` — фактическая частота публикации;
* `cpu_pct` — загрузка одного ядра процессом навигатора, `rss_mb` — резидентная память.

**Как оценщик сравнивает с GNSS:** GNSS (`NavSatFix`) переводится в локальную систему карты
(начало — первая точка карты с lat/lon или `geo_origin`). Эталон интерпретируется линейно на метку
времени каждой оценки (без экстраполяции, разрывы > 1 с не интерполируются). Эталонная скорость —
центральная разность GNSS-координат на ±2 с. Ошибка вдоль пути — разность проекций на линию.
Подробно — [ACCURACY.md](ACCURACY.md).

## 6. Частые вопросы

* **Нет сообщений на `/result/*`** — проверьте, что имена/типы входных топиков в yaml совпадают с bag
  (при старте навигатор печатает, что слушает); для bag нужен `use_sim_time:=true` и
  `ros2 bag play ... --clock`.
* **Положение сдвинуто с самого начала** — задайте `initial_latlon` / `initial_position`.
* **Карта в x, y без lat/lon, а эталон — GNSS** — задайте `geo_origin: [lat0, lon0]` обеим нодам.
* **Режим навигатора** виден в `/diagnostics`: `BLIND` — одометрии нет, идёт счисление по модели,
  неопределённость (`sigma_s_m`) растёт.
