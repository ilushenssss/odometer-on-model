# Тесты построения траектории

Отчёт создан скриптом `scripts/run_trajectory_tests.py`. Навигатор оценивает путь $s$, траектория на плоскости получается через карту $s \rightarrow (x, y, \psi)$ и сравнивается с истинной траекторией симулятора.

**Итог: 32 из 32 тестов пройдено** (с сквозного теста ROS 2), время 193 с.

| тест | что проверяется | результат | время, с |
|---|---|---|---|
| `test_distance_to_track_straight_line` | расстояние до рельсов (прямая, продолжение за концами) | пройден | 0.0 |
| `test_projection_lies_on_rails_everywhere` | проекция s → (x, y) лежит на рельсах по всей линии | пройден | 0.2 |
| `test_projection_heading_matches_track_tangent` | курс совпадает с касательной к пути | пройден | 0.0 |
| `test_projection_is_arc_length_parameterised` | 1 м по s = 1 м на плоскости (параметризация длиной дуги) | пройден | 0.0 |
| `test_metrics_identity_and_chord_property` | метрики: тождество и «хорда ≤ дуги» | пройден | 0.0 |
| `test_trajectory_accuracy` | точность траектории (СКО и максимум 2D) — `nominal` | пройден | 11.1 |
| `test_trajectory_better_than_odometry` | траектория точнее обычной одометрии — `nominal` | пройден | 0.0 |
| `test_trajectory_stays_on_rails` | все точки траектории на рельсах — `nominal` | пройден | 0.0 |
| `test_trajectory_heading` | ошибка курса — `nominal` | пройден | 0.0 |
| `test_trajectory_continuous_while_moving` | непрерывность при движении, скачки только с уменьшением ошибки — `nominal` | пройден | 0.0 |
| `test_trajectory_error_within_reported_uncertainty` | ошибка внутри сообщаемой 3σ — `nominal` | пройден | 0.0 |
| `test_trajectory_accuracy` | точность траектории (СКО и максимум 2D) — `autumn` | пройден | 10.9 |
| `test_trajectory_better_than_odometry` | траектория точнее обычной одометрии — `autumn` | пройден | 0.0 |
| `test_trajectory_stays_on_rails` | все точки траектории на рельсах — `autumn` | пройден | 0.0 |
| `test_trajectory_heading` | ошибка курса — `autumn` | пройден | 0.0 |
| `test_trajectory_continuous_while_moving` | непрерывность при движении, скачки только с уменьшением ошибки — `autumn` | пройден | 0.0 |
| `test_trajectory_error_within_reported_uncertainty` | ошибка внутри сообщаемой 3σ — `autumn` | пройден | 0.0 |
| `test_trajectory_accuracy` | точность траектории (СКО и максимум 2D) — `faults_blackout` | пройден | 11.0 |
| `test_trajectory_better_than_odometry` | траектория точнее обычной одометрии — `faults_blackout` | пройден | 0.0 |
| `test_trajectory_stays_on_rails` | все точки траектории на рельсах — `faults_blackout` | пройден | 0.0 |
| `test_trajectory_heading` | ошибка курса — `faults_blackout` | пройден | 0.0 |
| `test_trajectory_continuous_while_moving` | непрерывность при движении, скачки только с уменьшением ошибки — `faults_blackout` | пройден | 0.0 |
| `test_trajectory_error_within_reported_uncertainty` | ошибка внутри сообщаемой 3σ — `faults_blackout` | пройден | 0.0 |
| `test_trajectory_accuracy` | точность траектории (СКО и максимум 2D) — `param_change` | пройден | 10.6 |
| `test_trajectory_better_than_odometry` | траектория точнее обычной одометрии — `param_change` | пройден | 0.0 |
| `test_trajectory_stays_on_rails` | все точки траектории на рельсах — `param_change` | пройден | 0.0 |
| `test_trajectory_heading` | ошибка курса — `param_change` | пройден | 0.0 |
| `test_trajectory_continuous_while_moving` | непрерывность при движении, скачки только с уменьшением ошибки — `param_change` | пройден | 0.0 |
| `test_trajectory_error_within_reported_uncertainty` | ошибка внутри сообщаемой 3σ — `param_change` | пройден | 0.0 |
| `test_trajectory_follows_rails_during_blackout` | счисление по рельсам при полной потере одометрии | пройден | 11.3 |
| `test_trajectory_on_a_different_line` | другая геометрия линии (петля с S-образной кривой) | пройден | 11.5 |
| `test_ros_trajectory_end_to_end` | сквозной тест ROS 2: /clock → топики → nav_msgs/Odometry → траектория | пройден | 68.9 |

## Метрики восстановленной траектории

`nominal`, `autumn`, `param_change`: 400 с; `faults_blackout`: отказы датчиков и 150 с полной потери одометрии (220–370 с); `loop`: другая геометрия линии, мокрые рельсы.

| прогон | путь, м | СКО 2D, м | p95 2D, м | макс. 2D, м | макс. 2D одометрии, м | курс p95, ° | макс. отклонение от рельсов, м | внутри 3σ, % |
|---|---|---|---|---|---|---|---|---|
| nominal | 2957 | **3.6** | 6.6 | 7.6 | 43.5 | 4.0 | 0.0000 | 100 |
| autumn | 2665 | **9.1** | 16.6 | 18.1 | 38.7 | 3.7 | 0.0000 | 100 |
| faults_blackout | 3184 | **6.6** | 19.2 | 20.6 | 388.7 | 4.8 | 0.0000 | 100 |
| param_change | 2644 | **3.4** | 6.7 | 7.2 | 38.4 | 3.9 | 0.0000 | 100 |
| loop (другая линия) | 2866 | **6.0** | 11.2 | 11.5 | 42.5 | 4.1 | 0.0000 | 100 |
