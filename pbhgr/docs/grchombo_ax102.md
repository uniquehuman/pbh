# GRChombo на Hetzner AX102-1: сборка и первый запуск

Скрипт: `scripts/grchombo/build_grchombo_ax102.sh`. Целевая машина: AX102-1
(AMD Ryzen 9 7950X3D, 16 ядер / 32 потока, 128 ГБ DDR5 ECC, 2 × 1.92 ТБ NVMe),
Ubuntu 24.04 LTS из `installimage`. Рецепт взят из CI GRChombo
(`.github/workflows/gcc-tests.yml`) и шаблона
`InstallNotes/MakeDefsLocalExamples/ubuntu-gcc.Make.defs.local`; проверен
29.09.2026 на Ubuntu 24.04.4 (2 vCPU VPS) с GRChombo `37e6595`, Chombo (форк
GRChombo) `8684f2e`, GRTresna `2e671e1`, GCC 13.3, OpenMPI 4.1.6, HDF5 1.10.10,
PETSc 3.19.6.

## Что собирается

| Компонент | Откуда | Зачем |
|---|---|---|
| Chombo 3.2 (форк GRChombo) | github.com/GRChombo/Chombo | AMR-библиотека; собираются только BaseTools, BoxTools, AMRTools, AMRElliptic, AMRTimeDependent |
| GRChombo, пример `ScalarFieldCosmo` | github.com/GRChombo/GRChombo | эволюция BSSN/CCZ4 + массивное скалярное поле в периодическом ящике (старт этапа 1 плана 6.12) |
| GRTresna, пример `ScalarFieldCosmo` | github.com/GRTLCollaboration/GRTresna | решатель связей (CTTK) для нетривиальных начальных данных; выход подаётся в GRChombo через `restart_file` |
| Apparent-horizon finder | PETSc из apt (`petsc-dev`) | `-DUSE_AHFINDER`; без него не будет t_hor и M_AH |
| venv | numpy scipy matplotlib h5py yt | чтение plot-файлов Chombo HDF5 (yt понимает формат) |

Флаги: `OPT=HIGH DEBUG=FALSE MPI=TRUE OPENMPCC=TRUE`, `-O3 -march=native`
(на Zen 4 с GCC 13 это znver4, включая AVX-512 ветку simd-слоя GRChombo).
`-std=c++14 -fopenmp` Chombo добавляет сам.

## Порядок действий на новой машине

1. В Hetzner Robot заказать AX102-1, через `installimage` поставить Ubuntu 24.04
   (RAID 1 по умолчанию, ~1.9 ТБ под `/`). Зайти по ssh.
2. Скопировать скрипт и запустить целиком (под root или пользователем с sudo):

```bash
scp scripts/grchombo/build_grchombo_ax102.sh root@AX102:/root/
```

```bash
tmux new -s build 'bash /root/build_grchombo_ax102.sh all 2>&1 | tee /root/build.log'
```

   Стадии `all`: `pkgs clone config chombo grchombo grtresna env tests smoke python`.
   Любую можно перезапустить отдельно: `bash build_grchombo_ax102.sh chombo grchombo`.
   Время `all` на CCX33 (8 vCPU): около 5 минут, см. «Измерения» ниже;
   на AX102-1 (16 ядер) не дольше.
3. После сборки:

```bash
. /root/grchombo/env.sh && bash /root/build_grchombo_ax102.sh bench
```

   `bench` гоняет `ScalarFieldCosmo` N=96³ на 10 шагов в раскладках
   16×1, 8×2, 4×4, 32×1 (ранги × OpenMP-потоки; на CCX33 задать
   `BENCH_CONFIGS="8x1 4x2 4x1"`) и пишет `bench/summary.txt`.
   Выбрать самую быструю и использовать её в `grcrun -n R -t T`.

## Раскладка и запуск

```
/root/grchombo/
  Chombo/lib/mk/Make.defs.local   конфигурация сборки (генерируется стадией config)
  GRChombo/Examples/ScalarFieldCosmo/Main_ScalarFieldCosmo3d_ch.Linux.64.mpicxx.gfortran.OPTHIGH.MPI.OPENMPCC.ex
  GRTresna/Examples/ScalarFieldCosmo/Main_ScalarFieldCosmo3d_ch.*.ex
  env.sh          CHOMBO_HOME, PATH, OMP_*, ulimit, venv
  bin/grcrun      обёртка mpirun с привязкой к ядрам
  logs/           логи всех стадий и тестов
  smoke/          три проверочных прогона (аналитические НУ; GRTresna; рестарт из GRTresna)
  bench/          результаты бенчмарка
  VERSIONS.txt    коммиты трёх репозиториев на момент сборки
```

Рестарт из данных GRTresna: `restart_file` с абсолютным путём (относительный
GRChombo дописывает к `hdf5_subpath`), а каталоги `hdf5_*`, `pout_*`, `data_*`
надо создать заранее — на пути рестарта код их не создаёт и падает в
`H5Fcreate` (в `smoke 3/3` это учтено).

Запуск: `grcrun [-n RANKS] [-t THREADS] [-o log] EXE params.txt`. По умолчанию
16 рангов × 1 поток. Если `RANKS*THREADS` больше 16, включаются SMT-потоки
(`-n 32`). Под root обёртка сама добавляет `--allow-run-as-root`.

Переопределения через окружение: `GRC_ROOT`, `NJ`, `GRCHOMBO_REF`/`CHOMBO_REF`/
`GRTRESNA_REF` (ветка, тег или коммит), `WITH_AH=0` (без PETSc), `MARCH`, `OPT`,
`DEBUG=TRUE` (отладочная сборка, другое имя `.ex`), `EXAMPLES`, `TESTS`,
`SMOKE_N`, `SMOKE_RANKS`, `BENCH_*`, `PY_PKGS`.

## Что дальше (этап 1 плана 6.12)

Пример `ScalarFieldCosmo` уже содержит всё нужное для теста e = 0: периодический
ящик, начальное K из гамильтоновой связи (`InitialK.hpp`), диагностики
`rho_scaled`, `K_scaled`, среднее по ящику, lineout; AH finder подключён
(`AH_activate = 1` в params). Для нашей задачи заменяется `InitialScalarData.hpp`
(профиль ζ = μ exp(−k²r²/6) в chi = e^{−2ζ}, скалярное поле в осцилляциях с
m ≫ H, т.е. пылевой режим де Йонга) и связи решаются GRTresna
(`Examples/ScalarFieldCosmo` там же). Единицы: L ящика ≥ 4 r_m, H_i из
`scalar_mass` и амплитуды, как в de Jong et al. 2022.

## Измерения (29.09.2026)

Скрипт прогнан целиком на двух машинах; исправленные по ходу ошибки:
имена артефактов с `_ch` при `NAMESPACE=TRUE`, потеря `-DUSE_CTTK` GRTresna
из-за `cxxcppflags =` (теперь `+=`), отсутствие каталога `hdf5_*` при
рестарте, NaN при dx=4 в smoke-рестарте (теперь dx=2).

| Стадия | CCX33, 8 vCPU EPYC Milan, `-j8` | VPS, 2 vCPU, `-j2` |
|---|---|---|
| pkgs (apt) | 30 с | уже стояло |
| clone (3 репозитория) | 5 с | 20 с |
| chombo (5 библиотек) | 35 с | 7 мин |
| grchombo `ScalarFieldCosmo` | 15 с | 1.5 мин |
| grtresna `ScalarFieldCosmo` | ≈1 мин | 50 с |
| tests (сборка + запуск) | 1 мин 43 с (17 тестов) | ≈1.5 мин (3 теста) |
| smoke 1/2/3 | 1 с / 8 с / 2 с (N=32) | 2 с / 10 с / 2 с (N=16) |
| python (venv, yt 4.4.2) | 20 с | 20 с |

Бенчмарк `bench` на CCX33 (`ScalarFieldCosmo`, N=96³, один уровень, 10 шагов,
4 физических ядра × 2 SMT): 8×1 — 16.4 с, 4×2 — 16.1 с, 4×1 — 18.3 с,
2×2 — 19.4 с, 2×4 — 20.3 с. SMT даёт ~12 %, гибрид 4×2 не хуже 8×1;
на AX102-1 (16 ядер) ожидать ~0.4 с/шаг при 96³ на одном уровне.

Результат на <compute-server-ip>: `/root/grchombo`, версии Chombo `8684f2e`,
GRChombo `37e6595`, GRTresna `2e671e1`; 17/17 тестов, 3/3 smoke.
Запуск: `. /root/grchombo/env.sh && grcrun -n 8 -t 1 EXE params.txt`.
