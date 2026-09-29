#!/usr/bin/env bash
# ============================================================================
#  build_grchombo_ax102.sh — GRChombo + Chombo + GRTresna on Hetzner AX102-1
#  (AMD Ryzen 9 7950X3D 16C/32T, 128 GB DDR5 ECC, 2x1.92 TB NVMe, Ubuntu 24.04 LTS)
#
#  usage:   bash build_grchombo_ax102.sh [stage ...]          (default: all)
#  stages:  pkgs clone config chombo grchombo grtresna env tests smoke python
#           all   = the ten stages above, in that order
#           bench = MPI x OpenMP layout benchmark   (only when named explicitly)
#           tune  = CPU governor -> performance     (only when named explicitly)
#
#  environment overrides:
#    GRC_ROOT=$HOME/grchombo        install root: sources, logs, smoke, bench, venv
#    NJ=$(nproc)                    make parallelism
#    GRCHOMBO_REF=main  CHOMBO_REF=main  GRTRESNA_REF=main    branch / tag / commit
#    WITH_AH=1                      apparent-horizon finder (needs petsc-dev); PBH runs need it
#    MARCH=native  OPT=HIGH  DEBUG=FALSE
#    EXAMPLES="ScalarFieldCosmo"    GRChombo examples to build
#    TESTS="<all 3D tests>"         GRChombo unit tests to build and run
#    SMOKE_N=32  SMOKE_RANKS=4      smoke-test grid (N^3) and MPI ranks
#    BENCH_N=96  BENCH_STEPS=10  BENCH_CONFIGS="16x1 8x2 4x4 32x1"
#    PY_PKGS="numpy scipy matplotlib h5py yt"
#
#  Verified 2026-09-29 (all stages, 17/17 tests, 3/3 smoke runs) on Ubuntu 24.04 on a
#  Hetzner CCX33 (8 vCPU EPYC Milan, ~5 min) and a 2 vCPU VPS (~12 min) with GRChombo
#  37e6595 (main), Chombo fork 8684f2e (main), GRTresna 2e671e1 (main), GCC 13.3,
#  OpenMPI 4.1.6, HDF5 1.10.10, PETSc 3.19.6.  Sources of the recipe: GRChombo/.github/workflows/gcc-tests.yml and
#  GRChombo/InstallNotes/MakeDefsLocalExamples/ubuntu-gcc.Make.defs.local.
# ============================================================================
set -Eeuo pipefail

GRC_ROOT=${GRC_ROOT:-$HOME/grchombo}
NJ=${NJ:-$(nproc)}
GRCHOMBO_REF=${GRCHOMBO_REF:-main}
CHOMBO_REF=${CHOMBO_REF:-main}
GRTRESNA_REF=${GRTRESNA_REF:-main}
WITH_AH=${WITH_AH:-1}
MARCH=${MARCH:-native}
OPT=${OPT:-HIGH}
DEBUG=${DEBUG:-FALSE}
EXAMPLES=${EXAMPLES:-ScalarFieldCosmo}
TESTS=${TESTS:-}
SMOKE_N=${SMOKE_N:-32}
SMOKE_RANKS=${SMOKE_RANKS:-4}
BENCH_N=${BENCH_N:-96}
BENCH_STEPS=${BENCH_STEPS:-10}
BENCH_CONFIGS=${BENCH_CONFIGS:-"16x1 8x2 4x4 32x1"}
PY_PKGS=${PY_PKGS:-"numpy scipy matplotlib h5py yt"}

CHOMBO_HOME=$GRC_ROOT/Chombo/lib
GRCHOMBO_HOME=$GRC_ROOT/GRChombo
GRTRESNA_HOME=$GRC_ROOT/GRTresna
LOG=$GRC_ROOT/logs
export CHOMBO_HOME

if [[ $(id -u) -eq 0 ]]; then
    SUDO=""
    export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
else
    SUDO="sudo"
fi

CUR_STAGE=start
err_report() { [[ -n ${ERR_REPORTED:-} ]] && return 0; ERR_REPORTED=1
    echo "ERROR: build_grchombo_ax102.sh failed in stage \"$CUR_STAGE\" (line $1); logs in $LOG" >&2; }
trap 'err_report $LINENO' ERR
log() { printf '\n[%s] ===== %s =====\n' "$(date '+%F %T')" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
elapsed() { local s=$1; printf '%dm%02ds' $(( ($(date +%s) - s) / 60 )) $(( ($(date +%s) - s) % 60 )); }

# ---------------------------------------------------------------- pkgs
stage_pkgs() {
    log "pkgs: apt packages"
    . /etc/os-release
    [[ ${VERSION_ID:-} == 24.04 ]] || echo "WARNING: recipe tested on Ubuntu 24.04; this is ${PRETTY_NAME:-unknown}"
    local pkgs="build-essential gfortran csh perl make git pkg-config libgetopt-complete-perl
        openmpi-bin libopenmpi-dev libhdf5-dev libhdf5-openmpi-dev hdf5-tools
        libblas-dev liblapack-dev python3-venv python3-dev numactl hwloc tmux htop rsync bc"
    (( WITH_AH )) && pkgs+=" petsc-dev"
    $SUDO apt-get update -q
    # shellcheck disable=SC2086
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -q --no-install-recommends $pkgs
    mpicxx --version | head -1
    gfortran --version | head -1
}

# ---------------------------------------------------------------- clone
fetch_repo() {   # url dir ref
    local url=$1 dir=$2 ref=$3
    if [[ -d $dir/.git ]]; then
        git -C "$dir" fetch -q origin
    else
        git clone -q "$url" "$dir"
    fi
    if git -C "$dir" show-ref -q --verify "refs/remotes/origin/$ref"; then
        git -C "$dir" checkout -q "$ref" 2>/dev/null || git -C "$dir" checkout -q -b "$ref" "origin/$ref"
        git -C "$dir" merge -q --ff-only "origin/$ref" \
            || echo "WARNING: $dir has local commits on '$ref'; left as is (not fast-forwarded)"
    else
        git -C "$dir" checkout -q "$ref"
    fi
    printf '%-9s %s  %s  (%s)\n' "$(basename "$dir")" "$(git -C "$dir" rev-parse --short=12 HEAD)" \
        "$(git -C "$dir" log -1 --format=%cs)" "$ref" | tee -a "$GRC_ROOT/VERSIONS.txt"
}

stage_clone() {
    log "clone: Chombo (GRChombo fork), GRChombo, GRTresna -> $GRC_ROOT"
    mkdir -p "$GRC_ROOT" "$LOG"
    echo "# $(date -Is)" >> "$GRC_ROOT/VERSIONS.txt"
    fetch_repo https://github.com/GRChombo/Chombo.git            "$GRC_ROOT/Chombo"   "$CHOMBO_REF"
    fetch_repo https://github.com/GRChombo/GRChombo.git          "$GRCHOMBO_HOME"     "$GRCHOMBO_REF"
    fetch_repo https://github.com/GRTLCollaboration/GRTresna.git "$GRTRESNA_HOME"     "$GRTRESNA_REF"
}

# ---------------------------------------------------------------- config
stage_config() {
    log "config: $CHOMBO_HOME/mk/Make.defs.local"
    [[ -d $CHOMBO_HOME/mk ]] || die "Chombo not cloned (run the clone stage first)"
    local hdf_mpi_inc hdf_mpi_lib hdf_ser_inc hdf_ser_lib petsc_cpp="" petsc_lib=""
    # Ubuntu's hdf5-*.pc files carry the paths only in Cflags/Libs (no includedir variable)
    pc_first() { pkg-config "$1" "$2" 2>/dev/null | tr ' ' '\n' | grep -m1 "^$3" | sed "s/^$3//"; }
    hdf_mpi_inc=$(pc_first --cflags-only-I hdf5-openmpi -I); hdf_mpi_inc=${hdf_mpi_inc:-/usr/include/hdf5/openmpi}
    hdf_mpi_lib=$(pc_first --libs-only-L   hdf5-openmpi -L); hdf_mpi_lib=${hdf_mpi_lib:-/usr/lib/x86_64-linux-gnu/hdf5/openmpi}
    hdf_ser_inc=$(pc_first --cflags-only-I hdf5-serial  -I); hdf_ser_inc=${hdf_ser_inc:-/usr/include/hdf5/serial}
    hdf_ser_lib=$(pc_first --libs-only-L   hdf5-serial  -L); hdf_ser_lib=${hdf_ser_lib:-/usr/lib/x86_64-linux-gnu/hdf5/serial}
    [[ -f $hdf_mpi_inc/hdf5.h ]] || die "parallel HDF5 headers not found in $hdf_mpi_inc (apt: libhdf5-openmpi-dev)"
    if (( WITH_AH )); then
        pkg-config --exists petsc || die "PETSc not found by pkg-config (apt: petsc-dev) — or set WITH_AH=0"
        petsc_cpp="-I$(pkg-config --variable=includedir petsc) -DUSE_AHFINDER"
        petsc_lib="$(pkg-config --libs petsc) -Wl,-rpath,$(pkg-config --variable=libdir petsc)"
    fi
    local cpu; cpu=$(lscpu | sed -n 's/^Model name: *//p' | head -1)
    cat > "$CHOMBO_HOME/mk/Make.defs.local" <<EOF
# Generated by build_grchombo_ax102.sh on $(date -Is), host $(hostname), CPU: $cpu
# Based on GRChombo/InstallNotes/MakeDefsLocalExamples/ubuntu-gcc.Make.defs.local
DIM            = 3
DEBUG          = $DEBUG
OPT            = $OPT
PRECISION      = DOUBLE
PROFILE        = FALSE
CXX            = g++
FC             = gfortran
MPI            = TRUE
OPENMPCC       = TRUE
MPICXX         = mpicxx
USE_64         = TRUE
USE_HDF        = TRUE
HDFINCFLAGS    = -I$hdf_ser_inc
HDFLIBFLAGS    = -L$hdf_ser_lib -lhdf5 -lz
HDFMPIINCFLAGS = -I$hdf_mpi_inc
HDFMPILIBFLAGS = -L$hdf_mpi_lib -lhdf5 -lz
NAMESPACE      = TRUE
USE_MT         = FALSE
# Chombo's GNU compiler file adds -std=c++14 -fopenmp itself (cxxcomflags); only the
# optimisation flags are set here.  On the 7950X3D (Zen 4) -march=native = znver4 with
# GCC >= 13, which selects the AVX-512 branch of GRChombo's simd layer.
cxxoptflags    = -march=$MARCH -O3
foptflags      = -march=$MARCH -O3
# apparent-horizon finder (PETSc); empty if WITH_AH=0.  '+=' keeps flags that an example's
# GNUmakefile sets before including Make.example (GRTresna: cxxcppflags = -DUSE_CTTK).
cxxcppflags   += $petsc_cpp
syslibflags   += -lblas -llapack $petsc_lib
EOF
    cat "$CHOMBO_HOME/mk/Make.defs.local"
}

# ---------------------------------------------------------------- chombo
stage_chombo() {
    log "chombo: libraries BaseTools BoxTools AMRTools AMRElliptic AMRTimeDependent (-j$NJ)"
    mkdir -p "$LOG"; local t0; t0=$(date +%s)
    make -j"$NJ" -C "$CHOMBO_HOME" BaseTools BoxTools AMRTools AMRElliptic AMRTimeDependent 2>&1 | tee "$LOG/chombo.log"
    # with NAMESPACE=TRUE the libraries are named lib<name>3d_ch.<config>.a
    ls "$CHOMBO_HOME"/libamrtimedependent3d*.a "$CHOMBO_HOME"/libamrelliptic3d*.a >/dev/null \
        || die "Chombo libraries missing after build (see $LOG/chombo.log)"
    echo "chombo libraries built in $(elapsed "$t0"):"; ls "$CHOMBO_HOME"/lib*3d*.a
}

# ---------------------------------------------------------------- grchombo
stage_grchombo() {
    mkdir -p "$LOG"
    local ex t0
    for ex in $EXAMPLES; do
        log "grchombo: Examples/$ex (-j$NJ)"; t0=$(date +%s)
        [[ -d $GRCHOMBO_HOME/Examples/$ex ]] || die "no such example: $ex"
        make -j"$NJ" -C "$GRCHOMBO_HOME/Examples/$ex" all 2>&1 | tee "$LOG/grchombo_$ex.log"
        ls "$GRCHOMBO_HOME/Examples/$ex"/Main_*3d*.ex >/dev/null || die "executable missing for $ex"
        echo "$ex built in $(elapsed "$t0"): $(ls "$GRCHOMBO_HOME/Examples/$ex"/Main_*3d*.ex)"
    done
}

# ---------------------------------------------------------------- grtresna
stage_grtresna() {
    log "grtresna: Examples/ScalarFieldCosmo (initial-data solver, -j$NJ)"
    mkdir -p "$LOG"; local t0; t0=$(date +%s)
    make -j"$NJ" -C "$GRTRESNA_HOME/Examples/ScalarFieldCosmo" all 2>&1 | tee "$LOG/grtresna.log"
    ls "$GRTRESNA_HOME/Examples/ScalarFieldCosmo"/Main_*3d*.ex >/dev/null || die "GRTresna executable missing"
    echo "GRTresna built in $(elapsed "$t0"): $(ls "$GRTRESNA_HOME/Examples/ScalarFieldCosmo"/Main_*3d*.ex)"
}

# ---------------------------------------------------------------- env
stage_env() {
    log "env: $GRC_ROOT/env.sh and $GRC_ROOT/bin/grcrun"
    mkdir -p "$GRC_ROOT/bin"
    {
        echo "# source this file: . $GRC_ROOT/env.sh"
        echo "export GRC_ROOT=$GRC_ROOT"
        cat <<'EOF'
export CHOMBO_HOME=$GRC_ROOT/Chombo/lib
export GRCHOMBO_HOME=$GRC_ROOT/GRChombo
export GRTRESNA_HOME=$GRC_ROOT/GRTresna
export PATH=$GRC_ROOT/bin:$PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_PROC_BIND=true OMP_PLACES=cores
ulimit -s unlimited 2>/dev/null || true
if [ "$(id -u)" = 0 ]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi
[ -f "$GRC_ROOT/venv/bin/activate" ] && . "$GRC_ROOT/venv/bin/activate"
EOF
    } > "$GRC_ROOT/env.sh"
    cat > "$GRC_ROOT/bin/grcrun" <<'EOF'
#!/usr/bin/env bash
# grcrun — run a GRChombo / GRTresna executable on one node: RANKS MPI ranks x THREADS OpenMP threads
# usage: grcrun [-n RANKS] [-t THREADS] [-o LOGFILE] [-v] EXE PARAMS [extra mpirun args ...]
# defaults: RANKS=16 THREADS=1 (all physical cores of the AX102-1, one rank per core).
# If RANKS*THREADS exceeds the physical cores, SMT hardware threads are used (e.g. -n 32).
set -Eeuo pipefail
R=${GRC_RANKS:-16}; T=${GRC_THREADS:-1}; LOGF=""; VERB=""
while getopts "n:t:o:v" o; do
    case $o in n) R=$OPTARG;; t) T=$OPTARG;; o) LOGF=$OPTARG;; v) VERB="--report-bindings";; *) exit 2;; esac
done
shift $((OPTIND-1))
[[ $# -ge 2 ]] || { echo "usage: grcrun [-n RANKS] [-t THREADS] [-o LOG] [-v] EXE PARAMS [mpirun args]" >&2; exit 2; }
EXE=$(readlink -f "$1"); PAR=$2; shift 2
NPHYS=$(lscpu -p=core 2>/dev/null | grep -v '^#' | sort -u | wc -l); NPHYS=${NPHYS:-$(nproc)}
extra=()
if (( R * T > NPHYS )); then     # SMT threads as processing elements
    extra+=(--use-hwthread-cpus)
    if (( T == 1 )); then extra+=(--map-by hwthread --bind-to hwthread)
    else extra+=(--map-by "slot:PE=$T" --bind-to hwthread); fi
elif (( T == 1 )); then
    extra+=(--map-by core --bind-to core)
else
    extra+=(--map-by "slot:PE=$T" --bind-to core)
fi
[[ $(id -u) -eq 0 ]] && extra+=(--allow-run-as-root)
[[ -n $VERB ]] && extra+=("$VERB")
export OMP_NUM_THREADS=$T OMP_PROC_BIND=true OMP_PLACES=cores
cmd=(mpirun -n "$R" "${extra[@]}" -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES "$@" "$EXE" "$PAR")
echo "[grcrun] $(date '+%F %T')  ranks=$R threads=$T  cwd=$PWD"
echo "[grcrun] ${cmd[*]}"
if [[ -n $LOGF ]]; then "${cmd[@]}" 2>&1 | tee "$LOGF"; else "${cmd[@]}"; fi
EOF
    chmod +x "$GRC_ROOT/bin/grcrun"
    echo "written: $GRC_ROOT/env.sh, $GRC_ROOT/bin/grcrun"
}

# ---------------------------------------------------------------- tests
stage_tests() {
    local tests=$TESTS t rc pass=0 fail=0 failed=""
    if [[ -z $tests ]]; then   # all 3D tests (the 2D AH test needs a DIM=2 build)
        tests=$(cd "$GRCHOMBO_HOME/Tests" && ls -d */ | tr -d / | grep -v 2D | tr '\n' ' ')
    fi
    log "tests: $tests"
    mkdir -p "$LOG"
    for t in $tests; do
        printf '%-32s build ... ' "$t"
        if make -j"$NJ" -C "$GRCHOMBO_HOME/Tests/$t" all >"$LOG/test_$t.build.log" 2>&1; then
            printf 'run ... '
            if make -C "$GRCHOMBO_HOME/Tests/$t" run >"$LOG/test_$t.run.log" 2>&1; then
                echo PASS; pass=$((pass+1))
            else
                echo "FAIL (see $LOG/test_$t.run.log)"; fail=$((fail+1)); failed+=" $t"
            fi
        else
            echo "BUILD FAILED (see $LOG/test_$t.build.log)"; fail=$((fail+1)); failed+=" $t"
        fi
    done
    echo "tests: $pass passed, $fail failed${failed:+ ($failed)}"
    (( fail == 0 )) || die "some GRChombo tests failed"
}

# ---------------------------------------------------------------- smoke
smoke_params_grchombo() {   # src dst N box  -> writes dst (short run: 8 steps, plot every 4)
    local src=$1 dst=$2 n=$3 box=$4
    sed -e "s/^N_full *=.*/N_full = $n/" \
        -e "s/^max_box_size *=.*/max_box_size = $box/" \
        -e "s/^min_box_size *=.*/min_box_size = $box/" \
        -e "s/^checkpoint_interval *=.*/checkpoint_interval = 8/" \
        -e "s/^plot_interval *=.*/plot_interval = 4/" \
        -e "s/^verbosity *=.*/verbosity = 1/" "$src" > "$dst"
    echo "max_steps = 8" >> "$dst"
}

stage_smoke() {
    [[ -x $GRC_ROOT/bin/grcrun ]] || stage_env
    local n=$SMOKE_N box=$(( SMOKE_N >= 32 ? 16 : 8 )) run=$GRC_ROOT/bin/grcrun t0
    local exe texe d
    exe=$(ls "$GRCHOMBO_HOME/Examples/ScalarFieldCosmo"/Main_ScalarFieldCosmo3d*.ex | head -1)
    texe=$(ls "$GRTRESNA_HOME/Examples/ScalarFieldCosmo"/Main_ScalarFieldCosmo3d*.ex | head -1)
    rm -rf "$GRC_ROOT/smoke"; mkdir -p "$GRC_ROOT/smoke"

    log "smoke 1/3: GRChombo ScalarFieldCosmo, analytic initial data, N=$n, $SMOKE_RANKS ranks, 8 steps"
    d=$GRC_ROOT/smoke/cosmo_analytic; mkdir -p "$d"; t0=$(date +%s)
    smoke_params_grchombo "$GRCHOMBO_HOME/Examples/ScalarFieldCosmo/params.txt" "$d/params.txt" "$n" "$box"
    (cd "$d" && "$run" -n "$SMOKE_RANKS" -t 1 -o run.log "$exe" params.txt >/dev/null)
    [[ -f $d/hdf5/Cosmop_000008.3d.hdf5 ]] || die "smoke 1: plot file not written (see $d/run.log)"
    grep -qi nan "$d/pout/pout.0" && die "smoke 1: NaN in pout"
    echo "smoke 1 OK in $(elapsed "$t0"): $(ls "$d/hdf5" | tr '\n' ' ')"

    # the GRTresna example is set up for N=64, L=128 (dx = 2); keep dx = 2 at any N (dx = 4 gives NaNs at step 1)
    local L=$(( 2 * n ))
    log "smoke 2/3: GRTresna constraint solver (CTTK), N=$n, L=$L, $SMOKE_RANKS ranks"
    d=$GRC_ROOT/smoke/cosmo_grtresna; mkdir -p "$d"; t0=$(date +%s)
    sed -e "s/^N *=.*/N = $n $n $n/" -e "s/^L *=.*/L = $L/" \
        -e "s/^block_factor *=.*/block_factor = $box/" \
        -e "s/^max_grid_size *=.*/max_grid_size = $box/" \
        -e "s#^output_path *=.*#output_path = $d/solver_out/#" \
        "$GRTRESNA_HOME/Examples/ScalarFieldCosmo/params.txt" > "$d/params_solver.txt"
    (cd "$d" && "$run" -n "$SMOKE_RANKS" -t 1 -o solver.log "$texe" params_solver.txt >/dev/null)
    [[ -f $d/solver_out/InitialDataFinal.3d.hdf5 ]] || die "smoke 2: solver output missing (see $d/solver.log)"
    echo "smoke 2 OK in $(elapsed "$t0"): $(ls "$d/solver_out")"

    log "smoke 3/3: GRChombo restart from the GRTresna data, 8 steps"
    t0=$(date +%s)
    smoke_params_grchombo "$GRCHOMBO_HOME/Examples/ScalarFieldCosmo/GRTresna_restart_setup/params_LR.txt" \
        "$d/params_evolve.txt" "$n" "$box"
    sed -i -e "s#^restart_file *=.*#restart_file = $d/solver_out/InitialDataFinal.3d.hdf5#" \
           -e "s/^L_full *=.*/L_full = $L/" "$d/params_evolve.txt"
    # on the restart code path GRChombo does not create the hdf5/pout/data sub-directories itself
    mkdir -p "$d/hdf5_LR" "$d/pout_LR" "$d/data_LR"
    (cd "$d" && "$run" -n "$SMOKE_RANKS" -t 1 -o evolve.log "$exe" params_evolve.txt >/dev/null)
    [[ -f $d/hdf5_LR/Cosmop_000008.3d.hdf5 ]] || die "smoke 3: plot file not written (see $d/evolve.log)"
    grep -qi nan "$d/pout_LR/pout.0" && die "smoke 3: NaN in pout"
    echo "smoke 3 OK in $(elapsed "$t0"): $(ls "$d/hdf5_LR" | tr '\n' ' ')"
    echo "smoke: all three runs passed"
}

# ---------------------------------------------------------------- bench
stage_bench() {
    [[ -x $GRC_ROOT/bin/grcrun ]] || stage_env
    local exe cfg r t d t0 dt run=$GRC_ROOT/bin/grcrun
    exe=$(ls "$GRCHOMBO_HOME/Examples/ScalarFieldCosmo"/Main_ScalarFieldCosmo3d*.ex | head -1)
    log "bench: ScalarFieldCosmo N=$BENCH_N, $BENCH_STEPS steps, configs: $BENCH_CONFIGS"
    mkdir -p "$GRC_ROOT/bench"
    { echo "# $(date -Is)  N=$BENCH_N steps=$BENCH_STEPS  $(lscpu | sed -n 's/^Model name: *//p')"; echo "# ranks x threads   wall[s]"; } > "$GRC_ROOT/bench/summary.txt"
    for cfg in $BENCH_CONFIGS; do
        r=${cfg%x*}; t=${cfg#*x}; d=$GRC_ROOT/bench/$cfg; rm -rf "$d"; mkdir -p "$d"
        sed -e "s/^N_full *=.*/N_full = $BENCH_N/" -e "s/^checkpoint_interval *=.*/checkpoint_interval = 1000000/" \
            -e "s/^plot_interval *=.*/plot_interval = -1/" \
            "$GRCHOMBO_HOME/Examples/ScalarFieldCosmo/params.txt" > "$d/params.txt"
        echo "max_steps = $BENCH_STEPS" >> "$d/params.txt"
        t0=$(date +%s.%N)
        (cd "$d" && "$run" -n "$r" -t "$t" -o run.log "$exe" params.txt >/dev/null) || { echo "$cfg FAILED"; continue; }
        dt=$(echo "$(date +%s.%N) - $t0" | bc)
        printf '%-8s %8.1f\n' "$cfg" "$dt" | tee -a "$GRC_ROOT/bench/summary.txt"
    done
    echo "bench summary: $GRC_ROOT/bench/summary.txt"
}

# ---------------------------------------------------------------- python
stage_python() {
    log "python: venv $GRC_ROOT/venv ($PY_PKGS)"
    [[ -d $GRC_ROOT/venv ]] || python3 -m venv "$GRC_ROOT/venv"
    "$GRC_ROOT/venv/bin/pip" install -q -U pip
    # shellcheck disable=SC2086
    "$GRC_ROOT/venv/bin/pip" install -q $PY_PKGS
    "$GRC_ROOT/venv/bin/python" -c "import h5py, numpy; print('h5py', h5py.__version__, 'numpy', numpy.__version__)"
}

# ---------------------------------------------------------------- tune (optional)
stage_tune() {
    log "tune: CPU frequency governor -> performance"
    DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -q --no-install-recommends linux-tools-common "linux-tools-$(uname -r)" || true
    $SUDO cpupower frequency-set -g performance || echo "cpupower failed (amd-pstate in passive mode?) — not critical"
    grep -h . /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true
}

# ---------------------------------------------------------------- main
main() {
    local stages=("$@") s T0
    (( ${#stages[@]} )) || stages=(all)
    [[ ${stages[*]} == *all* ]] && stages=(pkgs clone config chombo grchombo grtresna env tests smoke python)
    T0=$(date +%s)
    echo "build_grchombo_ax102.sh: GRC_ROOT=$GRC_ROOT NJ=$NJ WITH_AH=$WITH_AH MARCH=$MARCH OPT=$OPT DEBUG=$DEBUG"
    echo "stages: ${stages[*]}"
    for s in "${stages[@]}"; do
        case $s in
            pkgs|clone|config|chombo|grchombo|grtresna|env|tests|smoke|bench|python|tune) CUR_STAGE=$s; "stage_$s" ;;
            *) die "unknown stage '$s'" ;;
        esac
    done
    log "done in $(elapsed "$T0")"
    [[ -f $GRC_ROOT/VERSIONS.txt ]] && { echo "versions:"; tail -3 "$GRC_ROOT/VERSIONS.txt"; }
    echo "next: . $GRC_ROOT/env.sh ; grcrun -n 16 -t 1 <Main_*.ex> params.txt"
}

main "$@"
