#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run the maintained RGK 6.535 GeV EPPI0 pipeline in an isolated work directory.

Required:
  --work-dir PATH          New directory for this run (must not already exist)
  --data PATH              RGK 6.535 GeV data HIPO file or directory
  --mc PATH                RGK 6.535 GeV acceptance GEMC HIPO file or directory
  --born-lund PATH         Born/non-radiative LUND file or directory
  --radiative-lund PATH    Radiative LUND file or directory
  --born-norm PATH         Born .norm/.sum file or directory of sidecars
  --radiative-norm PATH    Radiative .norm/.sum file or directory of sidecars
  --aao-xsec PATH          aao_xsec executable used for bin centering

Options:
  --workers N              Bin-centering worker processes (default: 8)
  --bin-centering-N N      Midpoint samples per dimension (default: 4)
  --max-files N            Limit data and GEMC inputs; 0 means all (default: 0)
  --max-lund-files N       Limit each LUND/normalization sample (default: 0/all)
  --progress-events N      hipo2root/post_process progress interval (default: 1000000)
  --resume                 Reuse the work directory and skip completed stages
  --dry-run                Print commands without running them
  -h, --help               Show this help

Load the JLab clas12 and qadb modules before running this script. No existing
results are moved, replaced, or deleted.
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

WORK_DIR=
DATA_INPUT=
MC_INPUT=
BORN_LUND=
RADIATIVE_LUND=
BORN_NORM=
RADIATIVE_NORM=
AAO_XSEC=
WORKERS=8
BIN_CENTERING_N=4
MAX_FILES=0
MAX_LUND_FILES=0
PROGRESS_EVENTS=1000000
RESUME=0
DRY_RUN=0

while (($#)); do
    case "$1" in
        --work-dir) WORK_DIR=${2:?missing value for --work-dir}; shift 2 ;;
        --data) DATA_INPUT=${2:?missing value for --data}; shift 2 ;;
        --mc) MC_INPUT=${2:?missing value for --mc}; shift 2 ;;
        --born-lund) BORN_LUND=${2:?missing value for --born-lund}; shift 2 ;;
        --radiative-lund) RADIATIVE_LUND=${2:?missing value for --radiative-lund}; shift 2 ;;
        --born-norm) BORN_NORM=${2:?missing value for --born-norm}; shift 2 ;;
        --radiative-norm) RADIATIVE_NORM=${2:?missing value for --radiative-norm}; shift 2 ;;
        --aao-xsec) AAO_XSEC=${2:?missing value for --aao-xsec}; shift 2 ;;
        --workers) WORKERS=${2:?missing value for --workers}; shift 2 ;;
        --bin-centering-N) BIN_CENTERING_N=${2:?missing value for --bin-centering-N}; shift 2 ;;
        --max-files) MAX_FILES=${2:?missing value for --max-files}; shift 2 ;;
        --max-lund-files) MAX_LUND_FILES=${2:?missing value for --max-lund-files}; shift 2 ;;
        --progress-events) PROGRESS_EVENTS=${2:?missing value for --progress-events}; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for required_name in WORK_DIR DATA_INPUT MC_INPUT BORN_LUND RADIATIVE_LUND BORN_NORM RADIATIVE_NORM AAO_XSEC; do
    if [[ -z ${!required_name} ]]; then
        printf 'Missing required option for %s\n\n' "$required_name" >&2
        usage >&2
        exit 2
    fi
done

for integer_value in "$WORKERS" "$BIN_CENTERING_N" "$MAX_FILES" "$MAX_LUND_FILES" "$PROGRESS_EVENTS"; do
    if [[ ! $integer_value =~ ^[0-9]+$ ]]; then
        printf 'Expected a non-negative integer, got: %s\n' "$integer_value" >&2
        exit 2
    fi
done
if ((WORKERS == 0 || BIN_CENTERING_N == 0)); then
    printf '%s\n' '--workers and --bin-centering-N must be positive' >&2
    exit 2
fi

for path_value in "$WORK_DIR" "$DATA_INPUT" "$MC_INPUT" "$BORN_LUND" "$RADIATIVE_LUND" "$BORN_NORM" "$RADIATIVE_NORM" "$AAO_XSEC"; do
    if [[ $path_value != /* ]]; then
        printf 'Use absolute paths for all run inputs and outputs: %s\n' "$path_value" >&2
        exit 2
    fi
done

if [[ -e $WORK_DIR && $RESUME -eq 0 ]]; then
    printf 'Refusing to use existing work directory: %s\n' "$WORK_DIR" >&2
    printf 'Choose a new path, or pass --resume for this isolated run.\n' >&2
    exit 2
fi
if [[ -e $WORK_DIR && $RESUME -eq 1 && ! -f $WORK_DIR/provenance/run.txt ]]; then
    printf 'Refusing to resume a directory without this runner provenance: %s\n' "$WORK_DIR" >&2
    exit 2
fi

if ((DRY_RUN == 0)); then
    for input_path in "$DATA_INPUT" "$MC_INPUT" "$BORN_LUND" "$RADIATIVE_LUND" "$BORN_NORM" "$RADIATIVE_NORM"; do
        if [[ ! -e $input_path ]]; then
            printf 'Input does not exist: %s\n' "$input_path" >&2
            exit 2
        fi
    done
    if [[ ! -x $AAO_XSEC ]]; then
        printf 'aao_xsec is not executable: %s\n' "$AAO_XSEC" >&2
        exit 2
    fi
fi

BUILD_DIR="$WORK_DIR/build"
ROOT_DIR="$WORK_DIR/root"
RESULTS_DIR="$WORK_DIR/results"
LOG_DIR="$WORK_DIR/logs"
STATE_DIR="$WORK_DIR/.stages"
PROVENANCE_DIR="$WORK_DIR/provenance"

quote_command() {
    printf '%q ' "$@"
    printf '\n'
}

run_command() {
    local log_file=$1
    shift
    printf '+ '
    quote_command "$@"
    if ((DRY_RUN == 0)); then
        "$@" 2>&1 | tee -a "$log_file"
    fi
}

run_in_root_dir() {
    local log_file=$1
    shift
    printf '+ (cd %q && ' "$ROOT_DIR"
    quote_command "$@"
    if ((DRY_RUN == 0)); then
        (cd "$ROOT_DIR" && "$@") 2>&1 | tee -a "$log_file"
    fi
}

run_stage() {
    local stage=$1
    shift
    local marker="$STATE_DIR/$stage.done"
    local log_file="$LOG_DIR/$stage.log"
    if [[ -f $marker && $RESUME -eq 1 ]]; then
        printf 'SKIP completed stage: %s\n' "$stage"
        return
    fi
    printf '\n== %s ==\n' "$stage"
    "$@" "$log_file"
    if ((DRY_RUN == 0)); then
        date -u +'%Y-%m-%dT%H:%M:%SZ' > "$marker"
    fi
}

stage_build() {
    local log_file=$1
    run_command "$log_file" cmake -S "$REPO_ROOT" -B "$BUILD_DIR"
    run_command "$log_file" cmake --build "$BUILD_DIR" -j "$WORKERS"
}

stage_convert_data() {
    local log_file=$1
    run_in_root_dir "$log_file" "$BUILD_DIR/hipo2root" \
        "$REPO_ROOT/configs/processing/rgk/6.535/eppi0_data.json" \
        "$DATA_INPUT" "$MAX_FILES" "$PROGRESS_EVENTS"
}

stage_select_data() {
    local log_file=$1
    run_in_root_dir "$log_file" "$BUILD_DIR/post_process" \
        "$REPO_ROOT/configs/post/rgk/6.535/eppi0_data.json" \
        "$ROOT_DIR/6.535_rgk_eppi0_data.root" "$PROGRESS_EVENTS"
}

stage_export_data() {
    local log_file=$1
    run_command "$log_file" python3 "$REPO_ROOT/analysis/export_selected_data.py" \
        "$ROOT_DIR/6.535_rgk_eppi0_data_selected.root" \
        "$ROOT_DIR/6.535_rgk_eppi0_data.root" \
        "$RESULTS_DIR/data_events.npz" \
        --dictionary "$BUILD_DIR/libROOTBranchesDict.so"
}

stage_data_exclusivity() {
    local log_file=$1
    run_command "$log_file" python3 "$REPO_ROOT/analysis/derive_exclusivity.py" \
        "$RESULTS_DIR/data_events.npz" \
        --config "$REPO_ROOT/configs/analysis/rgk/6.535.json" \
        --cuts "$RESULTS_DIR/data_exclusivity.npz" \
        --mask "$RESULTS_DIR/data_exclusivity.npy"
}

stage_convert_mc() {
    local log_file=$1
    run_in_root_dir "$log_file" "$BUILD_DIR/hipo2root" \
        "$REPO_ROOT/configs/processing/rgk/6.535/eppi0_mc_acceptance.json" \
        "$MC_INPUT" "$MAX_FILES" "$PROGRESS_EVENTS"
}

stage_select_mc() {
    local log_file=$1
    run_in_root_dir "$log_file" "$BUILD_DIR/post_process" \
        "$REPO_ROOT/configs/post/rgk/6.535/eppi0_mc_acceptance.json" \
        "$ROOT_DIR/6.535_rgk_eppi0_mc_acceptance.root" "$PROGRESS_EVENTS"
}

stage_check_mc_keys() {
    local log_file=$1
    run_command "$log_file" python3 "$REPO_ROOT/analysis/check_event_keys.py" \
        "$ROOT_DIR/6.535_rgk_eppi0_mc_acceptance.root"
}

stage_mc_exclusivity() {
    local log_file=$1
    run_command "$log_file" python3 "$REPO_ROOT/analysis/derive_exclusivity.py" \
        "$ROOT_DIR/6.535_rgk_eppi0_mc_acceptance_selected.root" \
        --format selected-root \
        --dictionary "$BUILD_DIR/libROOTBranchesDict.so" \
        --config "$REPO_ROOT/configs/analysis/rgk/6.535.json" \
        --cuts "$RESULTS_DIR/gemc_exclusivity.npz" \
        --mask "$RESULTS_DIR/gemc_selected_exclusivity.npy"
}

stage_response() {
    local log_file=$1
    run_command "$log_file" python3 "$REPO_ROOT/analysis/run_analysis.py" response-root \
        "$ROOT_DIR/6.535_rgk_eppi0_mc_acceptance.root" \
        "$ROOT_DIR/6.535_rgk_eppi0_mc_acceptance_selected.root" \
        --config "$REPO_ROOT/configs/analysis/rgk/6.535.json" \
        --output-dir "$RESULTS_DIR/response" \
        --dictionary "$BUILD_DIR/libROOTBranchesDict.so" \
        --selection-mask "$RESULTS_DIR/gemc_selected_exclusivity.npy"
}

stage_radiative_correction() {
    local log_file=$1
    local limit_args=()
    if ((MAX_LUND_FILES > 0)); then
        limit_args+=(--max-files "$MAX_LUND_FILES" --max-normalization-files "$MAX_LUND_FILES")
    fi
    run_command "$log_file" python3 "$REPO_ROOT/analysis/run_analysis.py" radiative-correction \
        "$BORN_LUND" "$RADIATIVE_LUND" \
        --config "$REPO_ROOT/configs/analysis/rgk/6.535.json" \
        --output "$RESULTS_DIR/C_rad.npz" \
        --born-normalization-file "$BORN_NORM" \
        --radiative-normalization-file "$RADIATIVE_NORM" \
        "${limit_args[@]}" \
        --progress-chunks 1 \
        --diagnostic-pdf "$RESULTS_DIR/C_rad_diagnostics.pdf" \
        --diagnostic-csv "$RESULTS_DIR/C_rad_diagnostics.csv" \
        --diagnostic-quilt
}

stage_unfold() {
    local log_file=$1
    run_command "$log_file" python3 "$REPO_ROOT/analysis/run_analysis.py" unfold \
        "$RESULTS_DIR/data_events.npz" \
        "$RESULTS_DIR/response/response_matrix.npz" \
        "$RESULTS_DIR/response/response_meta.npz" \
        --config "$REPO_ROOT/configs/analysis/rgk/6.535.json" \
        --output "$RESULTS_DIR/unfolding.npz" \
        --selection-mask "$RESULTS_DIR/data_exclusivity.npy" \
        --radiative-correction "$RESULTS_DIR/C_rad.npz"
}

stage_bin_centering() {
    local log_file=$1
    run_command "$log_file" python3 "$REPO_ROOT/analysis/run_analysis.py" bin-centering \
        --config "$REPO_ROOT/configs/analysis/rgk/6.535.json" \
        --output "$RESULTS_DIR/C_BC.npz" \
        --exe "$AAO_XSEC" \
        --N "$BIN_CENTERING_N" \
        --workers "$WORKERS" \
        --progress-chunks 10
}

stage_cross_section() {
    local log_file=$1
    run_command "$log_file" python3 "$REPO_ROOT/analysis/run_analysis.py" cross-section \
        "$RESULTS_DIR/unfolding.npz" \
        --config "$REPO_ROOT/configs/analysis/rgk/6.535.json" \
        --output "$RESULTS_DIR/cross_section.npz" \
        --bin-centering "$RESULTS_DIR/C_BC.npz"
}

stage_harmonics() {
    local log_file=$1
    run_command "$log_file" python3 "$REPO_ROOT/analysis/run_analysis.py" fit-harmonics \
        "$RESULTS_DIR/cross_section.npz" \
        --output "$RESULTS_DIR/harmonics.npz"
}

stage_plots() {
    local log_file=$1
    run_command "$log_file" python3 "$REPO_ROOT/analysis/run_analysis.py" acceptance-plots \
        "$RESULTS_DIR/response/response_meta.npz" \
        --response-matrix "$RESULTS_DIR/response/response_matrix.npz" \
        --output-dir "$RESULTS_DIR/plots/acceptance" --quilt
    run_command "$log_file" python3 "$REPO_ROOT/analysis/run_analysis.py" response-plots \
        "$RESULTS_DIR/response/response_matrix.npz" \
        "$RESULTS_DIR/response/response_meta.npz" \
        --output "$RESULTS_DIR/plots/response_diagnostics.pdf"
    run_command "$log_file" python3 "$REPO_ROOT/analysis/run_analysis.py" cross-section-plots \
        "$RESULTS_DIR/cross_section.npz" "$RESULTS_DIR/harmonics.npz" \
        --output-dir "$RESULTS_DIR/plots/cross_section"
    run_command "$log_file" python3 "$REPO_ROOT/analysis/run_analysis.py" harmonic-plots \
        "$RESULTS_DIR/harmonics.npz" \
        --output-dir "$RESULTS_DIR/plots/harmonics" --quilt
}

if ((DRY_RUN == 0)); then
    mkdir -p "$BUILD_DIR" "$ROOT_DIR" "$RESULTS_DIR" "$LOG_DIR" "$STATE_DIR" "$PROVENANCE_DIR"
    provenance_lines=(
        "work_dir=$WORK_DIR"
        "data_input=$DATA_INPUT"
        "mc_input=$MC_INPUT"
        "born_lund=$BORN_LUND"
        "radiative_lund=$RADIATIVE_LUND"
        "born_norm=$BORN_NORM"
        "radiative_norm=$RADIATIVE_NORM"
        "aao_xsec=$AAO_XSEC"
        "bin_centering_N=$BIN_CENTERING_N"
        "workers=$WORKERS"
        "max_files=$MAX_FILES"
        "max_lund_files=$MAX_LUND_FILES"
    )
    if [[ -f $PROVENANCE_DIR/run.txt ]]; then
        for provenance_line in "${provenance_lines[@]}"; do
            if ! grep -Fqx -- "$provenance_line" "$PROVENANCE_DIR/run.txt"; then
                printf 'Resume settings do not match the original run: %s\n' "$provenance_line" >&2
                exit 2
            fi
        done
    else
        {
            printf 'started_utc=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
            printf 'repository=%s\n' "$REPO_ROOT"
            printf 'git_commit=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
            printf '%s\n' "${provenance_lines[@]}"
        } > "$PROVENANCE_DIR/run.txt"
        git -C "$REPO_ROOT" status --short > "$PROVENANCE_DIR/git-status.txt"
        git -C "$REPO_ROOT" diff > "$PROVENANCE_DIR/git-diff.patch"
        cp "$REPO_ROOT/configs/analysis/rgk/6.535.json" "$PROVENANCE_DIR/analysis-config.json"
        cp "$REPO_ROOT/configs/processing/rgk/6.535/eppi0_data.json" \
            "$PROVENANCE_DIR/processing-data-config.json"
        cp "$REPO_ROOT/configs/processing/rgk/6.535/eppi0_mc_acceptance.json" \
            "$PROVENANCE_DIR/processing-mc-config.json"
        cp "$REPO_ROOT/configs/post/rgk/6.535/eppi0_data.json" \
            "$PROVENANCE_DIR/post-data-config.json"
        cp "$REPO_ROOT/configs/post/rgk/6.535/eppi0_mc_acceptance.json" \
            "$PROVENANCE_DIR/post-mc-config.json"
        cp "$REPO_ROOT/parameters/proton_energy_loss/6.535RGK_INCLUSIVE_GEMC_100M.json" \
            "$PROVENANCE_DIR/proton-energy-loss.json"
        cp "$REPO_ROOT/parameters/sampling_fraction/SF_sigma_cut_params_6.535RGKSKIM1.json" \
            "$PROVENANCE_DIR/sampling-fraction-data.json"
        cp "$REPO_ROOT/parameters/sampling_fraction/SF_sigma_cut_params_6.535RGK_INCLUSIVE_GEMC_100M.json" \
            "$PROVENANCE_DIR/sampling-fraction-mc.json"
    fi
else
    printf 'DRY RUN: work directory would be %s\n' "$WORK_DIR"
fi

run_stage 01_build stage_build
run_stage 02_convert_data stage_convert_data
run_stage 03_select_data stage_select_data
run_stage 04_export_data stage_export_data
run_stage 05_data_exclusivity stage_data_exclusivity
run_stage 06_convert_mc stage_convert_mc
run_stage 07_select_mc stage_select_mc
run_stage 08_check_mc_keys stage_check_mc_keys
run_stage 09_mc_exclusivity stage_mc_exclusivity
run_stage 10_response stage_response
run_stage 11_radiative_correction stage_radiative_correction
run_stage 12_unfold stage_unfold
run_stage 13_bin_centering stage_bin_centering
run_stage 14_cross_section stage_cross_section
run_stage 15_harmonics stage_harmonics
run_stage 16_plots stage_plots

printf '\nRGK 6.535 GeV pipeline complete.\nResults: %s\nLogs: %s\n' "$RESULTS_DIR" "$LOG_DIR"
