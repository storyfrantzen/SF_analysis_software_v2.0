#!/usr/bin/env bash

# Submit the production AAO bin-centering N scan for RGK 6.535 GeV and
# RGA 10.604 GeV.  The RGA artifact is shared by the torus+1 and torus-1
# campaigns because bin centering depends on beam energy and analysis bins,
# not detector polarity.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/submit_bin_centering_swif.sh [options]

Options:
  --run                 Start the workflow after creating all jobs.
                        Without this flag the workflow remains suspended.
  --stamp UTC_STAMP     Stage identifier (default: current UTC time).
  --repo PATH           Repository root (default: inferred from this script).
  --base PATH           Campaign base directory
                        (default: /w/hallb-scshelf2102/clas12/$USER).
  --aao-xsec PATH       Compiled aao_xsec executable.
  --maid-table PATH     maid07-PPpi.tbl input.
  --module-setup PATH   JLab tcsh module setup file.
  --chunks COUNT        Chunks per campaign and N value (default: 100).
  --help                Show this help.

The production scan always evaluates N = 10, 20, and 30.  It creates two
phase-0 jobs (RGK and RGA N=10 part000) and 598 phase-1 jobs when COUNT=100.
The phase barrier prevents the bulk scan from starting unless both real
phase-0 artifacts complete successfully.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
DEFAULT_REPO="$(dirname "$SCRIPT_DIR")"

RUN_WORKFLOW=0
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPO="$DEFAULT_REPO"
BASE="/w/hallb-scshelf2102/clas12/${USER}"
CHUNKS=100
AAO_XSEC=""
MAID_TABLE="/group/clas/parms/spp_tbl/maid07-PPpi.tbl"
MODULE_SETUP=""

while (($#)); do
    case "$1" in
        --run)
            RUN_WORKFLOW=1
            shift
            ;;
        --stamp)
            (($# >= 2)) || die "--stamp requires a value"
            STAMP="$2"
            shift 2
            ;;
        --repo)
            (($# >= 2)) || die "--repo requires a path"
            REPO="$2"
            shift 2
            ;;
        --base)
            (($# >= 2)) || die "--base requires a path"
            BASE="$2"
            shift 2
            ;;
        --aao-xsec)
            (($# >= 2)) || die "--aao-xsec requires a path"
            AAO_XSEC="$2"
            shift 2
            ;;
        --maid-table)
            (($# >= 2)) || die "--maid-table requires a path"
            MAID_TABLE="$2"
            shift 2
            ;;
        --module-setup)
            (($# >= 2)) || die "--module-setup requires a path"
            MODULE_SETUP="$2"
            shift 2
            ;;
        --chunks)
            (($# >= 2)) || die "--chunks requires a value"
            CHUNKS="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

REPO="$(readlink -f "$REPO")"
[[ -d "$REPO/.git" ]] || die "not a Git repository: $REPO"
[[ "$STAMP" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] || die "invalid UTC stamp: $STAMP"
[[ "$CHUNKS" =~ ^[1-9][0-9]*$ ]] || die "--chunks must be a positive integer"

if [[ -z "$AAO_XSEC" ]]; then
    AAO_XSEC="$REPO/external/aao_gen/aao_norad/build/aao_xsec"
fi
if [[ -z "$MODULE_SETUP" ]]; then
    MODULE_SETUP="$REPO/docs/jlab-module-setup.csh"
fi

AAO_XSEC="$(readlink -f "$AAO_XSEC")"
MAID_TABLE="$(readlink -f "$MAID_TABLE")"
MODULE_SETUP="$(readlink -f "$MODULE_SETUP")"
WORKER="$REPO/scripts/run_bin_centering_swif_job.csh"

command -v swif2 >/dev/null 2>&1 || die "swif2 is not available"
command -v python3 >/dev/null 2>&1 || die "python3 is not available"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is not available"
[[ -x "$AAO_XSEC" ]] || die "aao_xsec is not executable: $AAO_XSEC"
[[ -r "$MAID_TABLE" ]] || die "MAID table is not readable: $MAID_TABLE"
[[ -r "$MODULE_SETUP" ]] || die "module setup is not readable: $MODULE_SETUP"
[[ -r "$WORKER" ]] || die "worker script is not readable: $WORKER"

declare -a CAMPAIGNS=(rgk6535 rga10604)
declare -A CAMPAIGN_ROOT=(
    [rgk6535]="$BASE/rgk_fa18_eppi0_6.535/05_corrections/bin_centering/rgk_6.535"
    [rga10604]="$BASE/rga_fa18_eppi0/05_corrections/bin_centering/rga_10.604"
)
declare -A CAMPAIGN_CONFIG=(
    [rgk6535]="$REPO/configs/analysis/rgk/6.535.json"
    [rga10604]="$REPO/configs/analysis/rga/10.604.json"
)
declare -A STAGE_ROOT

WORKFLOW="eppi0_bc_N10_20_30_${STAMP}"
STAGE_NAME="Nscan_${STAMP}"

for campaign in "${CAMPAIGNS[@]}"; do
    [[ -r "${CAMPAIGN_CONFIG[$campaign]}" ]] || \
        die "analysis config is not readable: ${CAMPAIGN_CONFIG[$campaign]}"
    STAGE_ROOT[$campaign]="${CAMPAIGN_ROOT[$campaign]}/$STAGE_NAME"
    [[ ! -e "${STAGE_ROOT[$campaign]}" ]] || \
        die "stage already exists: ${STAGE_ROOT[$campaign]}"
done

TMPDIR_BC="$(mktemp -d "/tmp/eppi0_bc_${USER}_${STAMP}.XXXXXX")"
WORKFLOW_CREATED=0
SUBMISSION_COMPLETE=0

cleanup_on_exit() {
    rc=$?
    rm -rf -- "$TMPDIR_BC"

    # Before a workflow is released, every directory bearing this marker is
    # owned solely by this invocation and is safe to remove after a failed
    # submission.  Existing paths were rejected above, so this cannot remove
    # an earlier production stage.
    if ((rc != 0 && SUBMISSION_COMPLETE == 0)); then
        printf 'Submission failed; removing this invocation\047s incomplete stage.\n' >&2
        if ((WORKFLOW_CREATED)); then
            swif2 cancel "$WORKFLOW" -delete >/dev/null 2>&1 || true
        fi
        for campaign in "${CAMPAIGNS[@]}"; do
            stage="${STAGE_ROOT[$campaign]}"
            if [[ -f "$stage/provenance/.submission_incomplete" ]]; then
                rm -rf -- "$stage"
            fi
        done
    fi
}
trap cleanup_on_exit EXIT

BUNDLE="$TMPDIR_BC/analysis_code.tar.gz"
tar --exclude='__pycache__' --exclude='*.pyc' -czf "$BUNDLE" \
    -C "$REPO" analysis/run_analysis.py analysis/eppi0

# Fail before creating campaign stages or a workflow if the executable/table
# pairing cannot produce a positive finite AAO value.
AAO_TEST_VALUE="$(MAID07_TBL="$MAID_TABLE" "$AAO_XSEC" \
    -xB 0.3 -Q2 2.0 -t -0.2 -phi 30 -BeamEnergy 6.535 \
    -theory 5 -channel 1 -resonance 0)"
python3 -c 'import math,sys; v=float(sys.argv[1]); assert math.isfinite(v) and v>0.0' \
    "$AAO_TEST_VALUE" || die "aao_xsec preflight did not return a positive finite value"

SUBMISSION_LOG="$TMPDIR_BC/submission.log"
printf 'workflow=%s\nstamp=%s\nrepo=%s\nchunks=%s\nN_values=10,20,30\n' \
    "$WORKFLOW" "$STAMP" "$REPO" "$CHUNKS" >"$SUBMISSION_LOG"

for campaign in "${CAMPAIGNS[@]}"; do
    stage="${STAGE_ROOT[$campaign]}"
    mkdir -p "$stage/provenance" "$stage/parts" "$stage/logs"
    touch "$stage/provenance/.submission_incomplete"

    cp -p "$MODULE_SETUP" "$stage/provenance/jlab-module-setup.csh"
    cp -p "$BUNDLE" "$stage/provenance/analysis_code.tar.gz"
    cp -p "${CAMPAIGN_CONFIG[$campaign]}" "$stage/provenance/analysis_config.json"
    cp -p "$AAO_XSEC" "$stage/provenance/aao_xsec"
    cp -p "$MAID_TABLE" "$stage/provenance/maid07-PPpi.tbl"
    cp -p "$WORKER" "$stage/provenance/run_bin_centering_swif_job.csh"
    cp -p "$SCRIPT_PATH" "$stage/provenance/submit_bin_centering_swif.sh"

    {
        printf 'workflow=%s\n' "$WORKFLOW"
        printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'campaign=%s\n' "$campaign"
        printf 'repo=%s\n' "$REPO"
        printf 'git_head=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
        printf 'git_branch=%s\n' "$(git -C "$REPO" branch --show-current)"
        printf 'analysis_config=%s\n' "${CAMPAIGN_CONFIG[$campaign]}"
        printf 'aao_xsec=%s\n' "$AAO_XSEC"
        printf 'maid_table=%s\n' "$MAID_TABLE"
        printf 'module_setup=%s\n' "$MODULE_SETUP"
        printf 'N_values=10,20,30\n'
        printf 'chunks_per_N=%s\n' "$CHUNKS"
        printf 'workers=8\nchunk_size=64\n'
        printf 'cores=8\nram=4gb\ndisk=2gb\ntime=24h\n'
        printf 'phase0=N10_part000\nphase1=all_remaining_parts\n'
        printf '\ngit_status:\n'
        git -C "$REPO" status --short --branch
        printf '\ngit_log:\n'
        git -C "$REPO" log -5 --oneline
    } >"$stage/provenance/submission_provenance.txt"

    printf 'job_name\tN\tchunk\tphase\toutput\tstdout\tstderr\n' \
        >"$stage/provenance/jobs.tsv"
done

swif2 create -workflow "$WORKFLOW" -max-concurrent 50 -max-problems 5 \
    >>"$SUBMISSION_LOG" 2>&1
WORKFLOW_CREATED=1

job_count=0
for campaign in "${CAMPAIGNS[@]}"; do
    stage="${STAGE_ROOT[$campaign]}"
    prov="$stage/provenance"

    for n_value in 10 20 30; do
        for ((chunk=0; chunk<CHUNKS; ++chunk)); do
            printf -v part '%03d' "$chunk"
            job_name="${campaign}_N${n_value}_p${part}"
            phase=1
            if [[ "$n_value" == 10 && "$chunk" == 0 ]]; then
                phase=0
            fi

            part_path="$stage/parts/C_BC_N${n_value}_part${part}.npz"
            stdout_path="$stage/logs/${job_name}.stdout"
            stderr_path="$stage/logs/${job_name}.stderr"

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$job_name" "$n_value" "$chunk" "$phase" \
                "$part_path" "$stdout_path" "$stderr_path" \
                >>"$stage/provenance/jobs.tsv"

            swif2 add-job -workflow "$WORKFLOW" \
                -name "$job_name" \
                -phase "$phase" \
                -cores 8 \
                -ram 4gb \
                -disk 2gb \
                -time 24h \
                -input module_setup.csh "file:$prov/jlab-module-setup.csh" \
                -input analysis_code.tar.gz "file:$prov/analysis_code.tar.gz" \
                -input analysis_config.json "file:$prov/analysis_config.json" \
                -input aao_xsec "file:$prov/aao_xsec" \
                -input maid07-PPpi.tbl "file:$prov/maid07-PPpi.tbl" \
                -input run_bin_centering_swif_job.csh \
                    "file:$prov/run_bin_centering_swif_job.csh" \
                -stdout "$stdout_path" \
                -stderr "$stderr_path" \
                -output C_BC_part.npz "file:$part_path" \
                -- /bin/tcsh -f run_bin_centering_swif_job.csh \
                    "$n_value" "$chunk" "$CHUNKS" \
                >>"$SUBMISSION_LOG" 2>&1

            ((job_count += 1))
            if ((job_count % 25 == 0)); then
                printf 'Added %d jobs\n' "$job_count"
            fi
        done
    done
done

expected_jobs=$((2 * 3 * CHUNKS))
[[ "$job_count" == "$expected_jobs" ]] || \
    die "created $job_count jobs; expected $expected_jobs"

for campaign in "${CAMPAIGNS[@]}"; do
    stage="${STAGE_ROOT[$campaign]}"
    cp -p "$SUBMISSION_LOG" "$stage/provenance/swif_submission.log"
    (
        cd "$stage/provenance"
        sha256sum \
            jlab-module-setup.csh \
            analysis_code.tar.gz \
            analysis_config.json \
            aao_xsec \
            maid07-PPpi.tbl \
            run_bin_centering_swif_job.csh \
            submit_bin_centering_swif.sh \
            submission_provenance.txt \
            jobs.tsv \
            swif_submission.log \
            >SHA256SUMS
    )
done

for campaign in "${CAMPAIGNS[@]}"; do
    rm -f "${STAGE_ROOT[$campaign]}/provenance/.submission_incomplete"
done
SUBMISSION_COMPLETE=1

if ((RUN_WORKFLOW)); then
    swif2 run "$WORKFLOW"
    state="running"
else
    state="suspended"
fi

printf '\nPrepared %d jobs in workflow %s (%s).\n' \
    "$job_count" "$WORKFLOW" "$state"
printf 'RGK stage: %s\n' "${STAGE_ROOT[rgk6535]}"
printf 'RGA stage: %s\n' "${STAGE_ROOT[rga10604]}"
printf 'Status: swif2 status %s -summary -problems\n' "$WORKFLOW"
if ((! RUN_WORKFLOW)); then
    printf 'Start:  swif2 run %s\n' "$WORKFLOW"
fi
