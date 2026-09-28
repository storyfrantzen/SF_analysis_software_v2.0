#!/bin/bash
# Build AAO-radiative generated/reconstructed diagnostics with the maintained
# SF analysis candidate selection.  Supply AREPO, AAOHIPODIR, and AAOOUT with
# sbatch --export; no repository files are modified by the job.

#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=aao_rad_genrec

set -euo pipefail

: "${AREPO:?export the SF_analysis_software_v2.0 repository path}"
: "${AAOHIPODIR:?export the AAO radiative HIPO directory}"
: "${AAOOUT:?export a dedicated output directory}"
AAO_MAX_FILES=${AAO_MAX_FILES:-200}

source /etc/profile.d/modules.sh
module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module load clas12/5.4

processing_config="$AREPO/configs/processing/rga/10.604/eppi0_genrec_diagnostic_torus-1.json"
post_config="$AREPO/configs/diagnostics/rga/10.604/eppi0_genrec_candidate_torus-1.json"
analysis_config="$AREPO/configs/analysis/rga/10.604.json"
converter_root="$AAOOUT/10.604_rga_fa18_torus-1_eppi0_genrec_diagnostic.root"
selected_root="$AAOOUT/10.604_rga_fa18_torus-1_eppi0_genrec_candidates.root"
event_sample="$AAOOUT/aao_rad_event_sample.npz"
plot_dir="$AAOOUT/plots"
event_pdf="$plot_dir/reconstructed_event_kinematics.pdf"
all_hipo_files="$AAOOUT/all_hipo_files.txt"
hipo_manifest="$AAOOUT/aao_rad_${AAO_MAX_FILES}_files.txt"

test -x "$AREPO/build/hipo2root"
test -x "$AREPO/build/post_process"
test -f "$processing_config"
test -f "$post_config"
test -f "$analysis_config"
test -f "$AREPO/analysis/plot_event_kinematics.py"
test -d "$AAOHIPODIR"

mkdir -p "$AAOOUT" "$plot_dir" "$AAOOUT/.mplconfig"
export MPLCONFIGDIR="$AAOOUT/.mplconfig"
cd "$AAOOUT"

for output in "$converter_root" "$selected_root" "$event_sample"
do
    if [ -e "$output" ]; then
        echo "FATAL: refusing to replace existing output: $output" >&2
        exit 1
    fi
done
for output in \
    gen_kinematics.png rec_kinematics.png gen_vs_rec.png gen_rec_overlay.png efficiency.png \
    header_nu_check.png results.md reconstructed_event_kinematics.pdf \
    reconstructed_event_kinematics_summary.json
do
    if [ -e "$plot_dir/$output" ]; then
        echo "FATAL: refusing to replace existing output: $plot_dir/$output" >&2
        exit 1
    fi
done

find "$AAOHIPODIR" -type f -name '*.hipo' -print | sort > "$all_hipo_files"
sed -n "1,${AAO_MAX_FILES}p" "$all_hipo_files" > "$hipo_manifest"
selected_files=$(wc -l < "$hipo_manifest")
if [ "$selected_files" -ne "$AAO_MAX_FILES" ]; then
    echo "FATAL: requested $AAO_MAX_FILES HIPO files, found only $selected_files" >&2
    exit 1
fi

echo "AAO HIPO files available: $(wc -l < "$all_hipo_files")"
echo "AAO HIPO files selected: $selected_files ($hipo_manifest)"
echo "Converter config: $processing_config"
echo "Candidate config: $post_config"

"$AREPO/build/hipo2root" \
    "$processing_config" "@$hipo_manifest" 0 100000 \
    > "$AAOOUT/hipo2root.log" 2>&1
test -s "$converter_root"

"$AREPO/build/post_process" \
    "$post_config" "$converter_root" 100000 \
    > "$AAOOUT/post_process.log" 2>&1
test -s "$selected_root"

python3 "$AREPO/analysis/build_event_sample.py" \
    "$converter_root" "$selected_root" "$event_sample" \
    --beam-energy 10.604 \
    > "$AAOOUT/build_event_sample.log" 2>&1
test -s "$event_sample"

python3 "$AREPO/analysis/plot_event_kinematics.py" \
    "$selected_root" \
    --config "$analysis_config" \
    --output "$event_pdf" \
    --gemc-event-sample "$event_sample" \
    --gemc-output-dir "$plot_dir" \
    --label "AAO radiative, RGA Fall 2018 inbending, candidate-level detected" \
    > "$AAOOUT/plot_gen_rec.log" 2>&1

for output in \
    gen_kinematics.png rec_kinematics.png gen_vs_rec.png gen_rec_overlay.png efficiency.png \
    header_nu_check.png results.md reconstructed_event_kinematics.pdf \
    reconstructed_event_kinematics_summary.json
do
    test -s "$plot_dir/$output"
done

echo "DONE"
ls -lh "$converter_root" "$selected_root" "$event_sample" "$plot_dir"/*
