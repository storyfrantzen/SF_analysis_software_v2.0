#!/bin/tcsh -f

# Process the five AAORAD GEMC settings with one matched eppi0 configuration.
# Optionally pass a directory of 6.535 GeV RGK data HIPO files; it is processed
# with QADB and the identical reconstructed-candidate selection.

set script_dir = `dirname "$0"`
set project = `cd "$script_dir/.." && pwd`
set osg_base = /volatile/clas12/osg/storyf
set output_base = "$project/data/aao_osg_comparison"
set expected_files = 100

set mc_config = "$project/configs/processing/aao_6.535_eppi0_matched.json"
set data_config = "$project/configs/processing/rgk_6.535_eppi0_data.json"
set post_config = "$project/configs/post/aao_6.535_eppi0.json"

set job_ids = (11221 11222 11223 11224 11225)
set tags = ( \
    q2_0.7_ep_1.00_eg_0.005 \
    q2_0.7_ep_1.00_eg_0.010 \
    q2_0.7_ep_1.00_eg_0.015 \
    q2_0.9_ep_1.15_eg_0.005 \
    q2_0.9_ep_1.15_eg_0.010 \
)

if (! -x "$project/build/hipo2root") then
    echo "ERROR: missing executable: $project/build/hipo2root"
    exit 1
endif

if (! -x "$project/build/apply_cuts") then
    echo "ERROR: missing executable: $project/build/apply_cuts"
    exit 1
endif

mkdir -p "$output_base"

@ index = 1
while ($index <= $#job_ids)
    set job_id = "$job_ids[$index]"
    set tag = "$tags[$index]"
    set input_dir = "$osg_base/$job_id"
    set output_dir = "$output_base/$tag"

    if (! -d "$input_dir") then
        echo "[SKIP] OSG directory does not exist: $input_dir"
        @ index++
        continue
    endif

    set file_count = `find "$input_dir" -maxdepth 1 -type f -name '*.hipo' | wc -l`
    if ($file_count != $expected_files) then
        echo "[WARN] OSG $job_id has $file_count/$expected_files HIPO files; processing those present"
    endif

    mkdir -p "$output_dir"
    pushd "$output_dir" > /dev/null

    if (! -f aao_6.535_eppi0_matched.root) then
        echo "[PROCESS] OSG $job_id -> $tag"
        "$project/build/hipo2root" "$mc_config" "$input_dir"
        if ($status != 0) then
            echo "ERROR: hipo2root failed for OSG $job_id"
            popd > /dev/null
            exit 2
        endif
    else
        echo "[REUSE] $tag/aao_6.535_eppi0_matched.root"
    endif

    if (! -f aao_6.535_eppi0_selected.root) then
        "$project/build/apply_cuts" "$post_config" aao_6.535_eppi0_matched.root
        if ($status != 0) then
            echo "ERROR: apply_cuts failed for OSG $job_id"
            popd > /dev/null
            exit 3
        endif
    else
        echo "[REUSE] $tag/aao_6.535_eppi0_selected.root"
    endif

    popd > /dev/null
    @ index++
end

set data_selected = ""
set data_processed = ""
if ($#argv >= 1) then
    set data_input = "$argv[1]"
    set data_dir = "$output_base/data"
    mkdir -p "$data_dir"
    pushd "$data_dir" > /dev/null

    if (! -f rgk_6.535_eppi0_data.root) then
        echo "[PROCESS] RGK data -> data"
        "$project/build/hipo2root" "$data_config" "$data_input"
        if ($status != 0) then
            echo "ERROR: hipo2root failed for RGK data"
            popd > /dev/null
            exit 4
        endif
    endif

    if (! -f aao_6.535_eppi0_selected.root) then
        "$project/build/apply_cuts" "$post_config" rgk_6.535_eppi0_data.root
        if ($status != 0) then
            echo "ERROR: apply_cuts failed for RGK data"
            popd > /dev/null
            exit 5
        endif
    endif

    popd > /dev/null
    set data_selected = "$data_dir/aao_6.535_eppi0_selected.root"
    set data_processed = "$data_dir/rgk_6.535_eppi0_data.root"
endif

echo "[COMPARE] Writing plots under $output_base/plots"
if ("$data_selected" != "") then
    python3 "$project/scripts/compare_aao_reco.py" "$output_base" \
        --data-root "$data_selected" --data-processing-root "$data_processed"
else
    python3 "$project/scripts/compare_aao_reco.py" "$output_base"
endif
