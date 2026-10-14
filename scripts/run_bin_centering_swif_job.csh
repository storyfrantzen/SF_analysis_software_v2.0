#!/bin/tcsh -f

# Run one flattened (Q2, xB, -t) chunk of the AAO bin-centering scan.
# All files referenced here are staged by submit_bin_centering_swif.sh.

if ( $#argv != 3 ) then
    echo "Usage: $0 N chunk_index chunk_count" >&2
    exit 64
endif

set n_samples = "$1"
set chunk_index = "$2"
set chunk_count = "$3"

echo "host=`hostname`"
echo "workdir=$cwd"
echo "N=$n_samples chunk_index=$chunk_index chunk_count=$chunk_count"

if ( -e /etc/profile.d/modules.csh ) then
    source /etc/profile.d/modules.csh
endif

if ( ! -r module_setup.csh ) then
    echo "ERROR: missing staged module_setup.csh" >&2
    exit 65
endif
source module_setup.csh
set setup_status = $status
if ( $setup_status != 0 ) then
    echo "ERROR: module setup failed with status $setup_status" >&2
    exit $setup_status
endif

foreach required ( analysis_code.tar.gz analysis_config.json aao_xsec maid07-PPpi.tbl )
    if ( ! -r "$required" ) then
        echo "ERROR: missing staged input $required" >&2
        exit 66
    endif
end

tar -xzf analysis_code.tar.gz
set tar_status = $status
if ( $tar_status != 0 ) then
    echo "ERROR: analysis archive extraction failed with status $tar_status" >&2
    exit $tar_status
endif

if ( ! -f analysis/run_analysis.py ) then
    echo "ERROR: analysis archive does not contain analysis/run_analysis.py" >&2
    exit 67
endif

chmod u+x aao_xsec
setenv MAID07_TBL "$cwd/maid07-PPpi.tbl"

python3 analysis/run_analysis.py bin-centering \
    --config analysis_config.json \
    --output C_BC_part.npz \
    --exe "$cwd/aao_xsec" \
    --N "$n_samples" \
    --workers 8 \
    --chunk-size 64 \
    --progress-chunks 1 \
    --bin-chunks "$chunk_count" \
    --bin-chunk-index "$chunk_index"
set analysis_status = $status

if ( $analysis_status != 0 ) then
    echo "ERROR: bin-centering failed with status $analysis_status" >&2
    exit $analysis_status
endif

if ( ! -s C_BC_part.npz ) then
    echo "ERROR: C_BC_part.npz was not created" >&2
    exit 68
endif

ls -lh C_BC_part.npz
exit 0
