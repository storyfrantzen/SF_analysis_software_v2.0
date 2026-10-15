# jlab-module-setup.csh
#
# Source this file from tcsh.  The CLAS12 module supplies a matched Python,
# NumPy/SciPy/Matplotlib, and PyROOT stack.  Do not combine its Python package
# paths with /usr/bin/python3 or /usr/bin/python3.12: compiled extensions from
# the module are built for the module-selected interpreter.

set sf_analysis_setup_status = 0
set sf_analysis_python = ""

module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles
if ( $status != 0 ) set sf_analysis_setup_status = 1

module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
if ( $status != 0 ) set sf_analysis_setup_status = 1

module load clas12/5.4
if ( $status != 0 ) set sf_analysis_setup_status = 1

if ( $sf_analysis_setup_status == 0 ) then
    # tcsh caches command locations.  Refresh after the module changes PATH so
    # python3 resolves to the module interpreter rather than /usr/bin/python3.
    rehash
    which python3 >& /dev/null
    if ( $status != 0 ) then
        echo "ERROR: clas12/5.4 did not provide python3" >&2
        set sf_analysis_setup_status = 1
    else
        set sf_analysis_python = `which python3`
    endif
endif

if ( $sf_analysis_setup_status == 0 ) then
    "$sf_analysis_python" -c \
        'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
    if ( $status != 0 ) then
        echo "ERROR: analysis requires Python 3.10 or newer: $sf_analysis_python" >&2
        set sf_analysis_setup_status = 1
    endif
endif

if ( $sf_analysis_setup_status == 0 ) then
    "$sf_analysis_python" -c 'import numpy, scipy, matplotlib, ROOT'
    if ( $status != 0 ) then
        echo "ERROR: module Python cannot import NumPy, SciPy, Matplotlib, and PyROOT" >&2
        set sf_analysis_setup_status = 1
    endif
endif

if ( $sf_analysis_setup_status == 0 ) then
    setenv SF_ANALYSIS_PYTHON "$sf_analysis_python"
    echo "SF analysis Python: $SF_ANALYSIS_PYTHON"
    unset sf_analysis_python
    unset sf_analysis_setup_status
    /bin/true
else
    unset sf_analysis_python
    unset sf_analysis_setup_status
    /bin/false
endif
