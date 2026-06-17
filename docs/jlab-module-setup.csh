# jlab-module-setup.csh

module use /cvmfs/oasis.opensciencegrid.org/jlab/scicomp/sw/el9/modulefiles
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module load clas12/5.4

setenv LD_LIBRARY_PATH $PWD/build/install/lib:$LD_LIBRARY_PATH
