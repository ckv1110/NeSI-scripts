#!/bin/bash -e
#SBATCH --job-name=apptainer_build
#SBATCH --partition=milan
#SBATCH --time=0-02:00:00
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module purge
module load Apptainer/1.1.2

export APPTAINER_CACHEDIR=/nesi/nobackup/uoa03709/containers/to-containerize/apptainer_cache
export APPTAINER_TMPDIR=/nesi/nobackup/uoa03709/containers/to-containerize/apptainer_tmpdir
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
setfacl -b $APPTAINER_TMPDIR

cd /nesi/project/uoa03709/containers/to-containerize/

apptainer build --fakeroot smp_cv_0.2.9.1.sif smp_0-2-9-1.def
