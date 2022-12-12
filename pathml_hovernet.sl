#!/bin/bash -e
#SBATCH --job-name=hovernet-training
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=A100:2
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Load singularity module
module load Singularity

# Bind directories and append SLURM job ID to output directory
mkdir /nesi/nobackup/uoa03709/output/${SLURM_JOB_ID:-0}
export SINGULARITY_BIND="\
/nesi/project/uoa03709/work-dir/py-scripts:/var/inputdata,\
/nesi/nobackup/uoa03709/output/${SLURM_JOB_ID:-0}:/var/outputdata"

# Run container %runscript
srun singularity exec --nv smp-cv_0.1.4.sif python /var/inputdata/train_HoverNet.py
# srun singularity exec --nv smp-cv_0.1.4.sif nvidia-smi
# srun singularity exec --nv smp-cv_0.1.4.sif python -c "import torch; print(torch.cuda.is_available())"
