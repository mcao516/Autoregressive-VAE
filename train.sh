#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/mcao610/slurm-%j.out  # Write the log in $SCRATCH

# 1. Create your environement locally
module load miniconda3
source activate py37

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
# cp $SCRATCH/<dataset.zip> $SLURM_TMPDIR

# 3. Eventually unzip your dataset
# unzip $SLURM_TMPDIR/<dataset.zip> -d $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python main.py --ignore_pad_idx

# 5. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/<to_save> $SCRATCH
