#!/bin/bash
#SBATCH --job-name=jax-gpu      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH -o %j.output
#SBATCH -e %j.output

module purge
module load anaconda3/2021.11 cudatoolkit/11.7 cudnn/cuda-11.x/8.2.0
conda activate jax-gpu

python -u  estimate.py