#!/bin/bash
#SBATCH --job-name=exp     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --array=1-7          # Number of jobs to run in parallel
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=10G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=23:59:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2020.11
source activate juptorch
python ../run_sweep.py -f exp_Gauss_sweep_Meta_exp -N 7 -gpu 1
