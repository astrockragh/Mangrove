#!/bin/bash
#SBATCH --job-name=final_all     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --array=1-20            # Number of jobs to run in parallel
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1            # number of gpus per node
#SBATCH --time=23:59:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2020.11
source activate juptorch
python ../run_sweep.py -f exp_Sage_finals -N 8 -gpu 1