#!/bin/bash
#SBATCH --job-name=Mbh_no_w   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --array=1-7        # Number of jobs to run in parallel
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G         # memory per cpu-core (4G per cpu-core is default)

#SBATCH --time 23:59:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2020.11
module load julia/1.6.1

source activate juptorch_julia
python ../final_only_pysr.py -N 20000 -target 5 -w 0
