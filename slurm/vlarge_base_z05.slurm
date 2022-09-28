#!/bin/bash
#SBATCH --job-name=noscale_z05    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=24:00:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2020.11
source activate juptorch

python ../data/data_z.py -case vlarge_all_4t -vols 555 -zcut 0.5