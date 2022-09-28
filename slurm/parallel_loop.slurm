#!/bin/bash
#SBATCH --job-name=test_par     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=2              # Number of jobs to run in parallel
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:10:00         # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email when job begins, ends and fails
#SBATCH --mail-user=cj1223@princeton.edu

module purge
module load anaconda3/2020.11
source activate juptorch

srun -N 1 -n 1 python ../run_sweep.py -f exp_L2_sweep_Sage_const -N 1 -gpu 1 &
srun -N 1 -n 1 python ../run_sweep.py -f exp_L2_sweep_Sage_const -N 1 -gpu 1 &
# srun -N 1 -n 1 python ../run_sweep.py -f exp_L2_sweep_Sage_const -N 1 -gpu 1 &
# srun -N 1 -n 1 python ../run_sweep.py -f exp_L2_sweep_Sage_const -N 1 -gpu 1 &
wait
