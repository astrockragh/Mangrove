import os, json, shutil, argparse

import os.path as osp

parser = argparse.ArgumentParser()

##folder to take experiments from
parser.add_argument("-f", "--f", type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=False)

##should be implemented in the slurm script --------------------------------

# parser.add_argument("-gpu", "--gpu", type=str, required=False)
# parser.add_argument("-cpus", "--cpus", type=str, required=False)


## slurm  ------------------------------------------------------------------

os.chdir(osp.expanduser("~/work/GraphMerge"))

args = parser.parse_args()

exp0_folder = str(args.f)

##########################################################
#      Loop over JSON files and train models             # 
##########################################################

# Generate list over experiments to run
from dev.utils import list_experiments
if args.gpu=='1':
    from dev.train_script_gpu import train_model
else:
    from dev.train_script_cpu import train_model
    # from dev.train_script_cpu_debug import train_model



#### --------------------------------- ####
# implement for slurm script and slurm .out
# clean_done(exp0_folder)
#### --------------------------------- ####

exp_folder, exp_list = list_experiments(exp0_folder)
print('whoops')

print(f"Starting process with {len(exp_list)} experiments" )
print(exp_list)
# Loop over the experiments
for i, experiment in enumerate(exp_list):
    print('Running ' + experiment[:-5])
    # Load construction dictionary from json file
    with open(osp.join(exp_folder, experiment)) as file:
        construct_dict = json.load(file)
    construct_dict['experiment_name']=experiment[:-5]


    print(f"Starting experiment from {experiment[:-5]}")

    train_model(construct_dict)

    # print(f'Exited training after {epochexit} epochs')
    if construct_dict['move']:
        shutil.move(osp.join(exp_folder, experiment), osp.join(exp0_folder+"/done", experiment))
    print(f"Experiment {experiment[:-5]} done \t {experiment}: {i + 1} / {len(exp_list)}")
