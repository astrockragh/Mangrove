import os, json, argparse
import numpy as np
import os.path as osp

parser = argparse.ArgumentParser()

##folder to take experiments from
parser.add_argument("-f", "--f", type=str, required=True)
parser.add_argument("-gpu", "--gpu", type=str, required=False)
parser.add_argument("-N", "--N", type=str, required=True)


##should be implemented in the slurm script --------------------------------

# parser.add_argument("-gpu", "--gpu", type=str, required=False)
# parser.add_argument("-cpus", "--cpus", type=str, required=False)


## slurm  ------------------------------------------------------------------

os.chdir(osp.expanduser("~/work/GraphMerge"))

args = parser.parse_args()

exp0_folder = str(args.f)
N = int(str(args.N))

##########################################################
#      Loop over JSON files and train models             # 
##########################################################

# Generate list over experiments to run
from dev.utils import list_experiments
if args.gpu=='1':
    from dev.train_script_gpu import train_model
else:
    from dev.train_script_cpu import train_model

from dev.utils import perms

# Load construction dictionary from json file
with open(f"{exp0_folder}/diff.json") as file:
    diff = json.load(file)
    
keys=list(diff.keys())
exp_list=perms(diff)
# Load construction dictionary from json file
with open(f"{exp0_folder}/base.json") as file:
    base = json.load(file)

print(f"Starting process with {len(exp_list)} experiments")
print(diff)
# Loop over the experiments

if len(exp_list)>N:
    print(f'Selected too many compare runs, running {N} random runs')
    idxs = np.random.permutation(len(exp_list))
    exp_list=exp_list[idxs[:N]]

for i in range(len(exp_list)):
    construct_dict=base
    print('Exploring', keys)
    print('Currently doing', exp_list[i])
    # if i==0:
    #     construct_dict['data_params']['restart']=True
    # else:
    #     construct_dict['data_params']['restart']=False
    for j, key in enumerate(keys):
        if key in construct_dict['run_params']:
            typ=type(construct_dict['run_params'][key])
            construct_dict['run_params'][key]=typ(exp_list[i][j])
        elif key in construct_dict['learn_params']:
            typ=type(construct_dict['learn_params'][key])
            construct_dict['learn_params'][key]=typ(exp_list[i][j])
        elif key in construct_dict['hyper_params']:
            typ=type(construct_dict['hyper_params'][key])
            construct_dict['hyper_params'][key]=typ(exp_list[i][j])
        elif key in construct_dict['data_params']:
            typ=type(construct_dict['data_params'][key])
            construct_dict['data_params'][key]=typ(exp_list[i][j])
        elif key in construct_dict:
            typ=type(construct_dict[key])
            construct_dict[key]=typ(exp_list[i][j])
    #make_title
    title=''
    for key, val in zip(keys, exp_list[i]):
        title+=key[:3]+str(val)
    construct_dict['experiment_name']=title
    epochexit=train_model(construct_dict)
    print(f'Exited training after {epochexit} epochs')
    print(f"Experiment {i+1} done: {i + 1} / {len(exp_list)}")

