import os, sys, json, shutil
import os.path as osp
import numpy as np

from importlib import import_module


cwd = osp.abspath('')

def list_experiments(folder):
    experiment_folder = osp.join(cwd, folder, "todo") 
    experiment_files  = os.listdir(experiment_folder)
    return experiment_folder, experiment_files
### rewrite clean_done
def clean_done(folder):
    experiment_folder = osp.join(cwd, folder, "done") 
    legacy_path=osp.join(cwd, folder, 'legacy')
    if not osp.exists(legacy_path):
        os.mkdir(legacy_path)
        print('Legacy made')
    try:
        files  = os.listdir(experiment_folder)
        for f in files:
            shutil.move(osp.join(experiment_folder,f), folder, 'legacy/')
    except:
        os.mkdir(experiment_folder)
    print('Cleaned done folder')


def perms(diffs):
    from itertools import product
    keys=list(diffs.keys())
    val=list(diffs.values())
    for i, s in enumerate(val):
        if i==0:
            a=val[0]
        else:
            a=product(a, val[i])
    bs=[]
    for b in a:
        bs.append(b)
    output=[]
    def removeNestings(l):
        for i in l:
            if type(i) == tuple:
                removeNestings(i)
            else:
                output.append(i)
    removeNestings(bs)
    perms=np.array(output)
    return perms.reshape(-1, len(keys))
