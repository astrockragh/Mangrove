import pickle, time, os, random, argparse, string
import numpy as np
import os.path as osp
import pandas as pd

parser = argparse.ArgumentParser()

##folder to take experiments from
parser.add_argument("-target", "--target", type=int, required=True)
parser.add_argument("-N", "--N", type=int, required=True)
parser.add_argument("-w", "--w", type=int, required=True)
args = parser.parse_args()

from pysr import PySRRegressor

N = int(args.N)
weight = int(args.w)

scale=False

# Targets
os.chdir('..') ## avoid slurm issues

target = int(args.target)
target_cols = ['Mstar', 'Mcold', 'Zcold', 'SFR', 'SFR100', 'Mbh']
tcol = target_cols[target]

if scale:
    xs = pickle.load(open(osp.expanduser(f"~/../../../scratch/gpfs/cj1223/GraphStorage/standard_raw_final_6t/xs.pkl"), 'rb'))
    ys = pickle.load(open(osp.expanduser(f"~/../../../scratch/gpfs/cj1223/GraphStorage/standard_raw_final_6t/ys.pkl"), 'rb'))
else:
    xs = pickle.load(open(osp.expanduser(f"~/../../../scratch/gpfs/cj1223/GraphStorage/raw_raw_final_6t/xs.pkl"), 'rb'))
    ys = pickle.load(open(osp.expanduser(f"~/../../../scratch/gpfs/cj1223/GraphStorage/raw_raw_final_6t/ys.pkl"), 'rb'))
print('Loaded data')

from datetime import date
today = date.today()

today = today.strftime("%d%m%y")

fname = f'eqs_{today}'

if  not osp.isdir(fname):
    os.mkdir(fname)
    # df = pd.DataFrame(columns=['Mstar', 'Mcold', 'Zcold', 'SFR', 'SFR100', 'Mbh'])
    # df.to_csv(fname+'best.csv')

def make_id(length=6):
    # choose from all lowercase letters
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for _ in range(length))
    return result_str

idx=make_id()
fname = fname + '/' + idx + f'_{tcol}' +'.csv'
# +'|'+fname+idx+'_backup.csv'

P = (xs[:,1]+9)**3
P/=np.sum(P)
if N>2000:
    massMask = xs[:,1]>11.5
    idxM = np.arange(len(xs))[massMask]
    idx = np.random.choice(np.arange(len(xs))[~massMask],N - np.sum(massMask),replace=False, p = P[~massMask]/np.sum(P[~massMask]))
    idx = np.hstack([idxM, idx])
else:
    idx = np.random.choice(np.arange(len(xs)),N ,replace=False, p = P)


# idx = random.choices(np.arange(len(ys)), k=N)
print(f'Selecting {N} random halos')

x_pysr = xs[idx]
y_pysr = ys[idx,target]

if weight:
    weights = (xs[idx,1]-5)**5
else:
    weights = None

model = PySRRegressor(
    procs=40*2,
    niterations=100,
    populations=20,
    population_size = 1000,
    use_frequency=True,
    multithreading=True, 
    binary_operators=["plus", "sub", "mult", "div", "pow", 'greater'],
    unary_operators = ['neg', 'square', 'cube', 'log10_abs', 'exp', 'log_abs', 'tanh', 'abs', 'erf', 'sign', 'relu'], ##still need a sigmoid
    constraints={'pow': (-1, 1)}, ##=just added 17/04-22, set to {-1, 0} maybe
    batching=1, 
    batch_size=1024,
    # select_k_features=10,  ## this makes the functions a bit weird
    nested_constraints = {'pow': {'pow': 1}}, ## this may be important
    maxsize=25, 
    update=False,
    # cluster_manager = "slurm",
    equation_file = fname
)

model.fit(X=x_pysr, y=y_pysr, weights = weights)
eqs = model.equations

eq = eqs.sort_values(by='loss', ascending=True).iloc[0]
a=eq['lambda_format'](xs)
l = np.nanstd(a)

print(fname, l)

# with open(fname[:-4]+'_obj.pkl', 'wb') as handle:
#     # pickle.dumps(model, handle)
#     dill.dump(model, handle)

# if l < low:
#     print('Succeeded criterion')
#     dfbest=pd.read_csv('best.csv')
#     vals = pd.Series(f"{fname}_{l:.2f}", index=tcol)
#     dfbest = dfbest.append(vals, ignore_index=True)
#     dfbest.to_csv('best.csv')
