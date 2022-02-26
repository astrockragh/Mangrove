# GraphMerge
Using Dark Matter Merger Trees to infer properties of interest

For people looking to reproduce the results of [our paper](arxivlink), PAPER TITLE, the folders to look are in data and dev. 

You can preprocess the merger tree with data_z.py (or data_SAM.py if you want fewer nodes), although you'll have to fit a transformer independently, since I found that to be the best. You can find a procedure for doing so in the subfolder transform, but if you're going for speed and already know what subset of features you're interested in, I suggest you fit the transformer in a different way. Ours is fit for each column which isn't fast.

Having restructured the merger tree, you can then do the training either as single experiments (using run_experiment.py) or as a sweep (using run_sweep.py). All the things required to do the training and tracking are in the dev folder. Here, you'll find loss functions, learning rate schedulers, models and a script for doing the training on the gpu/cpu (the cpu version is outdated).

For examples of either a single experiment or a sweep, check out the guide.txt files in the exp_example for the single experiment and exp_sweep_example for a sweep.


## Required dependencies

This is for creating a conda environment to do your coding in

Replace your anaconda module with whatever you have on your computer/cluster

`module load anaconda3/2021.5`
`conda create --name jtorch pytorch==1.9.0 jupyter torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 matplotlib tensorboard --channel pytorch`
`conda activate jtorch`
`pip install accelerate scikit-learn pandas`

To determine the pytorch_geometric version that you need, check out https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html 
I recommend doing it through pip, not conda, like the below version, but check the docs

`pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu102.html`

## Structure of json experiment setup

- "experiment": "GraphMerge"
- "group":  "final_Gauss2dcorr_all" 
- "move":false
- "model":  "Sage"
- "log": true

- "run_params":
    - "n_epochs": 500
    - "n_trials": 1
    - "batch_size": 256
    - "val_epoch": 2
    - "early_stopping": false
    - "patience": 100
    - "l1_lambda":0
    - "l2_lambda":0
    - "loss_func": "Gauss2d_corr"
    - "metrics": "test_multi_varrho"
    - "performance_plot": "multi_base"
    - "save":true
    - "seed":true
    - "num_workers": 4

- "learn_params":
    - "learning_rate":   1e-2
    - "schedule":   "onecycle"
    - "g_up":1
    - "g_down":0.95
    - "warmup":4
    - "period":5
    - "eta_min":1e-5

- "hyper_params": 
    - "hidden_channels": 128  
    - "conv_layers": 5
    - "conv_activation": "relu"
    - "decode_activation": "leakyrelu"
    - "decode_layers":   3
    - "layernorm": true
    - "agg": "sum"
    - "variance": true
    - "rho": 1

- "data_params":
    - "case": "vlarge_all_4t_z0.0_quantile_raw"
    - "targets": [0,1],
    - "del_feats": []
    - "split": 0.8
    - "test": 0
    - "scale": 0

```json
{
    "experiment": "GraphMerge",
    "group":  "final_Gauss2dcorr_all", 
    "move":false,
    "model":  "Sage",
    "log": true,

    "run_params":{
    "n_epochs":    500,
    "n_trials": 1,
    "batch_size": 256,
    "val_epoch":   2,
    "early_stopping": false,
    "patience": 100,
    "l1_lambda":0,
    "l2_lambda":0,
    "loss_func": "Gauss2d_corr",
    "metrics": "test_multi_varrho",
    "performance_plot": "multi_base",
    "save":true,
    "seed":true,
    "num_workers": 4
}, 
"learn_params":{
    "learning_rate":   1e-2,
    "schedule":   "onecycle",
    "g_up":1,
    "g_down":0.95,
    "warmup":4,
    "period":5, 
    "eta_min":1e-5
},

    "hyper_params": {
        "hidden_channels":   128,  
        "conv_layers":   5,
        "conv_activation": "relu",
        "decode_activation": "leakyrelu",
        "decode_layers":   3,
        "layernorm":      true,
        "agg": "sum",
        "variance":     true,
        "rho":        1
    },

   "data_params":{ 
    "case": "vlarge_all_4t_z0.0_quantile_raw",
    "targets": [0,1,2],
    "del_feats": [],
    "split": 0.8,
    "test": 0,
    "scale": 0
}
}
```