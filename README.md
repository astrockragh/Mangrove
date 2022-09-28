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

- "experiment":(str) For logging on wandb, not necessary if local
- "group": (str) For logging locally (and grouping on wandb)
- "move": (bool) Whether or not to move a done experiment into done after running it
- "model":  Model name (str), as in dev/models.py
- "log": (bool) whether or not to log

- "run_params": Parameters= group for doing the run
    - "n_epochs": number of epochs (int)
    - "n_trials": number of trials of the same experiment (int)
    - "batch_size": batch size (int)
    - "val_epoch": The number of epochs between validation set checks (int)
    - "early_stopping": Whether to do early stopping (bool)
    - "patience": patience (in epoch) for early stopping (int)
    - "l1_lambda": L1 regularization (float)
    - "l2_lambda": L2 regularization (float)
    - "loss_func": loss function name from dev/loss_funcs.py (str)
    - "metrics": metric name from dev/metrics.py (str)
    - "performance_plot": plot name from dev/eval_plot.py (str)
    - "save":Whether or not to save the model and results (bool)
    - "seed": Setting torch/random seed or not (bool)
    - "num_workers": Number of workers for dataloader, should be equal to number of cpu cores available on the system (int)

- "learn_params": Parameters related to learning scheme
    - "learning_rate":   Learning rate (float)
    - "schedule":  Learning Rate scheduler name from dev/lr_schedule.py (str)
    - "g_up": Exponent for warmup (float). $(g_{up})^{epoch}*\text{learning rate}/ ((g_{up})^{warmup})$
    - "g_down": Exponent for cooldown (float). $(g_{down})^{epoch}*\text{learning rate}$
    - "warmup": Number of epochs to warm up for (int)
    - "period": Period for cosine annealing in epochs (int)
    - "eta_min": Mininum learning rate for cosine annealing (float)

- "hyper_params": Hyper parameters for model
    - "hidden_channels": Size of latent space (int)
    - "conv_layers": Number of convolutional layers (int)
    - "conv_activation": Activation function between convolutional layers (str) ['relu', 'leakyrelu']
    - "decode_activation": Activation function between decode layers (str)  ['relu', 'leakyrelu']
    - "decode_layers": Number of decode layers (int)
    - "layernorm": Whether or not to use layer normalization (bool)
    - "agg": What type of global aggregation to use (str) ['sum', 'max']
    - "variance": Whether or not to predict variance (bool)
    - "rho": number of correlations to predict (int)

- "data_params": Parameters for loading the data
    - "case": Data path (str)
    - "targets": Which targets to optimize for (list(int)),
    - "del_feats": Which features to leave out (list(int))
    - "split": Where to split between train/val (float (0;1) interval)
    - "test": Whether or not to use testing mode (bool)
    - "scale": Whether or not to scale targets to have mean 0, variance 1 (bool)


## All in all an experiment file looks like this
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