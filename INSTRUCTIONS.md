# GNNome Assembly


## Installation

1. Create a new virtual environment (either using conda or venv) with python 3.8. Install all the requirements in the from the requirements.txt file using pip. E.g.,
```
conda create --name assembly python=3.8 pip
pip install -r requirements.txt
```

2. Install the GPU-specific requirements:
First install PyTorch. My version is 1.9.0 and I'm using CUDA 11.1. Choose the install option based on your CUDA version:
https://pytorch.org/get-started/previous-versions/#linux-and-windows-7

Then install DGL, also based on your on the PyTorch version and the CUDA version.
https://www.dgl.ai/pages/start.html

In my case, that would be:
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
```


## Running the code

1. Adjust the hyperparameters.
All the hyperparameters are in the `hyperparameters.py` stored inside a dictionary. I've commented out the ones that I used previously for testing and are not important anymore. These will probably be removed next time I will be refactoring the code.

2. Running the training loop.
```
python -u train.py --data <data_path> --out <out_name> &> output.log &
```

Argument `--data` is the path where the training data is stored. The file structure of the data directory should have a specific structure (DGL graph needs to be inside the `processed` directory), so if I will send you the data I will probably just send it with that particular structure, even though I might not include additional files which are not necesary.

Argument `--out` is the name that will be used for saving the model and the checkpoints during the training. The models are saved inside the `pretrained` directory. E.g., with `--out gatedgcn_12-04-22`, the model will be saved int `pretrained/model_gatedgcn_12-04-22.pt`, and all the checkpoint information (optimizer parameters, loss, etc.) will be saved inside `checkpoints/gatedgcn_12-04-22.pt`.


**Note:** I'm using Weights & Biases for experiment tracking, so you might need to have your own account for running the experiments, or you can comment out all the W&B parts of the code, but then you won't get the plots. Not sure what's the best here, maybe if you had your own account and I could invite you into my workspace for this project (if possible).

## Parts of the code


Training: https://github.com/lvrcek/neural-path-finding/blob/master/train.py#L215-L258

Model: https://github.com/lvrcek/neural-path-finding/blob/master/models/block_graph.py#L27-L45

GatedGCN layer: https://github.com/lvrcek/neural-path-finding/blob/master/layers/gated_gcn_blocks.py#L13-L97

Inference: Still a bit too messy and spread out over several functions, I will clean it up soon and will update this.
