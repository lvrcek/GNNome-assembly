# GNNome Assembly


## Installation

1. Create a new virtual environment (either using conda or venv) with python 3.8. Install all the requirements in the from the requirements.txt file using pip. E.g.,
```bash
conda create --name assembly python=3.8 pip
pip install -r requirements.txt
```

2. Install the GPU-specific requirements:
First install PyTorch. My version is 1.9.0 and I'm using CUDA 11.1. Choose the install option based on your CUDA version:
https://pytorch.org/get-started/previous-versions/#linux-and-windows-7

Then install DGL, also based on your on the PyTorch version and the CUDA version.
https://www.dgl.ai/pages/start.html

In my case, that would be:
```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install dgl-cu111 dglgo -f https://data.dgl.ai/wheels/repo.html
```

## Quick start

To run a quick example, run:
```bash
python example.py
```
This will also download the CHM13 reference and set up the directory structure for simulating reads, constructing graphs and running experiments. Default location is in the `data` directory of the root directory of this project.


## Download the real data

Once you have the directory structure set up, you can download the real data. Note that this takes 180 GB.


## Running the code

1. Specify the train/valid/test split by editing `_train_dict`, `valid_dict`, and `_test_dict` inside `config.py`. Inside each dictionary specify how many graphs created from which chromosome you wish to train/validate/test on. For real data, add suffix "_r".
E.g., to train on two graphs of chromosome 19 and one graphs of chromosome 20, valdiate on one chromosome 19 graphs, and test on chromosome 21 graph created from real PacBio HiFi data, write:
```python
_train_dict = {'chr19': 2, 'chr20': 1}
_valid_dict = {'chr19': 1}
_test_dict = {'chr21_r': 1}
```
Note that all three chr19 graphs are created from different sets of reads (sampled anew each time), and thus are different themselves.
Also note that you cannot specify more than one graph per real chromosome.


2. [Optional ]Adjust the hyperparameters.
All the hyperparameters are in the `hyperparameters.py`. Change them by editing the dictionary inside the file.


3. Running the pipeline.
```bash
python -u pipeline.py --data <data_path> --out <out_name> &> output.log &
```

Argument `--data` is the path where the training data is stored. The file structure of the data directory should have a specific structure (DGL graph needs to be inside the `processed` directory), so if I will send you the data I will probably just send it with that particular structure, even though I might not include additional files which are not necesary.

Argument `--out` is the name that will be used for saving the model and the checkpoints during the training. The models are saved inside the `pretrained` directory. E.g., with `--out gatedgcn_12-04-22`, the model will be saved int `pretrained/model_gatedgcn_12-04-22.pt`, and all the checkpoint information (optimizer parameters, loss, etc.) will be saved inside `checkpoints/gatedgcn_12-04-22.pt`.

