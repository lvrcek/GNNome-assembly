# GNNome Assembly

A framework for training graph neural networks to untangle assembly graphs obtained from de novo genome assembly.

This framework consists of simulating the synthetic reads, generating the assebmly graphs, training a Graph Neural Network to predict edge probabilities, and finally decoding these edge probabilities with greedy search. The final result are assembly sequences stored in the commonly used FASTA format.

<p align="center">
  <img src="figures/framework.png" width="800" title="Framework">
</p>


## Installation

### Requirements
- conda: 4.6+

### Setting up virtual environment

#### 1. Create a conda virtual environment
```bash
conda env create -f environment.yml
conda activate gnn-assembly
```

#### 2. Install the GPU-specific requirements
Use pip for both of the installations bellow.

- **PyTorch**: Install version 1.9 or higher, based on your CUDA version:
https://pytorch.org/get-started/previous-versions/#linux-and-windows-7

- **DGL**: Install version 3.8 or higher, based on your CUDA version:
https://www.dgl.ai/pages/start.html

For example, for CUDA 11.1, this could be:
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

Apart from setting up the working directory, running the above example will simulate four read datasets for chromosome 19 and one read dataset for chromosome 21 and construct graphs from those reads. Subsequently, it will train a model on three chromosome 19 graphs, validate it on one chromosome 19 graph, and, finally, create assembly for chromosome 21. After reconstruction, the assembly sequences can be found in `data/experiments/test_example/assembly/0_assembly.fasta`.


## Download the real data

The real data is automatically downloaded by running the `example.py` script, as described above or if you run a custom run with `pipeline.py`, as described in the next section. However, you can also download the real dataset separately, by running:
```bash
bash download_dataset.sh <data_path>
```
after which the dataset will be saved in `<data_path>/real/`

**Note:** This dataset has 43 GB when compressed (during download) and 180 GB when uncompressed (used during training/evaluation).

## Reproduce the results

The results can easily be reproduced by running `reproduce.py` script, with `--mode` argument set to either `synth` for synthetic data or `real` for real data. For example:
```bash
python reproduce.py --mode real
```
will load the model pretrained on 15 chromosome 19 graphs, which is stored in `pretrained_models/model_15xchr19.pt`, and produce assembly sequences for all 23 real-data chromosomes. We suggest you evaluate the obtained assemblies with Quast, as explained at the end of this README.

## Usage

### 1. Specify the train/valid/test split
You can choose on which graphs to train, validate, and test, by editing `_train_dict`, `valid_dict`, and `_test_dict` inside `config.py`.
Inside each dictionary specify how many graphs created from which chromosome you wish to train, validate, and test on. For real data, add suffix "_r".

E.g., to train on two graphs of chromosome 19 and one graphs of chromosome 20, valdiate on one chromosome 19 graphs, and test on chromosome 21 graph created from real PacBio HiFi data, write:
```python
_train_dict = {'chr19': 2, 'chr20': 1}
_valid_dict = {'chr19': 1}
_test_dict = {'chr21_r': 1}
```
Note that all three chr19 graphs are created from different sets of reads (sampled anew each time), and thus are different themselves.
Also note that you cannot specify more than one graph per real chromosome.


### 2. [Optional] Adjust the hyperparameters

All the hyperparameters are in the `hyperparameters.py`. Change them by editing the dictionary inside the file.


### 3. Run the pipeline
```bash
python pipeline.py --data <data_path> --out <out_name>
```

Arguments:

- `--data`: (default value: `data/`) path where simulated sequences, real sequences, constructed graphs, and the experiments will be stores. 

- `--out`: (default value: timestamp at running the pipeline) name that will be used for naming the directories for train/valid/test, and saving the model and the checkpoints during the training. The models are saved inside the `pretrained` directory. E.g., with `--out example`, the model will be saved int `pretrained/model_example.pt`, and all the checkpoint information (optimizer parameters, loss, etc.) will be saved inside `checkpoints/example.pt`. In the same example, train/valid/test directories would be `<data_path>/experiments/train_example`, `<data_path>/experiments/valid_example`, `<data_path>/experiments/test_example`.

For example, if you want to save the data inside `other_dir/data` and call all the output files `example_run`, run:
```bash
python pipeline.py --data other_dir/data --out example_run
```

### 4. Evaluate assemblies
The assembly sequences, obtained for the graphs in the test set, wil lbe stored inside `<data_path>/experiments/test_<out>/assembly` directory. The easiest way to evaluate the obtained assemblies is with [Quast](https://github.com/ablab/quast). For installing Quast, we suggest creating a new conda environment (due to clashing dependencies):
```bash
conda create -n quast python=3.6
conda activate quast
conda install -c bioconda quast
```
The general usage is as follows:
```bash
quast -r <path_to_reference> -o <output_dir> <path_to_assembly_file>
```
For example, if inside `test_example` directory you only have one graph of chromosome 21, then you could evaluate the assembly sequence of that chromosome by running:
```bash
quast -r <ref_path>/chromosomes/chr21.fasta -o <data_path>/experiments/test_example/quast <data_path>/experiments/test_example/assembly/0_assembly.fasta
```
The report, containing all the evaluation metrics, will be located at `<data_path>/experiments/test_example/quast/report.txt`.


## Reference

Full paper can be found on arXiv (PLACEHOLDER).
