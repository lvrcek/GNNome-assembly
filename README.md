# Genome Reconstruction with GatedGCN

## Installation
Download the code and run `pip install -r requirements.txt`.

## Reproducibilty
Download the `test_model.zip` from the supplementary materials and unpack it so that `data` and `pretrained` directories are
immediately inside the project directory.

In order to reproduce the training, run:
```bash
python train.py.
```
The default data path will be `data/train`, and the default name for the trained model will be current timestamp.

**Note**: It is advisable to run the following script on native Linux.
Some issues with importing DGL were noticed while using WSL. We are working on fixing these issues.

In order to use the pretrained model for reproducing the reconstructed lengths,
we provide a script `reproduce.py`.
For reproducing the length analysis on 2 Mbp, run:
```bash
python reproduce.py 2
```
For reproducing the length analysis on 5 Mbp, run:
```bash
python reproduce.py 5
```
For reproducing the length analysis on 10 Mbp, run:
```bash
python reproduce.py 10
```

Reproducing the Raven results not possible at this moment, because the reads have over 20 GB in total 
so it wasn't possible to upload them.