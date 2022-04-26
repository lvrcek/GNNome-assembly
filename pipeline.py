import os
import subprocess

import requests
from Bio import SeqIO


def create_chr_dirs(pth):
    for i in range(1, 24):
        if i == 23:
            i = 'X'
        subprocess.run(f'mkdir chr{i}', shell=True, cwd=pth)
        subprocess.run(f'mkdir raw processed info tmp graphia solutions', shell=True, cwd=os.path.join(pth, f'chr{i}'))

# -1. Set up the data file structure
def file_structure_setup(data_path):
    if 'references' not in os.listdir(data_path):
        os.mkdir(os.path.join(data_path, 'references'))
        subprocess.run(f'mkdir CHM13 chromosomes lengths', shell=True, cwd=os.mkdir(os.path.join(data_path, 'references')))
    else:
        ref_path = os.path.join(data_path, 'references')
        if 'CHM13' not in os.listdir(ref_path):
            os.mkdir(os.path.join(ref_path, 'CHM13'))
        if 'chromosomes' not in os.listdir(ref_path):
            os.mkdir(os.path.join(ref_path, 'chromosomes'))
            
    if 'simulated' not in os.listdir(data_path):
        os.mkdir(os.path.join(data_path, 'simulated'))
        create_chr_dirs(os.path.join(data_path, 'simulated'))
    if 'real' not in os.listdir(data_path):
        os.mkdir(os.path.join(data_path, 'real'))
        create_chr_dirs(os.path.join(data_path, 'real'))

    if 'train' not in os.listdir(data_path):
        os.mkdir(os.path.join(data_path, 'train'))
        subprocess.run(f'mkdir raw processed info tmp graphia solutions', shell=True, cwd=os.path.join(data_path, 'train'))
    if 'valid' not in os.listdir(data_path):
        os.mkdir(os.path.join(data_path, 'valid'))
        subprocess.run(f'mkdir raw processed info tmp graphia solutions', shell=True, cwd=os.path.join(data_path, 'valid'))
    if 'test' not in os.listdir(data_path):
        os.mkdir(os.path.join(data_path, 'test'))
        subprocess.run(f'mkdir raw processed info tmp graphia solutions', shell=True, cwd=os.path.join(data_path, 'test'))


# 0. Download the CHM13 if necessary
def download_reference():
    ref_path = os.path.abspath('data/neurips/references/')
    chm_path = os.path.join(ref_path, 'CHM13')
    chr_path = os.path.join(ref_path, 'chromosomes')
    chm13_url = 'https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/chm13.draft_v1.1.fasta.gz'
    if len(os.listdir(chm_path)) == 0:
        # Download the CHM13 reference
        response = requests.get(chm13_url)
    if len(os.listdir(chr_path)) == 0:
        # Parse the CHM13 into individual chromosomes
        chm_path = os.path.join(chm_path, 'chm13.draft_v1.1.fasta.gz')
        for record in SeqIO.parse(chm_path, 'fasta'):
            SeqIO.write(record, os.path.join(chr_path, f'{record.id}.fasta'), 'fasta')


# 1. Simulate the sequences
def simualate_reads(chr_dict):
    # Dict saying how much of simulated datasets for each chromosome do we need
    # E.g., {'chr1': 4, 'chr6': 2, 'chrX': 4}
    ref_path = os.path.abspath('data/neurips/references/')
    chr_path = os.path.join(ref_path, 'chromosomes')
    len_path = os.path.join(ref_path, 'lengths')
    sim_path = os.path.abspath('data/neurips/simulated')
    for chrN, n_need in chr_dict.items():
        chr_raw_path = os.path.joins(f'sim_path/{chrN}/raw')
        n_have = len(os.listdir(chr_raw_path))
        if n_need <= n_have:
            continue
        else:
            n_diff = n_need - n_have
            # Simulate reads for chrN n_diff times
            # Be careful how you name them
            ...    


# 2. Generate the graphs
def generate_graphs(chr_dict):
    for chrN, n_need in chr_dict.items():
        chr_sim_path = os.path.abspath(f'data/neurips/simulated/{chrN}')
        chr_raw_path = os.path.join(chr_sim_path, 'raw')
        chr_prc_path = os.path.join(chr_sim_path, 'processed')
        n_raw = len(os.listdir(chr_raw_path))
        n_prc = len(os.listdir(chr_prc_path))
        if n_prc < n_raw:
            n_diff = n_raw - n_prc
            # Generate graphs for those reads that don't have them
            # Probably something with Raven
            # Then the graph_parser
            ...


# 2.5 Train-valid-test split
def train_valid_split(train_dict, valid_dict):
    #Both are chromosome dicts specifying which data to use for training/validation
    sim_path = os.path.abspath('data/neurips/simulated/')
    train_path = os.path.abspath('data/neurips/train')
    valid_path = os.path.abspath('data/neurips/valid')
    
    n_have = 0
    for chrN, n_need in train_dict.items():
        # copy n_need datasets from chrN into train dict
        for i in range(n_need):
            chr_sim_path = os.path.join(sim_path, chrN)
            subprocess.run(f'cp {chr_sim_path}/processed/{i}.dgl {train_path}/processed/{n_have}.dgl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{i}_succ.pkl {train_path}/info/{n_have}_succ.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{i}_pred.pkl {train_path}/info/{n_have}_pred.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{i}_edges.pkl {train_path}/info/{n_have}_edges.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/solutions/{i}_edges.pkl {train_path}/solutions/{n_have}_edges.pkl', shell=True)
            n_have += 1

    n_have = 0
    for chrN, n_need in valid_dict.items():
        # copy n_need datasets from chrN into train dict
        for i in range(n_need):
            chr_sim_path = os.path.join(sim_path, chrN)
            subprocess.run(f'cp {chr_sim_path}/processed/{i}.dgl {valid_path}/processed/{n_have}.dgl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{i}_succ.pkl {valid_path}/info/{n_have}_succ.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{i}_pred.pkl {valid_path}/info/{n_have}_pred.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{i}_edges.pkl {valid_path}/info/{n_have}_edges.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/solutions/{i}_edges.pkl {valid_path}/solutions/{n_have}_edges.pkl', shell=True)
            n_have += 1



# 3. Train the model
# I have already prepared the datasets here
# I just need to run the training with the correct arguments/parameters
def train_the_model():
    train()

# 4. Inference - get the results

# 5. Save the assembly

# 6. Evaluate the assembly (num_contigs, NG50))

# 7. Print out the report


if __name__ == '__main__':
    # Either get all the arguments
    # Or just reads everything from some config file
    # E.g., train = {chr1: 1, chr4: 3, chr5: 5}
    # E.g., eval = {chr6: 2, chr5: 3}

