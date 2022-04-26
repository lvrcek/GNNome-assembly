import argparse
import gzip
import os
import subprocess

from tqdm import tqdm
import requests
from Bio import SeqIO

import graph_dataset
import train

chr_lens = {
'chr1': 248387328,
'chr2': 242696752,
'chr3': 201105948,
'chr4': 193574945,
'chr5': 182045439,
'chr6': 172126628,
'chr7': 160567428,
'chr8': 146259331,
'chr9': 150617247,
'chr10': 134758134,
'chr11': 135127769,
'chr12': 133324548,
'chr13': 113566686,
'chr14': 101161492,
'chr15': 99753195,
'chr16': 96330374,
'chr17': 84276897,
'chr18': 80542538,
'chr19': 61707364,
'chr20': 66210255,
'chr21': 45090682,
'chr22': 51324926,
'chrX': 154259566,
}


def change_description(file_path):
    new_fasta = []
    for record in SeqIO.parse(file_path, file_path[-5:]): # path[-5:] is fasta for fasta file, anf fastq for fastq file
        des = record.description.split(",")
        id = des[0][5:]
        if des[1] == "forward":
            strand = '+'
        else:
            strand = '-'
        position = des[2][9:].split("-")
        start = position[0]
        end = position[1]
        record.id = id
        record.description = f'strand={strand}, start={start}, end={end}'
        new_fasta.append(record)
    SeqIO.write(new_fasta, file_path, "fasta")


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
    chm13_path = os.path.join(chm_path, 'chm13.draft_v1.1.fasta.gz')

    if len(os.listdir(chm_path)) == 0:
        # Download the CHM13 reference
        # Code for tqdm from: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
        response = requests.get(chm13_url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

        with open(chm13_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

    if len(os.listdir(chr_path)) == 0:
        # Parse the CHM13 into individual chromosomes
        with gzip.open(chm13_path, 'rt') as f:
            for record in SeqIO.parse(f, 'fasta'):
                SeqIO.write(record, os.path.join(chr_path, f'{record.id}.fasta'), 'fasta')


# 1. Simulate the sequences
def simulate_reads(chr_dict):
    # Dict saying how much of simulated datasets for each chromosome do we need
    # E.g., {'chr1': 4, 'chr6': 2, 'chrX': 4}

    # TODO: This is like this temporarily, should be done upon setting up the poject repo
    if 'seqrequester' not in os.listdir('vendor'):
        subprocess.run(f'git clone https://github.com/marbl/seqrequester.git', shell=True, cwd='vendor')
        subprocess.run(f'make', shell=True, cwd='vendor/seqrequester/src')

    ref_path = os.path.abspath('data/neurips/references/')
    chr_path = os.path.join(ref_path, 'chromosomes')
    len_path = os.path.join(ref_path, 'lengths')
    sim_path = os.path.abspath('data/neurips/simulated')
    for chrN, n_need in chr_dict.items():
        chr_raw_path = os.path.join(sim_path, f'{chrN}/raw')
        n_have = len(os.listdir(chr_raw_path))
        if n_need <= n_have:
            continue
        else:
            n_diff = n_need - n_have
            # Simulate reads for chrN n_diff times
            # Be careful how you name them
            chr_seq_path = os.path.join(chr_path, f'{chrN}.fasta')
            chr_dist_path = os.path.join(len_path, f'{chrN}.txt')
            chr_len = chr_lens[chrN]
            for i in range(n_diff):
                idx = n_have + i
                chr_save_path = os.path.join(chr_raw_path, f'{idx}.fasta')
                subprocess.run(f'./vendor/seqrequester/build/bin/seqrequester simulate -genome {chr_seq_path} ' \
                               f'-genomesize {chr_len} -coverage 32.4 -distribution {chr_dist_path} > {chr_save_path}',
                               shell=True)
                change_description(chr_save_path)



# 2. Generate the graphs
def generate_graphs(chr_dict):
    for chrN, n_need in chr_dict.items():
        chr_sim_path = os.path.abspath(f'data/neurips/simulated/{chrN}')
        chr_raw_path = os.path.join(chr_sim_path, 'raw')
        chr_prc_path = os.path.join(chr_sim_path, 'processed')
        n_raw = len(os.listdir(chr_raw_path))
        n_prc = len(os.listdir(chr_prc_path))
        n_diff = n_raw - n_prc
        graph_dataset.AssemblyGraphDataset(chr_sim_path)
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
            j = i + train_dict[chrN]
            chr_sim_path = os.path.join(sim_path, chrN)
            subprocess.run(f'cp {chr_sim_path}/processed/{j}.dgl {valid_path}/processed/{n_have}.dgl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{j}_succ.pkl {valid_path}/info/{n_have}_succ.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{j}_pred.pkl {valid_path}/info/{n_have}_pred.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{j}_edges.pkl {valid_path}/info/{n_have}_edges.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/solutions/{j}_edges.pkl {valid_path}/solutions/{n_have}_edges.pkl', shell=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/train', help='Path to directory with training data')
    parser.add_argument('--out', type=str, default=None, help='Output name for figures and models')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--split', action='store_true', default=False, help='Is the dataset already split into train/valid/test')
    args = parser.parse_args()
    # Either get all the arguments
    # Or just reads everything from some config file
    # E.g., train = {chr1: 1, chr4: 3, chr5: 5}
    # E.g., eval = {chr6: 2, chr5: 3}
    simulate_reads({'chr19': 5})
    generate_graphs({'chr19': 5})
    train_valid_split({'chr19': 3}, {'chr19': 2})
    train.train(args)
