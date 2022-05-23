import argparse
import gzip
import os
import pickle
import subprocess

from tqdm import tqdm
import requests
from Bio import SeqIO

import graph_dataset
import train
import inference
import evaluate


chr_lens = {
    'chr1' : 248387328,
    'chr2' : 242696752,
    'chr3' : 201105948,
    'chr4' : 193574945,
    'chr5' : 182045439,
    'chr6' : 172126628,
    'chr7' : 160567428,
    'chr8' : 146259331,
    'chr9' : 150617247,
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
    'chrX' : 154259566,
}


def change_description(file_path):
    new_fasta = []
    for record in SeqIO.parse(file_path, file_path[-5:]): # 'fasta' for FASTA file, 'fastq' for FASTQ file
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


def merge_dicts(d1, d2, d3={}):
    keys = {*d1, *d2, *d3}
    merged = {key: d1.get(key, 0) + d2.get(key, 0) + d3.get(key, 0) for key in keys}
    return merged


# -1. Set up the data file structure
def file_structure_setup(data_path):
    print(f'SETUP::filesystem:: Create directories for storing data')
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    # if 'references' not in os.listdir(data_path):
    #     os.mkdir(os.path.join(data_path, 'references'))
    #     subprocess.run(f'mkdir CHM13 chromosomes lengths', shell=True, cwd=os.path.join(data_path, 'references'))
    #     subprocess.run(f'cp -r lengths {data_path}/references', shell=True)
    
    ref_path = os.path.abspath('references')
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

    # if 'train' not in os.listdir(data_path):
    #     os.mkdir(os.path.join(data_path, 'train'))
    #     subprocess.run(f'mkdir raw processed info tmp graphia solutions', shell=True, cwd=os.path.join(data_path, 'train'))
    # if 'valid' not in os.listdir(data_path):
    #     os.mkdir(os.path.join(data_path, 'valid'))
    #     subprocess.run(f'mkdir raw processed info tmp graphia solutions', shell=True, cwd=os.path.join(data_path, 'valid'))
    # if 'test' not in os.listdir(data_path):
    #     os.mkdir(os.path.join(data_path, 'test'))
    #     subprocess.run(f'mkdir raw processed info tmp graphia solutions', shell=True, cwd=os.path.join(data_path, 'test'))


# 0. Download the CHM13 if necessary
def download_reference(data_path):
    # data_path = os.path.abspath(data_path)
    ref_path = os.path.abspath('references')
    chm_path = os.path.join(ref_path, 'CHM13')
    chr_path = os.path.join(ref_path, 'chromosomes')
    chm13_url = 'https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/chm13.draft_v1.1.fasta.gz'
    chm13_path = os.path.join(chm_path, 'chm13.draft_v1.1.fasta.gz')

    if len(os.listdir(chm_path)) == 0:
        # Download the CHM13 reference
        # Code for tqdm from: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
        print(f'SETUP::download:: CHM13 not found! Downloading...')
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
        print(f'SETUP::download:: Split CHM13 per chromosome')
        with gzip.open(chm13_path, 'rt') as f:
            for record in SeqIO.parse(f, 'fasta'):
                SeqIO.write(record, os.path.join(chr_path, f'{record.id}.fasta'), 'fasta')


# 1. Simulate the sequences
def simulate_reads(data_path, chr_dict):
    # Dict saying how much of simulated datasets for each chromosome do we need
    # E.g., {'chr1': 4, 'chr6': 2, 'chrX': 4}

    # TODO: This is temporarily, should be done upon setting up the poject repo
    print(f'SETUP::simulate')
    if 'seqrequester' not in os.listdir('vendor'):
        print(f'SETUP::simulate:: Download seqrequester')
        subprocess.run(f'git clone https://github.com/marbl/seqrequester.git', shell=True, cwd='vendor')
        subprocess.run(f'make', shell=True, cwd='vendor/seqrequester/src')

    data_path = os.path.abspath(data_path)
    ref_path = os.path.abspath('references')
    chr_path = os.path.join(ref_path, 'chromosomes')
    len_path = os.path.join(ref_path, 'lengths')
    sim_path = os.path.join(data_path, 'simulated')
    for chrN, n_need in chr_dict.items():
        if '_r' in chrN:
            continue
        chr_raw_path = os.path.join(sim_path, f'{chrN}/raw')
        n_have = len(os.listdir(chr_raw_path))
        if n_need <= n_have:
            continue
        else:
            n_diff = n_need - n_have
            print(f'SETUP::simulate:: Simulate {n_diff} datasets for {chrN}')
            # Simulate reads for chrN n_diff times
            # Be careful how you name them
            chr_seq_path = os.path.join(chr_path, f'{chrN}.fasta')
            # chr_dist_path = os.path.join(len_path, f'{chrN}.txt')
            chr_dist_path = os.path.join('lengths', f'{chrN}.txt')
            chr_len = chr_lens[chrN]
            for i in range(n_diff):
                idx = n_have + i
                chr_save_path = os.path.join(chr_raw_path, f'{idx}.fasta')
                print(f'\nStep {i}: Simulating reads {chr_save_path}')
                subprocess.run(f'./vendor/seqrequester/build/bin/seqrequester simulate -genome {chr_seq_path} ' \
                               f'-genomesize {chr_len} -coverage 32.4 -distribution {chr_dist_path} > {chr_save_path}',
                               shell=True)
                change_description(chr_save_path)


# 2. Generate the graphs
def generate_graphs(data_path, chr_dict):
    print(f'SETUP::generate')

    if 'raven' not in os.listdir('vendor'):
        print(f'SETUP::generate:: Download Raven')
        subprocess.run(f'git clone -b print_graphs https://github.com/lbcb-sci/raven', shell=True, cwd='vendor')
        subprocess.run(f'cmake -S ./ -B./build -DRAVEN_BUILD_EXE=1 -DCMAKE_BUILD_TYPE=Release', shell=True, cwd='vendor/raven')
        subprocess.run(f'cmake --build build', shell=True, cwd='vendor/raven')

    data_path = os.path.abspath(data_path)

    for chrN, n_need in chr_dict.items():
        if '_r' in chrN:
            continue
        chr_sim_path = os.path.join(data_path, 'simulated', f'{chrN}')
        chr_raw_path = os.path.join(chr_sim_path, 'raw')
        chr_prc_path = os.path.join(chr_sim_path, 'processed')
        n_raw = len(os.listdir(chr_raw_path))
        n_prc = len(os.listdir(chr_prc_path))
        n_diff = n_raw - n_prc
        print(f'SETUP::generate:: Generate {n_diff} graphs for {chrN}')
        specs = {
            'threads': 32,
            'filter': 0.99,
            'out': 'assembly.fasta'
        }
        graph_dataset.AssemblyGraphDataset(chr_sim_path, nb_pos_enc=None, specs=specs, generate=True)
        # Generate graphs for those reads that don't have them
        # Probably something with Raven
        # Then the graph_parser


# 2.1. Generate the real_graphs
def generate_graphs_real(data_path, chr_real_list):
    print(f'SETUP::generate')

    if 'raven' not in os.listdir('vendor'):
        print(f'SETUP::generate:: Download Raven')
        subprocess.run(f'git clone -b print_graphs https://github.com/lbcb-sci/raven', shell=True, cwd='vendor')
        subprocess.run(f'cmake -S ./ -B./build -DRAVEN_BUILD_EXE=1 -DCMAKE_BUILD_TYPE=Release', shell=True, cwd='vendor/raven')
        subprocess.run(f'cmake --build build', shell=True, cwd='vendor/raven')

    data_path = os.path.abspath(data_path)
    for chrN in chr_real_list:
        chr_sim_path = os.path.abspath(data_path, 'real', f'{chrN}')
        chr_raw_path = os.path.join(chr_sim_path, 'raw')
        chr_prc_path = os.path.join(chr_sim_path, 'processed')
        n_raw = len(os.listdir(chr_raw_path))
        n_prc = len(os.listdir(chr_prc_path))
        n_diff = n_raw - n_prc
        print(f'SETUP::generate:: Generate {n_diff} graphs for {chrN}')
        specs = {
            'threads': 32,
            'filter': 0.99,
            'out': 'assembly.fasta'
        }
        graph_dataset.AssemblyGraphDataset(chr_sim_path, nb_pos_enc=None, specs=specs, generate=True)
        # Generate graphs for those reads that don't have them
        # Probably something with Raven
        # Then the graph_parser


# 2.5 Train-valid-test split
def train_valid_split(data_path, train_dict, valid_dict, test_dict={}, out=None):
    #Both are chromosome dicts specifying which data to use for training/validation
    print(f'SETUP::split')
    data_path = os.path.abspath(data_path)
    sim_path = os.path.join(data_path, 'simulated')
    real_path = os.path.join(data_path, 'real')

    if out is None:
        train_path = os.path.join(data_path, f'train')
        valid_path = os.path.join(data_path, f'valid')
        test_path  = os.path.join(data_path, f'test')
    else:
        train_path = os.path.join(data_path, f'train_{out}')
        valid_path = os.path.join(data_path, f'valid_{out}')
        test_path  = os.path.join(data_path, f'test_{out}')
        if not os.path.isdir(train_path):
            os.mkdir(train_path)
            subprocess.run(f'mkdir raw processed info graphia tmp', shell=True, cwd=train_path)
        if not os.path.isdir(valid_path):
            os.mkdir(valid_path)
            subprocess.run(f'mkdir raw processed info graphia tmp', shell=True, cwd=valid_path)
        if not os.path.isdir(test_path) and len(test_dict) > 0:
            os.mkdir(test_path)
            subprocess.run(f'mkdir raw processed info graphia tmp', shell=True, cwd=test_path)
 
    train_g_to_chr = {}  # Remember chromosomes for each graph in the dataset
    train_g_to_org_g = {}  # Remember index of the graph in the master dataset for each graph in this dataset
    n_have = 0
    for chrN, n_need in train_dict.items():
        # copy n_need datasets from chrN into train dict
        print(f'SETUP::split:: Copying {n_need} graphs of {chrN} into {train_path}')
        for i in range(n_need):
            train_g_to_chr[n_have] = chrN
            chr_sim_path = os.path.join(sim_path, chrN)
            print(f'Copying {chr_sim_path}/processed/{i}.dgl into {train_path}/processed/{n_have}.dgl')
            subprocess.run(f'cp {chr_sim_path}/processed/{i}.dgl {train_path}/processed/{n_have}.dgl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{i}_succ.pkl {train_path}/info/{n_have}_succ.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{i}_pred.pkl {train_path}/info/{n_have}_pred.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{i}_edges.pkl {train_path}/info/{n_have}_edges.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{i}_reads.pkl {train_path}/info/{n_have}_reads.pkl', shell=True)
            # subprocess.run(f'cp {chr_sim_path}/solutions/{i}_edges.pkl {train_path}/solutions/{n_have}_edges.pkl', shell=True)
            train_g_to_org_g = {n_have: i}
            n_have += 1
    pickle.dump(train_g_to_chr, open(f'{train_path}/info/g_to_chr.pkl', 'wb'))
    pickle.dump(train_g_to_org_g, open(f'{train_path}/info/g_to_org_g.pkl', 'wb'))

    valid_g_to_chr = {}
    valid_g_to_org_g = {}
    n_have = 0
    for chrN, n_need in valid_dict.items():
        # copy n_need datasets from chrN into train dict
        print(f'SETUP::split:: Copying {n_need} graphs of {chrN} into {valid_path}')
        for i in range(n_need):
            valid_g_to_chr[n_have] = chrN
            j = i + train_dict.get(chrN, 0)
            chr_sim_path = os.path.join(sim_path, chrN)
            print(f'Copying {chr_sim_path}/processed/{j}.dgl into {valid_path}/processed/{n_have}.dgl')
            subprocess.run(f'cp {chr_sim_path}/processed/{j}.dgl {valid_path}/processed/{n_have}.dgl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{j}_succ.pkl {valid_path}/info/{n_have}_succ.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{j}_pred.pkl {valid_path}/info/{n_have}_pred.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{j}_edges.pkl {valid_path}/info/{n_have}_edges.pkl', shell=True)
            subprocess.run(f'cp {chr_sim_path}/info/{j}_reads.pkl {valid_path}/info/{n_have}_reads.pkl', shell=True)
            # subprocess.run(f'cp {chr_sim_path}/solutions/{j}_edges.pkl {valid_path}/solutions/{n_have}_edges.pkl', shell=True)
            valid_g_to_org_g = {n_have: j}
            n_have += 1
    pickle.dump(valid_g_to_chr, open(f'{valid_path}/info/g_to_chr.pkl', 'wb'))
    pickle.dump(valid_g_to_org_g, open(f'{valid_path}/info/g_to_org_g.pkl', 'wb'))

    if test_dict: 
        test_g_to_chr = {}
        test_g_to_org_g = {}
        n_have = 0
        for chrN, n_need in test_dict.items():
            # copy n_need datasets from chrN into train dict
            print(f'SETUP::split:: Copying {n_need} graphs of {chrN} into {test_path}')
            for i in range(n_need):
                if '_r' in chrN:
                    chrN = chrN[:-2]
                    chr_sim_path = os.path.join(real_path, chrN)
                    k = i
                else:
                    chr_sim_path = os.path.join(sim_path, chrN)
                    k = i + train_dict.get(chrN, 0) + valid_dict.get(chrN, 0)
                test_g_to_chr[n_have] = chrN
                # k = i + train_dict.get(chrN, 0) + valid_dict.get(chrN, 0)
                # chr_sim_path = os.path.join(sim_path, chrN)
                print(f'Copying {chr_sim_path}/processed/{k}.dgl into {test_path}/processed/{n_have}.dgl')
                subprocess.run(f'cp {chr_sim_path}/processed/{k}.dgl {test_path}/processed/{n_have}.dgl', shell=True)
                subprocess.run(f'cp {chr_sim_path}/info/{k}_succ.pkl {test_path}/info/{n_have}_succ.pkl', shell=True)
                subprocess.run(f'cp {chr_sim_path}/info/{k}_pred.pkl {test_path}/info/{n_have}_pred.pkl', shell=True)
                subprocess.run(f'cp {chr_sim_path}/info/{k}_edges.pkl {test_path}/info/{n_have}_edges.pkl', shell=True)
                subprocess.run(f'cp {chr_sim_path}/info/{k}_reads.pkl {test_path}/info/{n_have}_reads.pkl', shell=True)
                # subprocess.run(f'cp {chr_sim_path}/solutions/{j}_edges.pkl {valid_path}/solutions/{n_have}_edges.pkl', shell=True)
                n_have += 1
                test_g_to_org_g = {n_have: k}
        pickle.dump(test_g_to_chr, open(f'{test_path}/info/g_to_chr.pkl', 'wb'))
        pickle.dump(test_g_to_org_g, open(f'{test_path}/info/g_to_org_g.pkl', 'wb'))

    return train_path, valid_path, test_path


# 3. Train the model
def train_the_model(data, out, eval, overfit):
    print(f'SETUP::train')
    train.train(data, out, eval, overfit)


# 4. Inference - get the results
def predict(data_path, out, model_path=None, device='cpu'):
    if model_path is None:
        model_path = os.path.abspath(f'pretrained/model_{out}.pt')
    walks_per_graph, contigs_per_graph, walks_per_graph_ol, contigs_per_graph_ol, walks_per_graph_lab, contigs_per_graph_lab = inference.inference(data_path, model_path, device)
    g_to_chr = pickle.load(open(f'{data_path}/info/g_to_chr.pkl', 'rb'))
    for idx, contigs in enumerate(contigs_per_graph):
        chrN = g_to_chr[idx]
        num_contigs, longest_contig, reconstructed, n50, ng50 = evaluate.quick_evaluation(contigs, chrN)
        evaluate.print_summary(data_path, idx, chrN, num_contigs, longest_contig, reconstructed, n50, ng50)
    # return
    ############################
    for idx, contigs in enumerate(contigs_per_graph_ol):
        chrN = g_to_chr[idx]
        num_contigs, longest_contig, reconstructed, n50, ng50 = evaluate.quick_evaluation(contigs, chrN)
        print(f'ol_len: {idx=} {num_contigs=} {longest_contig=} {reconstructed=:.4f} {n50=} {ng50=}')
    for idx, contigs in enumerate(contigs_per_graph_lab):
        chrN = g_to_chr[idx]
        num_contigs, longest_contig, reconstructed, n50, ng50 = evaluate.quick_evaluation(contigs, chrN)
        print(f'labels: {idx=} {num_contigs=} {longest_contig=} {reconstructed=:.4f} {n50=} {ng50=}')
    ############################


def nips_exp1():
    """
        Predict the scores for all the synthetic chromosomes, given the model trained only on chr19.
        Calculate the prediction-metrics, don't assemble the genomes.
        Goal: Show that the model generalizes well to other synthetic chromosomes.
    """
    data_path = f'/home/vrcekl/scratch/nips_2022/experiments/valid_ALL-CHR'
    model_path = f'nips_submit/model_12-05_15v3-chr19_shuffle.pt'
    inference.inference(data_path, model_path)


def nips_exp1a():
    """
        Predict the scores for all the synthetic chromosomes, given the model trained only on chr19.
        Calculate the prediction-metrics, assemble all the genomes.
        Goal: Show that the model generalizes well to other synthetic chromosomes.
    """
    data_path = f'/home/vrcekl/scratch/nips_2022/experiments/valid_ALL-CHR'
    model_path = f'nips_submit/model_12-05_15v3-chr19_shuffle.pt'
    # model_path = f'nips_submit/model_18-05_15v3-mix_c91922_EXP6.pt'
    predict(data_path, '.', model_path)



def nips_exp2():
    """
        Reconstruct the real chr19, given the model trained only on synthetic chr19.
        Calculate the metrics AND assembly the genome.
        Goal: Show that the model generalizes from synthetic to real data.
    """
    data_path = f'/home/vrcekl/scratch/nips_2022/experiments/model_vs_raven/real/17-05/chr19'
    model_path = f'nips_submit/model_12-05_15v3-chr19_shuffle.pt'
    predict(data_path, '.', model_path, device='cuda:3')


def nips_exp3():
    """
        Predict the scores for all the real chromosomes, given the model trained only on synthetic chr19.
        Calculate the prediction-metrics, assemble all the genomes.
        Goal: Show that the model generalizes well to other real as well chromosomes.
    """
    for i in range(1, 24):
        if i == 23:
            i = 'X'
        data_path = f'/home/vrcekl/scratch/nips_2022/experiments/real/chr{i}/'
        model_path = f'nips_submit/model_12-05_15v3-chr19_shuffle.pt'
        predict(data_path, '.', model_path)



def nips_exp3_mix():
    """
        Predict the scores for all the real chromosomes, given the model trained only on synthetic chr19.
        Calculate the prediction-metrics, assemble all the genomes.
        Goal: Show that the model generalizes well to other real as well chromosomes.
    """
    for i in range(1, 24):
        if i == 23:
            i = 'X'
        data_path = f'/home/vrcekl/scratch/nips_2022/experiments/real/chr{i}/'
        # model_path = f'nips_submit/model_18-05_15v3-mix_EXP2.pt'
        model_path = f'nips_submit/model_18-05_15v3-mix_c91922_EXP6.pt'
        predict(data_path, '.', model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/train', help='Path to directory with training data')
    parser.add_argument('--out', type=str, default=None, help='Output name for figures and models')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--overfit', action='store_true', default=False, help='Overfit on the chromosomes in the train directory')
    args = parser.parse_args()
    # Either get all the arguments
    # Or just reads everything from some config file
    # E.g., train = {chr1: 1, chr4: 3, chr5: 5}
    # E.g., eval = {chr6: 2, chr5: 3}

    # train_dict = {'chr19': 15}
    # valid_dict = {'chr19': 3}
    train_dict = {'chr18': 5, 'chr19': 5, 'chr20': 5}
    valid_dict = {'chr19': 3}
    test_dict  = {'chr19': 1}

    # ALL Chromosomes
    # train_dict = {'chr19': 1}
    # valid_dict = {f'chr{i}': 1 for i in range(1, 23)} ; valid_dict['chrX'] = 1

    all_chr = merge_dicts(train_dict, valid_dict, test_dict)

    data = args.data
    out = args.out
    eval = args.eval
    overfit = args.overfit

    data_path = data

    # nips_exp3_mix()
    nips_exp1a()
    exit(1)

    file_structure_setup(data_path)
    download_reference(data_path)
    simulate_reads(data_path, all_chr)
    generate_graphs(data_path, all_chr)
    train_path, valid_path, test_path = train_valid_split(data_path, train_dict, valid_dict, test_dict, out)
    train_the_model(data, out, eval, overfit)
    
    # test_path = '/home/user/scratch/nips_2022/experiments/model_vs_raven/real/12-05_v2/chr18'
    # model_path = f'pretrained/model_10-05_chr19-15v3v2-exp-1.pt'
    # out = 'out'
    # predict(test_path, out, model_path, device='cpu')

