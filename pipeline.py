import os


# 0. Download the CHM13 if necessary
def download_reference():
    ref_path = os.path.abspath('data/neurips/references/')
    chm_path = os.path.join(ref_path, 'CHM13')
    chr_path = os.path.join(ref_path, 'chromosomes')
    if len(os.listdir(chm_path)) == 0:
        # Download the CHM13 reference
        ...
    if len(os.listdir(chr_path)) == 0:
        # Parse the CHM13 into individual chromosomes
        ...


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

