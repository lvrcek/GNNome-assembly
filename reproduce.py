import argparse

import pipeline


def untangle_synthetic():
    """
        Predict the scores for all the synthetic chromosomes, given the model trained only on chr19.
        Calculate the prediction-metrics, assemble all the genomes.
        Goal: Show that the model generalizes well to other synthetic chromosomes.
    """
    data_path = f'data'
    ref_path = f'data/references'
    out = 'assembly_synth'
    model_path = f'pretrained_models/model_15xchr19.pt'

    train_dict = {}
    valid_dict = {}
    test_dict = {f'chr{i}': 1 for i in range(1, 23)} ; test_dict['chrX'] = 1
    all_chr = pipeline.merge_dicts(train_dict, valid_dict, test_dict)

    pipeline.file_structure_setup(data_path, ref_path)
    pipeline.download_reference(ref_path)
    pipeline.simulate_reads(data_path, ref_path, all_chr)
    pipeline.generate_graphs(data_path, all_chr)
    train_path, valid_path, test_path = pipeline.train_valid_split(data_path, train_dict, valid_dict, test_dict, out)
    pipeline.predict(test_path, out=out, model_path=model_path)


def untangle_real():
    """
        Predict the scores for all the real chromosomes, given the model trained only on synthetic chr19.
        Calculate the prediction-metrics, assemble all the genomes.
        Goal: Show that the model generalizes well to other real as well chromosomes.
    """
    
    data_path = f'data'
    ref_path = f'data/references'
    out = 'assembly_real'
    model_path = f'pretrained_models/model_15xchr19.pt'

    train_dict = {}
    valid_dict = {}
    test_dict = {f'chr{i}_r': 1 for i in range(1, 23)} ; test_dict['chrX_r'] = 1
    all_chr = pipeline.merge_dicts(train_dict, valid_dict, test_dict)

    pipeline.file_structure_setup(data_path, ref_path)
    pipeline.download_reference(ref_path)
    pipeline.simulate_reads(data_path, ref_path, all_chr)
    pipeline.generate_graphs(data_path, all_chr)
    train_path, valid_path, test_path = pipeline.train_valid_split(data_path, train_dict, valid_dict, test_dict, out)
    pipeline.predict(test_path, out=out, model_path=model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=None, help='Assemble synthetic or real chromosome')
    args = parser.parse_args()
    mode = args.mode
    if mode == 'synth':
        untangle_synthetic()
    elif mode == 'real':
        untangle_real()
    else:
        print(f'Run with either "--mode synth" or "--mode real"!')

