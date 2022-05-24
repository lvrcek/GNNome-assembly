import pipeline

def exp1():
    """
        Predict the scores for all the synthetic chromosomes, given the model trained only on chr19.
        Calculate the prediction-metrics, assemble all the genomes.
        Goal: Show that the model generalizes well to other synthetic chromosomes.
    """
    data_path = f'data/examples'
    out = 'example_1'
    train_dict = {'chr19': 3}
    valid_dict = {'chr19': 1}
    test_dict = {'chr21': 1}
    all_chr = pipeline.merge_dicts(train_dict, valid_dict, test_dict)
    pipeline.file_structure_setup(data_path)
    pipeline.download_reference(data_path)
    pipeline.simulate_reads(data_path, all_chr)
    pipeline.generate_graphs(data_path, all_chr)
    train_path, valid_path, test_path = pipeline.train_valid_split(data_path, train_dict, valid_dict, test_dict, out)
    # pipeline.train_the_model(data_path, out, False)
    pipeline.predict(test_path, out=out)


def nips_exp2():
    """
        Reconstruct the real chr19, given the model trained only on synthetic chr19.
        Calculate the metrics AND assembly the genome.
        Goal: Show that the model generalizes from synthetic to real data.
    """
    data_path = f'/home/vrcekl/scratch/nips_2022/experiments/model_vs_raven/real/17-05/chr19'
    model_path = f'nips_submit/model_12-05_15v3-chr19_shuffle.pt'
    pipeline.predict(data_path, '.', model_path, device='cuda:3')


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
        pipeline.predict(data_path, '.', model_path)



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
        pipeline.predict(data_path, '.', model_path)


if __name__ == '__main__':
    exp1()
