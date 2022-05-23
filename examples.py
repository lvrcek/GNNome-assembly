from pipeline import predict


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

