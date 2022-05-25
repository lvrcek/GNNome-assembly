import pipeline


def example():
    """
        Predict the scores for all the synthetic chromosomes, given the model trained only on chr19.
        Calculate the prediction-metrics, assemble all the genomes.
        Goal: Show that the model generalizes well to other synthetic chromosomes.
    """
    data_path = f'data'
    ref_path = f'data/references'
    out = 'example'

    train_dict = {'chr19': 3}
    valid_dict = {'chr19': 1}
    test_dict = {'chr21': 1}
    all_chr = pipeline.merge_dicts(train_dict, valid_dict, test_dict)

    pipeline.file_structure_setup(data_path, ref_path)
    pipeline.download_reference(ref_path)
    pipeline.simulate_reads(data_path, ref_path, all_chr)
    pipeline.generate_graphs(data_path, all_chr)
    train_path, valid_path, test_path = pipeline.train_valid_split(data_path, train_dict, valid_dict, test_dict, out)
    pipeline.train_model(train_path, valid_path, out, False)
    pipeline.predict(test_path, out=out)


if __name__ == '__main__':
    example()

