################################################################################

# Edit these three dictionaries to specify graphs to train/validation/test
# Assemblies will be constructed only for the graphs in the test_dict

# To train/validate/test on multiple chromosomes, put the as separate
# entries in the dictionaries
# E.g., to train on 1 chr19 graph and 2 chr20 graphs: 
# _train_dict = {'chr19': 1, 'chr20': 2}

# To test on real chromosome put "_r" suffix. Don't put value higher than 1,
# since there is only 1 real HiFi dataset for each chromosomes
# E.g., to test on real chr21:
# _test_dict = {'chr21_r': 1}

_train_dict = {'chr19': 5}
_valid_dict = {'chr19': 2}
_test_dict = {'chr21': 1}

################################################################################

def get_config():
    return {
        'train_dict': _train_dict,
        'valid_dict': _valid_dict,
        'test_dict' : _test_dict
    }

