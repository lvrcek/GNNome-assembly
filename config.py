# Edit these three dictionaries to specify on which graphs to train/validate/test
# Assemblies will be constructed only for the graphs in the test_dict
#################################################################################
_train_dict = {'chr19': 15}
_valid_dict = {'chr19': 3}
_test_dict  = {f'chr{i}': 1 for i in range(1, 23)} ; _test_dict['chrX'] = 1
#################################################################################

def get_config():
    return {
        'train_dict': _train_dict,
        'valid_dict': _valid_dict,
        'test_dict' : _test_dict
    }

