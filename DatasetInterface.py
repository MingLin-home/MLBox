'''
Dataset interface template file.
'''

class Dataset:
    '''
    Template for Dataset classes
    '''
    def __init__(self, option_str):
        '''
        Initialize dataset class.
        :param option_str: option strings to specifiy dataset parameters
        '''

        pass
    pass

    def get_cv_id_list(self):
        '''
        Get the cross-validation id list
        :return: list of string or interger
        '''
        return ['full_train',0,1,2,3,4,] # five fold cv and one full training set
    pass # end def

    def get_repeat_id_list(self):
        '''
        Get the repeat id list
        :return:
        '''
        return range(10) # by default repeat 10 times
    pass # end def

    def get_dataset_name_list(self):
        '''

        :return: List of dataset name that can be loaded
        '''
        return ['example_dataset', 'that must be implemeted here']
    pass

    def get_name(self):
        return 'EmpytDatasetName'
    pass

    def load_dataset(self, set_name):
        '''
        Load train or test or validation dataset
        :param set_name: could be 'train', 'test', 'val'
        :return: X,Y
        '''
        return None, None

    pass

pass # end if