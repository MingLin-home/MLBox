'''
Estimator interface template file.
'''

class Estimator:
    '''
    Template for Dataset classes
    '''
    def __init__(self, option_str):
        '''
        Initialize estimator class.
        :param option_str: option strings to specifiy estimator parameters
        '''

        pass
    pass

    def get_name(self):
        return 'EmptyEstimator'
    pass

    def fit(self,X,Y):
        pass
    pass

    def predict(self, X):
        pass
    pass

    def save(self, save_file_name):
        pass
    pass
pass # end if