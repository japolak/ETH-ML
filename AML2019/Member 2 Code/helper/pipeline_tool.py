from sklearn.base import TransformerMixin, BaseEstimator

class PrintXShape(BaseEstimator, TransformerMixin):

    def __init__(self, step_name):
        self.step_name = step_name

    def transform(self, X):
        print('Debuging {}...'.format(self.step_name))
        print('X: {}\n'.format(X.shape))
        return X

    def fit(self, X, y=None, **fit_params):
        return self