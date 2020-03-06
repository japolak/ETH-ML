import os
import sys
dir_cur = os.path.abspath(os.path.dirname(__file__))
dir_par = os.path.dirname(dir_cur)
sys.path.insert(0, dir_par)
from helper.tuning import tune
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV, GridSearchCV
import sklearn

class LocalOutlierTransformer(TransformerMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def transform(self, X):
        lcf = LocalOutlierFactor(**self.kwargs)
        outliers = lcf.fit_predict(X)
        return X[outliers == 1]

    def fit(self, *args, **kwargs):
        return self


X = pd.read_csv("data/X_train.csv", index_col=0).values
y = pd.read_csv('data/y_train.csv', index_col=0).values
y = sklearn.utils.validation.column_or_1d(y)
X_test = pd.read_csv('data/X_test.csv', index_col=0).values

data_pre = Pipeline([
    ('sim', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

feature_kbest = Pipeline([
    ('var', VarianceThreshold(0.05)),
    ('kbest', SelectKBest(f_regression, k=200))
])

pipeline = Pipeline([
    ('data_pre', data_pre),
    ('feature_extract', feature_kbest),
    ('scaler', StandardScaler()),
    # ('rf', RandomForestRegressor(n_estimators=500, max_depth=15, n_jobs=-1)),
    ('lgbm', LGBMRegressor(verbose=0, num_leaves=36, learning_rate=0.05,
                           feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=5)),
])


def main():
    pipeline.fit(X, y)
    pred = pipeline.predict(X_test)
    np.savetxt('submission.csv', np.dstack((np.arange(0, pred.size), pred))[0], '%0.1f,%f',
               comments='', header="id,y")


param_grid = {
    'data_pre__sim__strategy': ['mean', 'median'],
    'feature_extract__var__threshold': [0.01, 0.05],
    'feature_extract__kbest__k': np.linspace(5, 400, num=5, dtype=np.int),
}

param_lgbm = {
    'num_leaves': np.linspace(5, 10, num=2, dtype=np.int),
    # 'learning_rate': [1e-3, 1e-2, 1e-1, 0.05, 0.5],
    # 'feature_fraction': np.linspace(0.2, 0.8, num=4, dtype=np.float),
    # 'bagging_fraction': np.linspace(0.2, 0.8, num=4, dtype=np.float),
}

if __name__ == '__main__':
    # main()
    param_grid = {'lgbm__'+k: v for k, v in param_lgbm.items()}
    tune(pipeline, param_grid, X, y, save=1)
