import os
import sys
dir_cur = os.path.abspath(os.path.dirname(__file__))
dir_par = os.path.dirname(dir_cur)
sys.path.insert(0, dir_par)
from helper.tuning import tune
from helper.pipeline_tool import PrintXShape
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignore tf warning

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.base import TransformerMixin
from sklearn.svm import SVC
# classifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from biosppy import ecg, tools

from time import gmtime, strftime, localtime, time

SAMPLING_RATE=float(300)

def transform_ecg(signal, sampling_rate=SAMPLING_RATE, verbose=False):
    signal = signal[~np.isnan(signal)]
    X = list()
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal, sampling_rate, show=verbose)
    rpeaks = ecg.correct_rpeaks(signal=signal, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.1)
    
    peaks = signal[rpeaks]
    if len(heart_rate) < 2:
        heart_rate = [0, 1]
    if len(heart_rate_ts) < 2:
        heart_rate_ts = [0, 1]
    
    X.append(np.mean(peaks))
    X.append(np.min(peaks))
    X.append(np.max(peaks))
    X.append(np.mean(np.diff(rpeaks)))
    X.append(np.min(np.diff(rpeaks)))
    X.append(np.max(np.diff(rpeaks)))
    X.append(np.mean(heart_rate))
    X.append(np.min(heart_rate))
    X.append(np.max(heart_rate))
    X.append(np.mean(np.diff(heart_rate)))
    X.append(np.min(np.diff(heart_rate)))
    X.append(np.max(np.diff(heart_rate)))
    X.append(np.mean(np.diff(heart_rate_ts)))
    X.append(np.min(np.diff(heart_rate_ts)))
    X.append(np.min(np.diff(heart_rate_ts)))
    X.append(np.max(np.diff(heart_rate_ts)))
    X.append(np.sum(filtered-signal))
    
    X += list(np.mean(templates, axis=0))
    X += list(np.min(templates, axis=0))
    X += list(np.max(templates, axis=0))
    X = np.array(X)
    
    X[np.isnan(X)] = 0
    return np.array(X)

class RowTransformer(TransformerMixin):

    def __init__(self, my_func):
        self.my_func = my_func

    def transform(self, X):
        X_ = np.apply_along_axis(self.my_func, 1, X)
        return X_

    def fit(self, *args, **kwargs):
        return self

def read_data(verbose=True, save_part=10):
    try:
        X = pd.read_pickle("{}/data/X_train.pkl".format(dir_cur))
        y = pd.read_pickle("{}/data/y_train.pkl".format(dir_cur))
    except FileNotFoundError:
        X = pd.read_csv("{}/data/X_train.csv".format(dir_cur), index_col=0)
        y = pd.read_csv("{}/data/y_train.csv".format(dir_cur), index_col=0)
        X.to_pickle("{}/data/X_train.pkl".format(dir_cur))
        y.to_pickle("{}/data/y_train.pkl".format(dir_cur))

    X_ = X.values
    y_ = sklearn.utils.validation.column_or_1d(y.values)


    if verbose:
        print('Reading data...')
        print('X: {}\ny: {}\n'.format(X_.shape, y_.shape))
        print('{} samples contain missing values'.format(len(np.isnan(X_))))
        print('{:4.0f} out of {} features MISSED on average'.format(np.average(np.argwhere(np.isnan(X_))), X_.shape[1]))

    if save_part:
        X.iloc[0:save_part].to_csv("{}/data/X_train_first_{}.csv".format(dir_cur, save_part))
        y.iloc[0:save_part].to_csv("{}/data/y_train_first_{}.csv".format(dir_cur, save_part))

    return X_, y_

pipeline = Pipeline([
    ('ecg_transformer', RowTransformer(transform_ecg)),
    ('scaler', StandardScaler()),
    # ('debug', PrintXShape('ECG Transformer')),
    # ('clf', RandomForestClassifier(n_estimators=500, max_depth=15)),
    ('clf', LGBMClassifier(random_state=0, n_estimators=100, max_depth=15, num_leaves=31)),
    # ('clf', GradientBoostingClassifier(random_state=0, n_estimators=120, max_depth=5, learning_rate=0.1)),
], verbose=1)

def main():
    print('Fitting...\n')
    t0 = time()
    pipeline.fit(X, y)
    print('Fitting time  : {:.3f} sec\n'.format(time() - t0))

    try:
        X_test = pd.read_pickle("{}/data/y_test.pkl".format(dir_cur))
    except FileNotFoundError:
        X_test = pd.read_csv("{}/data/X_test.csv".format(dir_cur), index_col=0)
        X_test.to_pickle("{}/data/X_test.pkl".format(dir_cur))
    X_test = X_test.values

    print('Predicting...\n')
    pred = pipeline.predict(X_test)
    np.savetxt("{}/submission.csv".format(dir_cur),
               np.dstack((np.arange(0, pred.size), pred))[0],
               '%0f,%0f', comments='', header="id,y")

param_grid = {
    # 'clf__n_estimators': [20, 100],
    'clf__max_depth': [-1, 5, 10, 15],
    'clf__num_leaves': [20, 30, 36, 40, 50],

}


if __name__ == "__main__":
    print('Starting at   :', strftime("%Y-%m-%d %H:%M:%S", localtime()))
    t0 = time()
    X, y = read_data()
    # main()
    tune(pipeline, param_grid, X, y, save=1, scoring='f1_micro', verbose=1)
    print('Ending at     :', strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print('Total runtime : {:.3f} sec\n'.format(time() - t0))
