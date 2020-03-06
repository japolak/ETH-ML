import os
import sys
dir_cur = os.path.abspath(os.path.dirname(__file__))
dir_par = os.path.dirname(dir_cur)
sys.path.insert(0, dir_par)
from helper.tuning import tune
from helper.pipeline_tool import PrintXShape
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore tf warning

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

from time import gmtime, strftime, localtime, time

from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape
from keras.wrappers.scikit_learn import KerasClassifier


def read_data(verbose=True, save_part=False):
    try:
        X = pd.read_pickle("{}/data/X_train.pkl".format(dir_cur))
        y = pd.read_pickle("{}/data/y_train.pkl".format(dir_cur))
    except FileNotFoundError:
        train_eeg1 = pd.read_csv(
            "{}/data/train_eeg1.csv".format(dir_cur), index_col=0)
        train_eeg2 = pd.read_csv(
            "{}/data/train_eeg2.csv".format(dir_cur), index_col=0)
        train_emg = pd.read_csv(
            "{}/data/train_emg.csv".format(dir_cur), index_col=0)
        y = pd.read_csv(
            "{}/data/train_labels.csv".format(dir_cur), index_col=0)
        X = pd.concat([train_eeg1, train_eeg2, train_emg], axis=1)
        X.to_pickle("{}/data/X_train.pkl".format(dir_cur))
        y.to_pickle("{}/data/y_train.pkl".format(dir_cur))

    X_ = X.values
    y_ = sklearn.utils.validation.column_or_1d(y.values)

    if verbose:
        print('Reading data...')
        print('X: {}\ny: {}\n'.format(X_.shape, y_.shape))
        print('{} samples contain missing values'.format(len(np.isnan(X_))))
        print('{:4.0f} out of {} features MISSED on average'.format(
            np.average(np.argwhere(np.isnan(X_))), X_.shape[1]))

    if save_part:
        X.iloc[0:save_part].to_csv(
            "{}/data/X_train_first_{}.csv".format(dir_cur, save_part))
        y.iloc[0:save_part].to_csv(
            "{}/data/y_train_first_{}.csv".format(dir_cur, save_part))

    return X_, y_


def read_test_data():
    try:
        X_test = pd.read_pickle("{}/data/X_test.pkl".format(dir_cur))
    except FileNotFoundError:
        test_eeg1 = pd.read_csv(
            "{}/data/test_eeg1.csv".format(dir_cur), index_col=0)
        test_eeg2 = pd.read_csv(
            "{}/data/test_eeg2.csv".format(dir_cur), index_col=0)
        test_emg = pd.read_csv(
            "{}/data/test_emg.csv".format(dir_cur), index_col=0)
        X_test = pd.concat([test_eeg1, test_eeg2, test_emg], axis=1)
        X_test.to_pickle("{}/data/X_test.pkl".format(dir_cur))
    X_test = X_test.values

    return X_test


def lstm():
    # create model
    model = Sequential()
    model.add(Reshape((512, 3), input_shape=(1536,)))
    model.add(LSTM(32, input_shape=(512, 3), dropout=0.2,
                   recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nn', KerasClassifier(build_fn=lstm, epochs=1, batch_size=100, verbose=1)),
], verbose=1)


def main():
    print('Fitting...\n')
    t0 = time()
    X, y = read_data()
    pipeline.fit(X, y)
    print('Fitting time  : {:.3f} sec\n'.format(time() - t0))

    X_test = read_test_data()

    print('Predicting...\n')
    pred = pipeline.predict(X_test)
    pred = sklearn.utils.validation.column_or_1d(pred)
    np.savetxt("{}/submission.csv".format(dir_cur),
               np.dstack((np.arange(0, pred.size), pred))[0],
               '%i,%i', comments='', header="Id,y")


if __name__ == "__main__":
    print('Starting at   :', strftime("%Y-%m-%d %H:%M:%S", localtime()))
    t0 = time()
    model = lstm()
    model.summary()
    # main()
    print('Ending at     :', strftime("%Y-%m-%d %H:%M:%S", localtime()))
    print('Total runtime : {:.3f} sec\n'.format(time() - t0))
