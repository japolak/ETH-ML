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
import keras
from keras.models import Model, Sequential
from keras.layers import Dropout, Dense, Input, Reshape, Flatten, GlobalAveragePooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score, ShuffleSplit, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.neighbors import LocalOutlierFactor
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline


def read_data():
    X = pd.read_csv("{}/data/X_train.csv".format(dir_cur), index_col=0).values
    y = pd.read_csv("{}/data/y_train.csv".format(dir_cur), index_col=0).values
    y = sklearn.utils.validation.column_or_1d(y)
    print('Reading data...')
    print('X: {}\ny: {}\n'.format(X.shape, y.shape))
    return X, y


def outliers_removal(X, y, verbose=False):
    outliers = LocalOutlierFactor(n_neighbors=40, contamination=0.08).fit_predict(X, y)
    X_, y_ = X[outliers == 1], y[outliers == 1]
    if not verbose:
        print('Removal outliers...')
        print('{} outliers removed\n'.format(X.shape[0] - X_.shape[0]))
    return X_, y_


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(200, input_dim=1000, activation='relu', kernel_initializer='random_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['categorical_accuracy'])
    return model


pipeline = Pipeline([
    ('sampler', ClusterCentroids()),
    ('scaler', StandardScaler()),
    # ('debug', PrintXShape('ClusterCentroids')),
    ('nn', KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=100, verbose=0)),
])

param_grid = {
    'nn__batch_size': [20, 100, 200]
}


def main():
    print('Fitting...')
    pipeline.fit(X, y)
    X_test = pd.read_csv(
        "{}/data/X_test.csv".format(dir_cur), index_col=0).values
    print('Predicting...')
    pred = pipeline.predict(X_test)
    np.savetxt("{}/submission.csv".format(dir_cur),
               np.dstack((np.arange(0, pred.size), pred))[0],
               '%0.1f,%0.1f', comments='', header="id,y")


if __name__ == "__main__":
    import time
    t0 = time.time()
    X, y = read_data()
    X, y = outliers_removal(X, y, 1)
    # main()
    tune(pipeline, param_grid, X, y, save=1, scoring='balanced_accuracy', verbose=1)
    print('Time : {:.3f} sec'.format(time.time() - t0))
