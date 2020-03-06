#!/usr/bin/env python
#%% Required Packages
import os
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import numbers
import scipy
from utils import *
import warnings
warnings.filterwarnings('ignore')


#%% Data Preprocessing
path = makePath()
X1 = importData("X_train")      # X1 = train
X2 = importData("X_test")       # X2 = test
y1 = importData("y_train")      # y1 = labels
X = concat(X1, X2)              # X1 + X2 = data

# Clean Data
X = removeId(X)
X = removeConstants(X)
X = removeZeros(X)
y1 = removeId(y1)

# Merge and split datasets
X1, X2 = unconcat(X)
X = concat(X1, X2)

# Impute Missing Values
missingValues(X)
if False:
    X = missingValues(X, opt='median')

# Remove Collinear Features
if False:
    X = selectVIF(X, threshold = 10)

# Outlier Detection
if True:
    X1, X2 = unconcat(X)            # Only on train data!
    X1o, y1o = removeOutliers(X1, y1, opt='covariance')
    X1oo, y1oo = removeOutliers(X1o, y1o, opt='isolation', rerun=10, max_features=0.9, max_samples=0.9)
    X, y1 = concat(X1oo, X2, y1oo)
    del X1o, y1o, X1oo, y1oo

# Sacle Features
if True:
    X = scaleFeatures(X,'standard')

# Feature Selection
if False:
    X1, X2 = unconcat(X)
    X1f, X2f = selectFeatures(X1, X2, y1, opt = 'covariance')
    
# Featrue Compression
if True:
    X = compressFeatures(X, opt='pca', components=430) # Variance explained: n_comp = 0.9: 430, 0.95: 630, 1.0: 1000
 

#%% Data parameters

# Select preprocessed data
X1, X2 = unconcat(X)
X1 = X1.values
X2 = X2.values
y1 = y1.values.ravel()

# Select parameters from data
n1,m = X1.shape
n2,m = X2.shape
c = len(np.unique(y1))
cw = n1 / (c * np.bincount(y1))
cw_obs = {0:6, 1:1, 2:6}
seed = 0
folds = 10

#%% Grid search for Hyperparameters
if False:
    
    submissionName = 'NestedCV'
    
    innerCV = 10                 # Classic Cross-Validation

    nestedCV = True            # Nested Cross-Validation
    outerCV = 5 

    randomizedGrid = False      # Parameters specified by distribution
    randomizedIter = 30 

    from sklearn.svm import SVC
    from sklearn.gaussian_process.kernels import RationalQuadratic

    base = SVC(kernel=RationalQuadratic(),
               decision_function_shape='ovo', class_weight=cw_obs)

    param_grid = {
        'C': [1, 10, 20, 30],
        'kernel__alpha': [1e-1, 1e-2, 1e-3],
        'kernel__length_scale': [50, 100, 150]
    }

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
    inner_cv = sklearn.model_selection.StratifiedKFold(n_splits=innerCV, shuffle=True, random_state=seed)

    if randomizedGrid: 
        clf = RandomizedSearchCV(estimator=base, param_distributions=param_grid,
                                 scoring='balanced_accuracy', cv=inner_cv,
                                 n_iter=randomizedIter, random_state=seed,
                                 n_jobs=-1, verbose=1)
    else: 
        clf = GridSearchCV(estimator=base, param_grid=param_grid,
                           scoring='balanced_accuracy', cv=inner_cv,
                           n_jobs=-1, verbose=1)

    if nestedCV:
        SearchCV = [None]*outerCV
        ScoreCV = np.zeros(outerCV)
        outer_cv = sklearn.model_selection.StratifiedKFold(n_splits=outerCV, shuffle=True, random_state=seed)
        for i, (Xf1_index, Xf2_index) in enumerate(outer_cv.split(X1, y1)):
            print('\n ----- Running Nested CV ----- \n Currently',i+1, '/', outerCV, 'Outer CVs')
            Xf1, Xf2 = X1[Xf1_index], X1[Xf2_index]
            yf1, yf2 = y1[Xf1_index], y1[Xf2_index]

            clf.fit(Xf1, yf1)
            bestHyperparameters(clf, n_best=3)

            ScoreCV[i] = validationScore(clf, Xf2, yf2, score_out=True, class_report=False)
            SearchCV[i] = bestHyperparameters(clf, best_out=True)

        best_params = nestedCVResults(SearchCV, ScoreCV)
        model = base.set_params(**best_params)

    else: 
        clf.fit(X1, y1)
        bestHyperparameters(clf, n_best=10)
        model = base.set_params(**clf.best_params_)

    submissionFile(path, model, X1, y1, X2, name=submissionName)



#%% Define models
from sklearn.naive_bayes import GaussianNB
params_nb = {
    'priors': [1./8, 6./8, 1./8]
}

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
params_lda = {
    'solver' : 'lsqr', 
    'priors' : [1./8, 6./8, 1./8],
    'shrinkage': 0.9
}

from sklearn.linear_model import LogisticRegression
params_lr = {
    'C': 0.001,
    'multi_class' : 'multinomial', 
    'penalty' : 'l2', 
    'solver' : 'sag', 
    'class_weight' : cw_obs
}

from sklearn.svm import SVC
params_svc_rbf = {
    'kernel': 'rbf',
    'C': 15,
    'gamma': 1e-5,
    'decision_function_shape': 'ovo',
    'class_weight': cw_obs,
    'probability': False
}

from sklearn.gaussian_process.kernels import RationalQuadratic
params_svc_rq = {
    'kernel': RationalQuadratic(),
    'C': 20,
    'kernel__alpha': 0.01,
    'kernel__length_scale': 150,
    'decision_function_shape': 'ovo',
    'class_weight': cw_obs,
    'probability': False
}

from sklearn.gaussian_process.kernels import Matern
params_svc_m = {
    'kernel': Matern(),
    'C': 15,
    'kernel__nu': 1.5,
    'kernel__length_scale': 200,
    'decision_function_shape': 'ovo',
    'class_weight': cw_obs,
    'probability': False
}

from sklearn.ensemble import RandomForestClassifier
params_rf = {
    'n_estimators': 500,
    'min_samples_leaf': 3,
    'max_features':'sqrt',
    'min_impurity_decrease': 1e-4,
    'class_weight': cw_obs,
    'n_jobs': -1,
    'random_state': seed
}

from sklearn.ensemble import GradientBoostingClassifier
params_gb = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'subsample': 1,
    'max_depth': 5,
    'max_features': None,
    'n_iter_no_change': 20,
    'random_state': seed
}

from sklearn.ensemble import ExtraTreesClassifier
params_et = {
    'n_estimators': 500,
    'min_samples_leaf': 3,
    'max_features': 'sqrt',
    'min_impurity_decrease': 1e-4,
    'class_weight': cw_obs,
    'n_jobs': -1,
    'random_state': seed
}

from sklearn.ensemble import AdaBoostClassifier
params_ada = {
    'n_estimators': 500,
    'learning_rate': 0.1,
    'random_state': seed
}

from sklearn.neighbors import KNeighborsClassifier
params_knn = {
    'n_neighbors': 5,
    'weights': 'distance',
    'metric': 'mahalanobis',
    'n_jobs': -1
}

from sklearn.neural_network import MLPClassifier
params_nn2 = {
    'hidden_layer_sizes': (512,256),
    'activation': 'relu',
    'batch_size': 32,
    'alpha': 0.001,
    'solver': 'adam', 
    'learning_rate_init': 0.01,
    'max_iter': 200,
    'early_stopping': True,
    'n_iter_no_change': 10,
    'random_state': seed
}

from sklearn.neural_network import MLPClassifier
params_nn3 = {
    'hidden_layer_sizes': (256, 256, 256),
    'activation': 'relu',
    'batch_size': 32,
    'alpha': 0.001,
    'solver': 'adam',
    'learning_rate_init': 0.01,
    'max_iter': 200,
    'early_stopping': True,
    'n_iter_no_change': 10,
    'random_state': seed
}

# %% Fit model/models

if False:
    estimators = {
        'NB': [GaussianNB, params_nb],
        'LDA': [LinearDiscriminantAnalysis, params_lda],
        'Logistic': [LogisticRegression, params_lr],
        'SVC_rbf': [SVC, params_svc_rbf],
        'SVC_rq': [SVC, params_svc_rq],
        'SVC_m': [SVC, params_svc_m],
        'RF': [RandomForestClassifier, params_rf],
        'GB': [GradientBoostingClassifier, params_gb],
        'ET': [ExtraTreesClassifier, params_et],
        'ADA': [AdaBoostClassifier, params_ada],
        'NN2': [MLPClassifier, params_nn2],
        'NN3': [MLPClassifier, params_nn3],
        'KNN': [KNeighborsClassifier, params_knn]
    }

    estimator = {
        'SVC_rbf': [SVC, params_svc_rbf],
    }

    use = estimators

    make_submission = True

    save_pred = False
    save_prob = False

    if save_pred: 
        i=0
        pred = np.zeros((n2, len(estimators)))
    if save_prob:
        j=0
        prob = np.ndarray((n2, c, len(estimators)))

    for key, value in use.items():
        print('Fitting', key, 'model')
        model = value[0]()
        params = value[1]
        model.set_params(**params)

        if save_pred:
            model.fit(X1,y1)
            pred[:,i] = model.predict(X2)
            i += 1
        elif save_prob:
            model.fit(X1,y1)
            prob[:,:,j] = model.predict_proba(X2)
            j += 1
        else:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            score = getScoreCV(model,X1,y1,skf)

        if make_submission:
            submissionFile(path, model, X1, y1, X2, name=key)


#%% Ensamble
if False:
    majority = scipy.stats.mode(pred, axis=1).mode.ravel()
    avgprob = prob.mean(axis=2).argmax(axis=1)

    submissionFile(path, model, X1, y1, X2, pred=majority, name='ensamble', do_fit=False, do_predict=False)


