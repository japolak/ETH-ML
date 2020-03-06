#!/usr/bin/env python
#%% Required Packages
import statistics
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import numbers
import scipy
import warnings
import matplotlib
#import biosppy
#import keras
#from biosppy.signals import ecg
from utils import *
#from ecg_utils import *
#from extract_features import *
warnings.filterwarnings('ignore')



#%% Read Data
'''
path = makePath()
X1 = importPickle("X_train")      # X1 = train
X2 = importPickle("X_test")       # X2 = test
y1 = importData("y_train")        # y1 = labels

#X = concat(X1, X2)                # X1 + X2 = data
#X1, X2 = unconcat(X)

#%% Data cleaning

X1c = denoiseData(X1, run=1)
X2c = denoiseData(X2, run=1)

X1cc = denoiseData(X1c, run=2)
X2cc = denoiseData(X2c, run=2)

savePickle(X1cc, 'X_train_clean')
savePickle(X2cc, 'X_test_clean')

#saveData(X1c,'X_train_clean')
#saveData(X2c,'X_test_clean')

#%% Data Preprocessing
y1 = importData("y_train")
X1c = importPickle("X_train_clean")
X2c = importPickle("X_test_clean")

# This just inspects which data would be inverted
inv_X1, inv_y1 = inspectInvertions(X1c, y1, verbose=0)
inv_X2 = inspectInvertions(X2c, verbose=0)

#%% Data Preprocessing

#%%
milan = importData("invert_indices_train")
inv_X1_m = []
for i in range(0,milan.shape[0]):
    which = milan.iloc[i,0]
    check = milan.iloc[i,1]
    if check == 1:
        inv_X1_m.append(which-1)

jakub_train = inv_X1
milan_train = inv_X1_m
jaind_train = np.unique(jakub_train*np.invert(np.in1d(jakub_train, milan_train)))

#%%
milan = importData("indices_inverted_test")
inv_X2_m = []
for i in range(0, milan.shape[0]):
    which = milan.iloc[i, 0]
    check = milan.iloc[i, 1]
    if check == 1:
        inv_X2_m.append(which-1)

jakub_test = inv_X2
milan_test = inv_X2_m
jaind_test = np.unique(
    jakub_test*np.invert(np.in1d(jakub_test, milan_test)))

savePickle(pd.DataFrame(jaind_train),"ind_train")
savePickle(pd.DataFrame(jaind_test),"ind_test")
#%%
inv_train = importPickle("ind_train").values.ravel()
inv_test = importPickle("ind_test").values.ravel()

flip_train = [11,18,43,139,141,155,215,223,235,243,257,261,305,323,330,336,377,343,376,377,434,456,477,493,536,645,652,655,658,669,800,824,851,862,877,894,905,911,940,946,960,964,967,981,994,1031,1052,1056,1072,1122,1165,1208,1217,1246,1251,1277,1289,1300,1306,1309,1313,1394,1454,1472,1518,1533,1547,1599,1605,1610,1657,1708,1746,1799,1838,1883,1904,1918,1937,1978,1989,2006,2022,2026,2065,2082,2121,2178,2193,2217,2218,2221,2225,2233,2265,2288,2304,2340,2350,2358,2416,2433,2446,2450,2470,2478,2486,2498,2511,2514,2520,2548,2551,2573,2583,2662,2704,2818,2911,2954,3030,3051,3144,3146,3190,3142,3266,3172,3277,3280,3285,3297,3324,3371,3378,3384,3409,3418,3430,3434,3436,3440,3481,3506,3558,3648,3661,3709,3742,3803,3826,3894,3906,3919,3932,3947,3997,3999,4009,4017,4051,4066,4069,4075,4078,4133,4180,4207,4210,4221,4239,4241,4254,4272,4285,4304,4335,4362,4405,4442,4500,4501,4507,4541,4549,4551,4557,4637,4664,4697,4702,4740,4749,4834,4858,4862,4868,4874,4921,4958,4982,4984,4995,4996,4999,5041,5082,5088,5107,]
FLIP_TRAIN = np.concatenate((milan_train, np.array(flip_train)), axis=None)

flip_test = [6,12,36,49,50,93,149,237,294,313,351,365,367,403,404,477,506,546,550,553,558,594,604,622,629,678,680,683,689,697,719,730,757,771,778,810,814,824,828,840,855,863,877,891,917,956,965,987,1009,1013,1055,1071,1079,1114,1182,1198,1341,1373,1455,1475,1480,1558,1605,1661,1679,1722,1733,1780,1782,1786,1804,1881,1836,1855,1947,1983,2018,2030,2043,2046,2056,2078,2095,2106,2148,2156,2160,2179,2187,2221,2238,2330,2335,2336,2339,2404,2453,2515,2568,2573,2604,2616,2622,2631,2644,2652,2672,2694,2753,2765,2781,2785,2799,2829,2837,2944,2951,2956,2969,2995,3004,3034,3044,3097,3101,3111,3123,3214,3220,3373,3377,3399]
FLIP_TEST = np.concatenate((milan_test, np.array(flip_test)), axis=None)


#%%
# This actually inverts the dataset
X1cf = invertData(X1c, invert_list=FLIP_TRAIN)
X2cf = invertData(X2c, invert_list=FLIP_TEST)



savePickle(X1cf, 'X_train_clean_flip')
savePickle(X2cf, 'X_test_clean_flip')

savePickle(pd.DataFrame(FLIP_TRAIN),'flip_train')
savePickle(pd.DataFrame(FLIP_TEST),'flip_test')

#%% Save Data

#saveData(X_train_clean, 'X_train_clean')
#saveData(X_test_clean, 'X_test_clean')

##% Load preprocessed data

#%% Compute features
y1 = importData("y_train")
X1cf = importPickle("X_train_clean_flip")
X2cf = importPickle("X_test_clean_flip")

data = X2cf.copy()
n = data.shape[0]
Tdata = np.ndarray((8, 180, n))
Hdata = np.ndarray((4, n))
Sdata = np.ndarray((3,n))
for index, row in data.iterrows():
    raw_signal = row.dropna().values
    #index = 116
    #raw_signal = data.iloc[index, :].dropna().values
    signal = ecg.ecg(signal=raw_signal, sampling_rate=300, show=False)
    # Template heartbeat  (n,180)
    Stemp = signal['templates']
    hb = len(Stemp)
    Stempf = removeOut(Stemp, m=4)
    hbf = Stempf.shape[0]
    heartbeats = np.zeros((8, 180))
    for i,q in enumerate([.1,.25,.5,1,.75,.9,]):
        if q == 1:
            heartbeats[i, :] = np.mean(Stempf,axis=0)
        else: 
            heartbeats[i,:] = np.quantile(Stempf,q,axis=0)
    Tdata[:,:,index]=heartbeats 

    # Heart rate Instantaneous (n-1,)
    Sheartr = signal['heart_rate']
    if len(Sheartr)==0:
        Hdata[:, index] = 0
        print('Index', index, 'empty heartrate')
    else:
        Hdata[0,index]=min(Sheartr)
        Hdata[1,index]=max(Sheartr)
        Hdata[2,index]=np.mean(Sheartr)
        Hdata[3,index]=np.std(Sheartr)

    # Filtered ECG signal (N,)
    Sfilt = signal['filtered']
    Sdata[0,index]=np.std(Sfilt)
    zero_cross = np.zeros(hbf)
    for i in range(0,hbf):
        zero_cross[i] = len(np.where(np.diff(np.sign(Stempf[i, :])))[0])
    Sdata[1,index]=np.mean(zero_cross)
    Sdata[2,index]=np.std(zero_cross)


###
X2 = Tdata[:,:,:].reshape(-1,n).T
#y = y1.values.ravel()
#%% Featurue Computation
from sklearn.manifold import TSNE
from sklearn import decomposition
estim_p = decomposition.PCA(n_components=7)
estim_t = TSNE(n_components=3)
X1e_tsne = estim_t.fit_transform(X1)                # 3
X1e_pca = estim_p.fit_transform(X1)                 # 7
X1e_mae = abs(Tdata).sum(axis=1).T                  # 6
X1e_sig = Sdata.T                                   # 3
X1e_hr = Hdata.T                                    # 4

X1e = np.hstack((X1e_tsne,X1e_pca,X1e_mae,X1e_sig,X1e_hr,X1e_ae))
#####
X2e_tsne = estim_t.fit_transform(X2)                # 3
X2e_pca = estim_p.transform(X2)                     # 7
X2e_mae = abs(Tdata).sum(axis=1).T                  # 6
X2e_sig = Sdata.T                                   # 3
X2e_hr = Hdata.T                                    # 4

X2e = np.hstack((X2e_tsne, X2e_pca, X2e_mae, X2e_sig, X2e_hr,X2e_ae))
##
saveData(pd.DataFrame(X1e),"jakub_features_train")
saveData(pd.DataFrame(X2e),"jakub_features_test")
#%% AE
####
#X1_ae= Tdata[:, :, :].reshape(-1, n).T
#X2_ae = Tdata[:, :, :].reshape(-1, n).T
####

from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
enco = 20 # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
inp = 1440

input_img = Input(shape=(inp,))
encoded = Dense(180, activation='relu')(input_img)
encoded = Dense(50, activation='relu')(encoded)
encoded = Dense(enco, activation='relu')(encoded)

decoded = Dense(50, activation='relu')(encoded)
decoded = Dense(180, activation='relu')(decoded)
decoded = Dense(inp, activation='linear')(decoded)
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(enco,))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_layer3 = autoencoder.layers[-1](decoder_layer2)
decoder = Model(encoded_input, decoder_layer3)

autoencoder.compile(optimizer='adadelta', loss='mean_absolute_error')

###
autoencoder.fit(X1_ae, X1_ae,
                epochs=100,
                batch_size=64,
                sample_weight=sw_obs,
                shuffle=True,
                validation_data=(X2_ae, X2_ae),
                verbose=2)

X1e_ae = encoder.predict(X1_ae)
X2e_ae = encoder.predict(X2_ae)

#%% Data Visualisation
y1 = importData("y_train")        
X1cf = importPickle("X_train_clean_flip")
X2cf = importPickle("X_test_clean_flip")

data = X1cf.copy()

#############
index = 4463
signal = data.iloc[index, :].dropna()
plt.plot(np.linspace(0, len(signal), len(signal), endpoint=False), signal, lw=2)
plt.show()
#############

#############
index = 2719
print(y1.iloc[index].values)
signal = ecg.ecg(signal=data.iloc[index, :].dropna(), sampling_rate=300, show=True)
#############

#############
for index in range(0,100):
    print(y1.iloc[index].values, index)
    signal = ecg.ecg(signal=data.iloc[index, :].dropna(), sampling_rate=300, show=True)
#############

#############
for index in []:
    print(y1.iloc[index].values, index)
    signal = ecg.ecg(signal=data.iloc[index, :].dropna(), sampling_rate=300, show=True)
#############

#############
for i, index in enumerate([]):
    if inv_y1[i] == 3:
        print(inv_y1[i], index)
        signal = ecg.ecg(signal=data.iloc[index, :].dropna(), sampling_rate=300, show=True)
#############


#%%

X1_m = importData("milan_features_train").drop("y.train",axis=1)
X1_n = importData("milan_HB_train")
X1_l = importData("leander_features_train")
X1_j = importData("jakub_features_train")

X2_m = importData("milan_features_test")
X2_n = importData("milan_HB_test")
X2_l = importData("leander_features_test").iloc[0:3411,:]
X2_j = importData("jakub_features_test")

y1 = importData("y_train")

X1 = pd.DataFrame(np.hstack((X1_m, X1_n, X1_l, X1_j)))
X2 = pd.DataFrame(np.hstack((X2_m, X2_n, X2_l, X2_j)))

X = concat(X1, X2)      
X = removeConstants(X)
X1, X2 = unconcat(X)

savePickle(X1,"X_features_train")
savePickle(X2,"X_features_test")
'''

#%%%
path = makePath()
X1 = importPickle("X_features_train")
X2 = importPickle("X_features_test")
y1 = importData("y_train")

X = concat(X1, X2)
#X = scaleFeatures(X)
X1, X2 = unconcat(X)

X1 = X1.values
X2 = X2.values
y1 = y1.values.ravel()

# Select parameters from data
n1,m = X1.shape
n2,m = X2.shape
c = len(np.unique(y1))
cw = n1 / (c * np.bincount(y1))
cw_obs = {0:0.5, 1:3, 2:1, 3:7}
seed = 0
folds = 10

#%% Grid search for Hyperparameters
if True:
    
    submissionName = 'CV_GB'
    
    innerCV = 10                 # Classic Cross-Validation

    nestedCV = True            # Nested Cross-Validation
    outerCV = 5 

    randomizedGrid = False      # Parameters specified by distribution
    randomizedIter = 30 

    from sklearn.ensemble import GradientBoostingClassifier

    base = GradientBoostingClassifier(
        n_estimators= 500,
        learning_rate= 0.01,
        subsample=0.5,
        max_depth= 5,
        n_iter_no_change= 20,
        random_state= seed
        )

    param_grid = {
        'n_estimators': [300,500,800],
        'learning_rate': [0.01,0.001],
        'max_features':[20,50]
        }



    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
    inner_cv = sklearn.model_selection.StratifiedKFold(n_splits=innerCV, shuffle=True, random_state=seed)

    if randomizedGrid: 
        clf = RandomizedSearchCV(estimator=base, param_distributions=param_grid,
                                 scoring='accuracy', cv=inner_cv,
                                 n_iter=randomizedIter, random_state=seed,
                                 n_jobs=-1, verbose=1)
    else: 
        clf = GridSearchCV(estimator=base, param_grid=param_grid,
                           scoring='accuracy', cv=inner_cv,
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
'''
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
    'max_features': 50,
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
'''
