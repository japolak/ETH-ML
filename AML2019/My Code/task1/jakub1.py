#%% 
"""
Required Packages.

Install the following if doesn't contain already
"""
import os
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import numbers
import scipy
import xgboost as xgb
from utils import *

#%% 
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

# Impute Missing Values
missingValues(X)
X = missingValues(X, opt='median')

# Remove Collinear Features
X = selectVIF(X, threshold = 10)

# Outlier Detection
X1, X2 = unconcat(X)            # Only on train data!
X1o, y1o = removeOutliers(X1, y1, opt='covariance')
X1oo, y1oo = removeOutliers(X1o, y1o, opt='isolation', rerun=10, max_features=0.5,max_samples=0.5)
X, y1 = concat(X1oo, X2, y1oo)
del X1o, y1o, X1oo, y1oo

# Sacle Features
X = scaleFeatures(X,'standard')


# Feature Selection
X1, X2 = unconcat(X)
#%%
X1f, X2f = selectFeatures(X1, X2, y1, opt = 'covariance')

# Select preprocessed data
X1 = X1f
X2 = X2f
y1 = y1



#%% Rum model
model = xgb.XGBRegressor(n_estimators=100, max_depth=6, objective="reg:linear", random_state=0)
model.fit(X1,y1.values.ravel())
y2 = model.predict(X2)

submissionFile(y2)
