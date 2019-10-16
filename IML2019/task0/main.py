# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a main script file for Intro to Machine Learning.
@title: task0
@author: japolak
"""
#%% Import packages
import os
import numpy as np
import pandas as pd
import warnings
import copy
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from scipy import stats
import matplotlib.pyplot as plt
%matplotlib inline


#%% Miscallaneous


#%% Select directory
path_dir = os.path.dirname(os.path.realpath('/Users/Jakub/Documents/ETH/S19/IML/task0/main.py'))
path_train= os.path.join(path_dir + '/train.csv')
path_test = os.path.join(path_dir + '/test.csv')

#%% Utils functions
def save_solution(csv_file,pred_prob):
	with open(csv_file, 'w') as csv:
		df = pd.DataFrame.from_dict({'Id':range(10000,10000+len(pred_prob)),'y': pred_prob})
		df.to_csv(csv,index = False)

def read_data(csv_file):
    with open(csv_file,'r') as csv:
        df = pd.read_csv(csv)
    return df

def get_data(csv_file):
    with open(csv_file, 'r') as csv:
        df = pd.read_csv(csv)
        df = df.loc[:, df.columns != 'y']
        data = df.drop('Id', axis = 1)
    return data

def get_target(csv_file):
	with open(csv_file, 'r') as csv:
		df = pd.read_csv(csv)
		y = df['y']
	y = np.array(y)
	return y

#%% Read Data
df=read_data(path_train)
train=get_data(path_train)
test=get_data(path_test)
y=pd.DataFrame(get_target(path_train),columns = list('y'))
#%% Specify dataset to work with
df = copy.copy(train)

#%% Explore Data
# df.shape
# df.dtypes
# df.describe()

''' # Plot Scatter Matrix (30sec)
from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(10, 10), diagonal='kde')
'''

''' # Plot histograms
import matplotlib.pyplot as plt
for i in range(df.shape[1]):
    print('\n',df.columns.get_values().tolist()[i])
    plt.hist(df.iloc[:,1],bins=50)
    plt.show()
'''

''' #Plot Boxplots
df.boxplot()
'''

#%% Missing values
df_missing = copy.copy(df)
# Check Missing values   eg. df_missing.loc[0][0]=np.NaN
print(df_missing.isna().sum().sum())

def impute_values(df_missing):
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    df = pd.DataFrame(data=imp.fit_transform(df_missing),columns=df_missing.columns)
    return df
with warnings.catch_warnings():         #Suppres warning
    warnings.simplefilter("ignore")     #ignore DeprecationWarning
    df = impute_values(df_missing)
# df = impute_values(df_missing)   
del df_missing


#%% Outlier Selection
df_outlier = copy.copy(df)        
# make outlier       
df_outlier.loc[0][0]=df_outlier.loc[0][0]*10000
rs = np.random.RandomState(42)
of = 0.0001         # Outlier fraction HEAVILY influences the number of outliers found!! You should be looking on a case when all classifiers predict the same number and same outlier
     
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP        
        
classifiers = {
    'Angle-based Outlier Detector (ABOD)': ABOD(contamination=of),
    'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(contamination=of,check_estimator=False, random_state=rs),
    'Feature Bagging': FeatureBagging(LOF(n_neighbors=35),contamination=of,check_estimator=False,random_state=rs),
    'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=of),
    'Isolation Forest': IForest(contamination=of, random_state=rs),
    'K Nearest Neighbors (KNN)': KNN(contamination=of),
    'Average KNN': KNN(method='mean',contamination=of),
    'Median KNN': KNN(method='median',contamination=of),
    'Local Outlier Factor (LOF)': LOF(n_neighbors=35, contamination=of),
    'Minimum Covariance Determinant (MCD)': MCD(contamination=of, random_state=rs),
    'One-class SVM (OCSVM)': OCSVM(contamination=of,random_state=rs),
    'Principal Component Analysis (PCA)': PCA(contamination=of, random_state=rs),
}

# Subset of dataset         df_outlier = copy.copy(np.array(df_outlier)[0:10000,:])    
df_outlier = copy.copy(np.array(df_outlier))      # need np.array
df_outlier_pred = np.zeros((df_outlier.shape[0],len(classifiers)))
    
with warnings.catch_warnings():         #Suppres warning
    warnings.simplefilter("ignore")     #ignore FutureWarning
    for i, (clf_name,clf) in enumerate(classifiers.items()) :
        clf.fit(df_outlier)                                    # fit the dataset to the model
        scores_pred = clf.decision_function(df_outlier)*-1     # predict raw anomaly score
        y_pred = clf.predict(df_outlier)                       # prediction of a datapoint category outlier or inlier
        df_outlier_pred[:,i]=y_pred
        n_outliers = np.count_nonzero(y_pred == 1)          # count number of outliers
        print( 'Outliers:',n_outliers,' - ',clf_name,'( fraction =',of,')')
del clf_name, classifiers, i, n_outliers, scores_pred, y_pred

df_outlier_prob = np.array(df_outlier_pred.sum(1)/df_outlier_pred.shape[1])
n=int(df_outlier.shape[0]*of)
idx = (-df_outlier_prob).argsort()[:n]
prob = df_outlier_prob[idx]
d = {'idx':idx,'prob':prob}
outliers_prob=pd.DataFrame(data=d)

del idx,prob,d,df_outlier_prob,df_outlier_pred, n, of

#%% Scaling variables
df_scale = copy.copy(df)
def scale_variables(df_scale):
    scl = StandardScaler().fit(df_scale)
    df = pd.DataFrame(scl.transform(df_scale),columns=df_scale.columns)
    return df
df = scale_variables(df_scale)
del df_scale

#%% Make Predictions
train_hat=np.array(train.mean(1))
test_hat=np.array(test.mean(1))

#%% Performance Measure
y_true = y
y_pred = train_hat
RMSE = mean_squared_error(y_true, y_pred)**0.5
print('RMSE',RMSE)

#%% Save solution
solution = test_hat
path_solution = os.path.join(path_dir + '/solution.csv')
save_solution(path_solution,solution)
# eg.
#dummy_solution = 0.1*np.ones(len(test))
#save_solution(solution_path, dummy_solution)

#%%

