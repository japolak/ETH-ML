#%% Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Trees ensamble models
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.neighbors import (KNeighborsClassifier,RadiusNeighborsClassifier)

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score




#%% Data Preprocessing

# Load in the train and test datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop ID columns
train = train.drop('Id', axis = 1)
test = test.drop('Id', axis = 1)

# Load Y values
train_labels = pd.DataFrame(train['y'])
y_train = train['y'].ravel()
train = train.drop(['y'], axis=1)

# We should drop features x20, x19, x18
train = train.drop(['x20','x19','x18'],axis =1)
test = test.drop(['x20','x19','x18'],axis =1)

# Create Numpy arrays of train, test and target dataframes to feed into our models
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

# Scaling the data
ScaleData=True
if ScaleData :  
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
del ScaleData    

# Create validation set
nvalid = 0.0 # 0 means no validation set (ratio!)
x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size=nvalid,random_state=0)
if nvalid ==0 : del x_valid, y_valid, nvalid

# Some useful parameters 
ntrain = train.shape[0]
ntest = test.shape[0]
nclasses = len(np.unique(train_labels['y']))

nseed = 0 
nfolds = 10
skf = StratifiedKFold(n_splits= nfolds, shuffle=True, random_state=nseed)
# nweights = ntrain / (nclasses * np.bincount(train_labels['y']))

#%% Pearson correlation map and distribution
'''
colormap = plt.cm.RdBu
plt.figure(figsize=(24,22))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)

g = sns.pairplot(train, hue='y',palette="husl")
'''

#%% Grid Search for Hyperparameter tuning
if True:
    param_grid = {
     
}
    
    model = MLPClassifier(
            hidden_layer_sizes= (100, 100),
            activation='relu', #'logistic','tanh',
            solver='adam', #'lbfgs','sgd'
            alpha= 10,
            batch_size= 200,
            learning_rate='constant',#'invscaling','adaptive'
            max_iter=200 # These are epochs
            )
    clf = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

    clf.fit(x_train,y_train)
    print('\n Best parameters set: \n')
    print(clf.best_score_, 'for', clf.best_params_)
    print('\n Top 10 grid scores: \n')
    display(pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score')[['mean_test_score','params']].head(10))
    # Print all parameter combinations
    if False :
        print('\n Grid scores: \n')
        means = clf.cv_results_['mean_test_score'];stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print('%0.3f (+/-%0.03f) for %r' %(mean, std * 2, params))
        del means, stds, mean, std, params
    # Plot scores for only 1 parameter 
    if False:
        param_name = list(param_grid.keys())[0] #plot only if varying one numerical parameter
        def GridSearch_plot(clf,param_name):
            scores_df = pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score')
            best_row = scores_df.iloc[0, :]
            best_mean = best_row['mean_test_score'];best_stdev = best_row['std_test_score']
            best_param = best_row['param_' + param_name]
            scores_df = scores_df.sort_values(by='param_' + param_name)
            means = scores_df['mean_test_score'];stds = scores_df['std_test_score']
            params = scores_df['param_' + param_name]
            plt.figure(figsize=(9, 6))
            plt.semilogx(params,means, marker='o') #if log, then plt.semilogx
            plt.axhline(y=best_mean + best_stdev, color='red')
            plt.axhline(y=best_mean - best_stdev, color='red')
            plt.plot(best_param, best_mean, 'or')
            plt.title(param_name + " vs Score\nBest Score {:0.5f}".format(best_mean))
            plt.xlabel(param_name); plt.ylabel('Score')
            plt.show()
        GridSearch_plot(clf,param_name)

#%% Parameters setup for Classifiers 
        
params_lr = {
    'penalty': 'l1',
    'C': 10,
    'fit_intercept': True,
    'class_weight': 'balanced',
    'solver': 'saga',
    'multi_class': 'multinomial', 
    'n_jobs': -1
}        

# Random Forest parameters
params_rf = {
    'n_estimators': 500,
    'criterion':'gini',
    'max_features': 4,
    'max_depth': None, #10
    'min_samples_leaf': 2,
    'min_samples_split':2,
    'class_weight':'balanced',
    'warm_start': False
}

# Extra Trees Parameters
params_et = {
    'n_estimators': 500,
    'criterion': 'gini',
    'max_features': 4,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'warm_start': False,
    'class_weight': 'balanced',
    'n_jobs': -1
}

# AdaBoost parameters
params_ada = {
    'n_estimators': 50,
    'learning_rate' : 0.1
}

# Gradient Boosting parameters
params_gb = {
    'n_estimators': 1000,
    'learning_rate': 0.1,
    'subsample': 1,
    'max_features': 6,
    'max_depth': 8,
    'min_samples_split': 2,
    'min_samples_leaf': 2,
    'validation_fraction': 0.2,
}

# Support Vector Classifier Gaussian parameters 
params_svc = {
    'kernel' : 'rbf',
    'C' : 3, #5
    'gamma':0.1, # .05
    'decision_function_shape':'ovo',
    'class_weight':'balanced'
}

params_linsvc = {
    'penalty': 'l1',
    'C': 1,
    'multi_class': 'ovr',
    'fit_intercept': True,
    'class_weight': 'balanced',
    'dual': False
}

#K Nearest Neighbors  parameters
params_knn = {
    'n_neighbors':20, 
    'weights':'uniform',  #'distance','unform'
    'metric':'minkowski'
}

# Radius Nearest Neighbors parameters
params_rnn = {
    'radius': 4, 
    'weights':'uniform', #'distance'
    'outlier_label': 0
}

params_mlp = {
    'hidden_layer_sizes': (100, 100),
    'activation': 'relu', #'logistic','tanh',
    'solver': 'lbfgs', #'adam','sgd'
    'alpha': 10,
    'batch_size': 200,
    'learning_rate':'constant',#'invscaling','adaptive'
    'max_iter':200 # These are epochs
}


# eXtreme Gradient Boosting parameters
params_xgb = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mae',
    'nrounds': 350
}

#%% Define environment

# Class to extend the Sklearn classifier
class SklearnHelper():
    def __init__(self, clf, seed=False, params=None):
        if seed==False: self.clf = clf(**params)
        else : params['random_state'] = seed ; self.clf = clf(**params)

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def score(self,x,y_true):
        return self.clf.score(x,y_true)
    
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_

#    def get_params(self, params, deep=True):
#        return params
        
#    def set_params(self, **params):
#        for parameter, value in params.items():
#            setattr(self, parameter, value)
        
        
# Class to extend XGboost classifer    
class XgbHelper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def fit(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=np.log(y_train))
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return np.exp(self.gbdt.predict(xgb.DMatrix(x)))
        
# Function to get Out-of-fold predictions
def get_oof(clf, x_train, y_train, x_test):
    oof_train_pred = np.zeros((ntrain,))
    oof_test_pred = np.zeros((ntest,))
    oof_test_skf = np.empty((nfolds, ntest))
    oof_acc_skf = np.zeros((nfolds,))

    for i, (train_index, test_index) in enumerate(skf.split(x_train,y_train)):
        fold_x_train, fold_x_test = x_train[train_index], x_train[test_index]
        fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

        clf.fit(fold_x_train,fold_y_train)
        
        oof_train_pred[test_index] = clf.predict(fold_x_test)
        oof_test_skf[i, :] = clf.predict(x_test)
        oof_acc_skf[i]=accuracy_score(fold_y_test,oof_train_pred[test_index])
    
    oof_acc_skf_avg = np.mean(oof_acc_skf)    
    oof_acc_skf_std = np.std(oof_acc_skf)
    
    oof_test_pred[:] = np.median(oof_test_skf, axis=0)
    oof_acc= accuracy_score(y_train, oof_train_pred)
    
    #print('Kfold accuracy average:',oof_acc_skf_avg,'standard dev:',oof_acc_skf_std)
    print('Accuracy score :',oof_acc)
    return oof_train_pred, oof_test_pred #oof_train_pred.reshape(-1, 1), oof_test.reshape(-1, 1)

def get_score(clf,x_train,y_train):
    acc_skf = np.zeros((nfolds,))
    
    for i, (train_index, test_index) in enumerate(skf.split(x_train,y_train)):
        fold_x_train, fold_x_test = x_train[train_index], x_train[test_index]
        fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]
        
        model_svclin.fit(fold_x_train, fold_y_train)
        
        acc_skf[i] = model_svclin.score(fold_x_test,fold_y_test)
    
    acc = round(np.mean(acc_skf),4)
    return acc
        

#%% Models
# Create objects that represent our models
# Ensamble
model_rf = SklearnHelper(clf=RandomForestClassifier, seed=nseed, params=params_rf)
model_et = SklearnHelper(clf=ExtraTreesClassifier, seed=nseed, params=params_et)
model_ada = SklearnHelper(clf=AdaBoostClassifier, seed=nseed, params=params_ada)
model_gb = SklearnHelper(clf=GradientBoostingClassifier, seed=nseed, params=params_gb)
# Suppoer Vector Machines
model_svc = SklearnHelper(clf=SVC, seed=nseed, params=params_svc)
model_linsvc = SklearnHelper(clf=LinearSVC, seed=nseed, params=params_linsvc)
# Neighbors
model_rnn = SklearnHelper(clf=RadiusNeighborsClassifier, seed=False, params=params_rnn)
model_knn = SklearnHelper(clf=KNeighborsClassifier, seed=False, params=params_knn)
# Neural Nets
model_mlp = SklearnHelper(clf=MLPClassifier, seed=nseed, params=params_mlp)

#misc
#model_xgb = XgbHelper(seed=nseed, params=params_xgb)

#%% Predictions

# Create our OOF train and test predictions. These base results will be used as new features

print('Extra Trees')
oof_train_et, oof_test_et = get_oof(model_et, x_train, y_train, x_test) 
print('Random Forest')
oof_train_rf, oof_test_rf = get_oof(model_rf, x_train, y_train, x_test) 
print('AdaBoost')
oof_train_ada, oof_test_ada = get_oof(model_ada, x_train, y_train, x_test)
print('Gradient Boosting') 
oof_train_gb, oof_test_gb = get_oof(model_gb, x_train, y_train, x_test) 
print('Support Vector Gaussian')
oof_train_svc, oof_test_svc = get_oof(model_svc, x_train, y_train, x_test) 
print('Support Vector Linear')
oof_train_linsvc, oof_test_linsvc = get_oof(model_linsvc, x_train, y_train, x_test) 
print('K Nearest Neighbors')
oof_train_knn, oof_test_knn = get_oof(model_knn, x_train, y_train, x_test)
print('Radius Nearest Neighbors')
oof_train_rnn, oof_test_rnn = get_oof(model_rnn, x_train, y_train, x_test)
print('Neural Network')
oof_train_mlp, oof_test_mlp = get_oof(model_mlp, x_train, y_train, x_test)


#print('eXtreme Gradient Boosting') 
#oof_train_xgb, oof_test_xgb = get_oof(model_xgb, x_train, y_train, x_test) 
print("Training is complete")


#%% Tune parameters
findpar = ['auto','sqrt','log2']
findval = np.linspace(1,17,17)
for i in findpar:   
    params_rf = {
        'n_estimators': 500,
        'max_features':i, # auto, sqrt, log2, 0.2
        'max_depth': 10,    #influences overfitting
        'min_samples_leaf': 2,
        'min_samples_split': 2,    
        'warm_start': False, 
        'n_jobs': -1  
    }
    model_rf = SklearnHelper(clf=RandomForestClassifier, seed=nseed, params=params_rf)
    print(i)
    oof_train_rf, oof_test_rf = get_oof(model_rf, x_train, y_train, x_test) 


#%% Feature importances
features_rf = model_rf.feature_importances(x_train,y_train)
features_et = model_et.feature_importances(x_train, y_train)
features_ada = model_ada.feature_importances(x_train, y_train)
features_gb = model_gb.feature_importances(x_train,y_train)

#%%

# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': train.columns.values,
     'Random Forest feature importances': features_rf,
     'Extra Trees  feature importances': features_et,
      'AdaBoost feature importances': features_ada,
    'Gradient Boost feature importances': features_gb
    })
#%% Feature importance plot (interactive)
# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Create the new column containing the average of values

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe.head(3)

y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')

#%% Second level predictions

base_predictions_train = pd.DataFrame( {
    'RandomForest': rf_oof_train.ravel(),
    'ExtraTrees': et_oof_train.ravel(),
    'AdaBoost': ada_oof_train.ravel(),
    'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()

#%% PLot correlation
data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')

#%% Concentrate predictions
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train),axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

#%% Second level model
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,                        
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                        intercept_scaling=1, class_weight=None, random_state=None, solver='warn', 
                        max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None)

#%% Save solution
# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)



#%% Misc
from collections import Counter
b=Counter(y_train)
b.values()
b.keys()
