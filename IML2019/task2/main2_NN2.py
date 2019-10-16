# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 07:29:10 2019

@author: Raphael Egli
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier

raw= pd.read_csv('train.csv')

# shuffle, split, scale
raw = raw.sample(frac=1).reset_index(drop=True)

Y = raw.loc[:,'y']
Y_test = raw.loc[:399,'y']
Y_train = raw.loc[400:,'y']
X = raw.loc[:,'x1':]

# Scale X data (since Y is label): mean=0, sd=1
X_s = scale(X, axis=0, with_mean=True, with_std=True, copy=True)
X_s.mean(axis=0) # verify mean
X_s.std(axis=0) # verify std

X_test = X_s[:400,:]
X_train = X_s[400:,:]

# split into test and train


# Consider:
# - scaling before use of NN
# - use CV to assess generality
# - don't overfit the NN (architecture/run-time)



# define KFolds
rkf = RepeatedKFold(n_splits=10, n_repeats=20, random_state=None)

# Setup the NN
NN = MLPClassifier(activation='relu', solver='sgd', max_iter=1000, batch_size='auto', shuffle=True, verbose=False, warm_start=False)

# Setup optimizer (Grid-Search) and the Grid

p_grid = [{'hidden_layer_sizes': [(50,50)], 'tol': [1e-5, 1e-6, 1e-7], 'alpha': [1e0, 1e-1, 1e-2]}]

gs = GridSearchCV(estimator=NN, param_grid=p_grid, scoring='accuracy', n_jobs=-1, iid=True, refit=True, cv=rkf, verbose=1, error_score='raise')

# fit the model
gs.fit(X_train, Y_train)
print(sorted(gs.cv_results_))
print(gs.best_estimator_)

gs_one = gs.best_estimator_
gs_one.fit(X_train, Y_train)
# predict Y_hat
Y_hat = gs_one.predict(X_test)

acc=sum(Y_hat==Y_test)/(len(Y_hat))

file = open("Task2/model.txt", 'w')
file.write(str(acc))
file.write(str(gs_one)) 
file.close()

# best model from CV grid search
gs_one= MLPClassifier(activation='relu', alpha=1.0, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(50, 50), learning_rate='constant',
       learning_rate_init=0.001, max_iter=10000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='sgd', tol=1e-06, validation_fraction=0.1,
       verbose=False, warm_start=False)

# import Test.csv and predict classes for result
# import Test.csv
X_task = pd.read_csv('Task2/test.csv', sep=',',header=0, index_col=0)
# Scale (since modle is from scaled data aswell)
X_task_s = scale(X_task, axis=0, with_mean=True, with_std=True, copy=True)
X_task_s.mean(axis=0) # verify mean
X_task_s.std(axis=0) # verify std
# predict class
gs_one.fit(X_s,Y)
Y_hat = gs_one.predict(X_task_s)
ident = [x for x in range(2000,5000)]

sol = pd.DataFrame(Y_hat,ident)
sol.columns = ['y']

# save class in csv
sol.to_csv("Task2/sol_ANN.csv", header=["y"])



