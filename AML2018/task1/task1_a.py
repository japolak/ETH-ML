#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:27:35 2018

@author: leslie
"""
#%%

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing.imputation import Imputer
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge


#### SET WORK FOLDER - ADD YOURS AND WE CAN COMMENT EACH OTHER'S OUT

#WORK_FOLDER = '/home/leslie/Desktop/AML/Task1/'
#WORK_FOLDER = FEDE
WORK_FOLDER = '/Users/Jakub/Documents/ETH/AML/task1/'



#### LOAD DATA

x_train_data = pd.read_csv(WORK_FOLDER + "X_train.csv", dtype=np.float64)
y_train_data = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
x_test_data = pd.read_csv(WORK_FOLDER + "X_test.csv", dtype=np.float64)



#### PREPARE DATA

# Remove ID column from x train
col_names_x = x_train_data.columns.get_values()
col_names_x = col_names_x.tolist()
features = col_names_x[1:]
x_train_data = x_train_data[features]

# Remove id column from y train
col_names_y = y_train_data.columns.get_values()
col_names_y = col_names_y.tolist()
response = col_names_y[1:]
y_train_data = y_train_data[response]

# Remove id from x test and get ID column (to make the CSV later)
ids = col_names_x[0]
id_data = x_test_data[ids]
x_test_data = x_test_data[features]



#### CLEAN DATA

# replace missing values (may want to change approach, here I just impute
# missing values using column medians - probably a better way)

# Check initial number of NAs in data (replace with whichever dataset you want)
#print(y_train_data.isna().sum())

imp = Imputer(missing_values='NaN', strategy='median', axis=0)

x_train_data = imp.fit_transform(x_train_data)
x_train_data = pd.DataFrame(data=x_train_data)

x_test_data = imp.fit_transform(x_test_data)
x_test_data = pd.DataFrame(data=x_test_data)

y_train_data = imp.fit_transform(y_train_data)
y_train_data = pd.DataFrame(data=y_train_data)

# Confirm no more NAs in a particular dataframe:
#print(y_train_data.isna().sum())



#### SET UP CROSS VALIDATION: double CV to 1) choose lambda parameter and 2) estimate Expected Test MSE


# create outer folds manually (sorry this is so ugly...still adjusting to python)
kfolds = np.repeat(1, len(x_train_data)/5+1).tolist() + np.repeat(2, len(x_train_data)/5+1).tolist() + np.repeat(3, len(x_train_data)/5).tolist() + np.repeat(4, len(x_train_data)/5).tolist() + np.repeat(5, len(x_train_data)/5).tolist()
index = np.random.choice(kfolds, size=len(x_train_data), replace=False)

chosen_alpha = []
expected_r2 = []

iter = [1,2,3,4,5]
for i in iter:
    x_train = x_train_data[index!=i]
    y_train = y_train_data[index!=i]

    x_test = x_train_data[index==i]
    y_test = y_train_data[index==i]

    # find optimal lambda using built in CV for inner CV
    alphas = 10**np.linspace(10,-2,200)

    ridgecv = RidgeCV(alphas=alphas, scoring="r2", normalize=True)
    ridgecv.fit(x_train, y_train)
    chosen_alpha.append(ridgecv.alpha_)

    ridge_reg = Ridge(alpha=ridgecv.alpha_, normalize=True)
    ridge_reg.fit(x_train, y_train)

    y_pred = ridge_reg.predict(x_test)
    expected_r2.append(r2_score(y_test, y_pred))


print("expected r^2:", np.mean(expected_r2))
best_alpha = chosen_alpha[expected_r2.index(min(expected_r2))]



#### FIT CHOSEN MODEL ON FULL DATA AND PREDICT Y

final_model = Ridge(alpha = best_alpha, normalize=True)
final_model.fit(x_train_data, y_train_data)

y_pred_test = final_model.predict(x_test_data)
y_pred_test = pd.DataFrame(data=y_pred_test)
y_pred_test.columns = ["y"]

y_predictions = y_pred_test["y"]


##### EXPORT TO CSV

# prepare results data frame

y_test_pred = pd.DataFrame({"id": id_data, "y": y_predictions})

# save results to csv
y_test_pred.to_csv(WORK_FOLDER + "task1.csv", index=False)
