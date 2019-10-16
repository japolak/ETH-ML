#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:27:35 2018
This script implements Elastic Nets to:
    1. do variable selection
    2. come up with predicted values for y
This strategy was chosen over Ridge, because we wanted many of the coefficients to be EXACTLY 0
and over Lasso, because the latter encountered a few issues (see below)
@author: Leslie and Fede
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import os
from sklearn.linear_model import RidgeCV, Ridge,  LassoCV, Lasso, ElasticNetCV, ElasticNet
from sklearn.preprocessing.imputation import Imputer
from sklearn import linear_model
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.datasets import make_regression

#### SET WORK FOLDER - ADD YOURS AND WE CAN COMMENT EACH OTHER'S OUT
# WORK_FOLDER = '/home/leslie/Desktop/AML/Task1/'
# WORK_FOLDER = '/home/fred/Documents/ETH/HS2018/Advanced Machine Learning/Projects/Task1/task1-data/'
WORK_FOLDER = '/Users/Jakub/Documents/ETH/AML/task1/'


#### LOAD DATA

x_train_data = pd.read_csv(WORK_FOLDER + "X_train.csv", dtype=np.float64)
y_train_data = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
x_test_data = pd.read_csv(WORK_FOLDER +  "X_test.csv", dtype=np.float64)


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


"""
Clean data:
    replace missing values (may want to change approach, here I just impute
    missing values using column medians - probably a better way)
    To begin with we check the initial number of NAs in data (replace with whichever dataset you want)
    # print(y_train_data.isna().sum())

@param 'MD_output': Output from Mahala1D
@param 'x_df': data frame with predictors
@output: data frame without outliers
@author: Leslie (reworked by Fede-_> function delivers same result!)
"""
def impute_median(data_to_clean):
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    data_to_clean= imp.fit_transform(data_to_clean)
    data_to_clean = pd.DataFrame(data=data_to_clean)
    return data_to_clean

x_train_data=impute_median(x_train_data)

x_test_data=impute_median(x_test_data)

y_train_data=impute_median(y_train_data)

# Confirm no more NAs in a particular dataframe:
#print(y_train_data.isna().sum())

"""
# Plotting predicors (Commented out)

import plotly
import plotly.graph_objs as go
import numpy as np
plotly.offline.plot({
    "data": [go.Histogram(x=x_train_data[1])]
}, auto_open=True)

# "Seems normal"--> I assume all predictors are!
# The Mahalanobis distance can be considered VERY informative in this context
"""

"""
Mahalanobis distance (for outlier detection)
# From: http://kldavenport.com/mahalanobis-distance-and-outliers/
covariance_xs = np.cov(x_train_data, rowvar=0)
inv_covariance_xs = np.linalg.inv(covariance_xs)
# Issue: cannot take the inverse! Too large
# of a dimensionality of our predictors-->
# do variable selection first?
# No! Probably they onl inserted univariate outliers, a 1D
# strategy should be sufficient!
"""


"""
1D Mahalanobis distance:
    We only consider univariate outliers.

@param 'x_df': data frame with all predictors
@output: a list of dimension p. Each element contains n entries
@author: Fede
"""
from scipy.stats import chi2

def Mahala1D(x_df):
    p= len(x_df.columns)
    mR=[]
    for i in range(p):
        subset_xdf=x_df[i]
        Sx = subset_xdf.var()
        mean = subset_xdf.mean()
        md=((subset_xdf-mean)/(Sx**0.5))**2
        p_vals= 1 - chi2.cdf(md, df=1)
        mR.append(p_vals)
    return mR

resOut_train=Mahala1D(x_train_data)
resOut_test=Mahala1D(x_test_data)

"""
Outlier elimination:
    We take the output from Mahala1D.
    We remove all observations whose p-value (considering the
    approximate Chi2 that the squared MD follows ) is less than 1%
@param 'MD_output': Output from Mahala1D
@param 'x_df': data frame with predictors
@output: data frame without outliers
@author: Fede
"""

def Outlier_remove(MD_output, x_df):
    p= len(MD_output)
    for i in range(p):
        x_df.iloc[MD_output[i]<0.01,i]=np.NaN
    return x_df

x_train_data=Outlier_remove(resOut_train,x_train_data)
x_test_data=Outlier_remove(resOut_test,x_test_data)

x_train_data=impute_median(x_train_data)

x_test_data=impute_median(x_test_data)

# Procedure seems to work!
# print(sum(x_train_data.isna().sum()))


#### SET UP CROSS VALIDATION: double CV to 1) choose lambda parameter and 2) estimate Expted Test MSE

# create outer folds manually (sorry this is so ugly...still adjusting to python)
kfolds = np.repeat(1, len(x_train_data)/5+1).tolist() + np.repeat(2, len(x_train_data)/5+1).tolist() + np.repeat(3, len(x_train_data)/5).tolist() + np.repeat(4, len(x_train_data)/5).tolist() + np.repeat(5, len(x_train_data)/5).tolist()
index = np.random.choice(kfolds, size=len(x_train_data), replace=False)
iter = range(1, max(kfolds)+1)
def CV_method(method, nr_CV, x_data):
    #Saving Key parameters for the methods implemented
    p=len(x_data.columns)
    chosen_alpha= []
    expected_r2= []
    estimated_beta =[]
    alphas = 10**np.linspace(10,-2,200)
    for i in nr_CV:
        """
        CV strategy, the same across all methods we apply
        """
        x_train = x_data[index!=i]
        y_train = y_train_data[index!=i]
        x_test = x_data[index==i]
        y_test = y_train_data[index==i]
        # find optimal lambda using built in CV for inner CV
        """
        METHOD SPECIFIC CODE
        """
        if method.lower()=="ridge":
            """
            Ridge
            @ author: Leslie
            """
            ridgecv = RidgeCV(alphas=alphas, scoring="r2", normalize=True)
            ridgecv.fit(x_train, y_train)
            chosen_alpha.append(ridgecv.alpha_)
            ridge_reg = Ridge(alpha=ridgecv.alpha_, normalize=True)
            ridge_reg.fit(x_train, y_train)
            y_pred = ridge_reg.predict(x_test)
            expected_r2.append(r2_score(y_test,y_pred ))
            estimated_beta.append(ridge_reg.coef_)
        elif method.lower()=="lasso":
            """
            Lasso
            https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/
            Seems to have problems! It is not unheard of that all betas are estimated to be zero (as in our case).
            This seems rather due to data features than to mis-coding...
            @ author: Fede
            """
            #Initialize the dataframe to store coefficients
            lassocv = LassoCV(cv=5)
            lassocv.fit(x_train,y_train)
            chosen_alpha.append(lassocv.alpha_)
            lasso_reg = Lasso(alpha = lassocv_alpha, normalize=True)
            lasso_reg.fit(x_train,y_train)
            y_pred = reg.predict(x_test)
            expected_r2.append(r2_score(y_test,y_pred ))
            estimated_beta.append(lasso_reg.coef_)
        elif method.lower()=="elasticnet":
            """
            Elastic Net
            Seems more 'adaptive' than lasso or ridge
            @ author: Fede
            """
            x_train, y_train = make_regression(n_features=p, random_state=0)
            EnetCV = ElasticNetCV(cv=5, random_state=0)
            EnetCV.fit(x_train, y_train)
            enet_reg=ElasticNet(alpha=EnetCV.alpha_)
            enet_reg.fit(x_train, y_train)
            y_pred = enet_reg.predict(x_test)
            expected_r2.append(r2_score(y_test,y_pred ))
            chosen_alpha.append(EnetCV.alpha_)
            estimated_beta.append(enet_reg.coef_)
        else:
            print("Wrong method selected!")
    return {"alphas": chosen_alpha, "R^2": expected_r2, "coeffs": estimated_beta}
"""
##########################################################
##########################################################
Ridge Prediction
Comment on the result:
    an alpha of 0.01 seems far too small!
    We wan to be more restrictive on how many betas we have...
    Moreover the fact that alpha is ALWAYS = 0.01 seems suspicious.
    The negative R^2 (which from the model documentation seems clear should be allowed), are still a very bad sign...
"""
ridge_res=CV_method("Ridge",iter,x_train_data)

bestR2= np.mean(ridge_res["R^2"])
best_alpha = ridge_res["alphas"][ridge_res["R^2"].index(min(ridge_res["R^2"]))]

# Moreover we already can leverage the information gained to do a first " variable selection".
p= len(x_train_data.columns)
avrg_coeff=[]
coeff_to_keep=[]
for ind_beta in range(p):
    cv1=ridge_res["coeffs"][0].tolist()[0][ind_beta]
    cv2=ridge_res["coeffs"][1].tolist()[0][ind_beta]
    cv3=ridge_res["coeffs"][2].tolist()[0][ind_beta]
    cv4=ridge_res["coeffs"][3].tolist()[0][ind_beta]
    cv5=ridge_res["coeffs"][4].tolist()[0][ind_beta]
    avg_beta=(cv1+cv2+cv3+cv4+cv5)/5
    avrg_coeff.append(avg_beta)
    p_useful= abs(avg_beta)>0.01
    coeff_to_keep.append(p_useful)

x_train_useful=pd.DataFrame({})
x_test_useful=pd.DataFrame({})
count=0
for i in range(p):
    if coeff_to_keep[i]==True:
        x_train_useful[count]=x_train_data[i]
        x_test_useful[count]=x_test_data[i]
        count+=1

ridge_res_improved=CV_method("Ridge",iter,x_train_useful)
bestR2= np.mean(ridge_res_improved["R^2"])
best_alpha_improved = ridge_res_improved["alphas"][ridge_res_improved["R^2"].index(min(ridge_res_improved["R^2"]))]

final_model = Ridge(alpha = best_alpha_improved, normalize=True)
final_model.fit(x_train_useful, y_train_data)
y_pred_train = final_model.predict(x_train_useful)

r2_score(y_train_data, y_pred_train)
"""
Function returning the predicted y:
    We input a fitted model and the test predictors.
    We obtain the predicted values
@param: fitted model
@param: test independent variables
@output: predicted values for y
@author: Leslie (Fede transformed in function)
"""
def generate_pred(fitted_mod, x_test_data):
    y_pred_test = fitted_mod.predict(x_test_data)
    y_pred_test = pd.DataFrame(data=y_pred_test)
    y_pred_test.columns = ["y"]
    return y_pred_test

y_predictions =  generate_pred(final_model, x_test_useful)['y']

"""
Function saving observation id and  predicted y in .csv file
@param 'id_vals': id_values
@param  'y_pred': predicted y (from generate_pred)
@param 'file_name': how should we call the file?
@output: .csv file --> ready for upload
@author: Leslie
"""
def save_csv(id_vals, y_pred, file_name):
    y_test_pred = pd.DataFrame({"id": id_vals, "y": y_pred})
    y_test_pred.to_csv(WORK_FOLDER + str(file_name)+".csv", index=False)

save_csv(id_data, y_predictions, "task1_ridge")

"""
###############################################
###############################################
Lasso Prediction
Comment on the result: an alpha of 0.01 seems far too small!
We want to be more restrictive on how many betas we have, thus
"""
lasso_res=CV_method("Lasso",iter,x_train_data)

bestR2= np.mean(lasso_res["R^2"])
best_alpha = lasso_res["alphas"][lasso_res["R^2"].index(min(lasso_res["R^2"]))]
lasso_res["coeffs"]
# All coefficients are 0! No wonder though, with such a large alpha...

"""
Elastic Net
I have issues here as well... Why are all my predictions the same?
"""
enet_res=CV_method("ElasticNet",iter,x_train_data)

bestR2= np.mean(enet_res["R^2"])
best_alpha = enet_res["alphas"][enet_res["R^2"].index(min(enet_res["R^2"]))]

# Moreover we already can leverage the information gained to do a first " variable selection".

p= len(x_train_data.columns)
avrg_coeff=[]
coeff_to_keep=[]
for ind_beta in range(p):
    cv1=enet_res["coeffs"][0].tolist()[ind_beta]
    cv2=enet_res["coeffs"][1].tolist()[ind_beta]
    cv3=enet_res["coeffs"][2].tolist()[ind_beta]
    cv4=enet_res["coeffs"][3].tolist()[ind_beta]
    cv5=enet_res["coeffs"][4].tolist()[ind_beta]
    avg_beta=(cv1+cv2+cv3+cv4+cv5)/5
    avrg_coeff.append(avg_beta)
    p_useful= abs(avg_beta)>1
    coeff_to_keep.append(p_useful)

x_train_useful=pd.DataFrame({})
x_test_useful=pd.DataFrame({})
count=0
for i in range(p):
    if coeff_to_keep[i]==True:
        x_train_useful[count]=x_train_data[i]
        x_test_useful[count]=x_test_data[i]
        count+=1

enet_improved=CV_method("ElasticNet",iter,x_train_useful)
best_alpha_improved = enet_improved["alphas"][enet_improved["R^2"].index(min(enet_improved["R^2"]))]

final_model = ElasticNet(alpha = best_alpha_improved, normalize=True)
Enet_reg=final_model.fit(x_train_useful, y_train_data)
y_pred_train = Enet_reg.predict(x_train_useful)

r2_score(y_train_data, y_pred_train)

y_predictions =  generate_pred(final_model, x_test_useful)['y']

save_csv(id_data, y_predictions, "task1_enet")
