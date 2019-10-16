#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:27:35 2018

@author: leslie
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
from sklearn.model_selection import train_test_split
from sklearn import linear_model, ensemble
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler, Imputer

#### SET WORK FOLDER

# WORK_FOLDER = '/home/leslie/Desktop/AML/Task2/'
# WORK_FOLDER = '/home/fred/Documents/ETH/HS2018/Advanced Machine Learning/Projects/Task2/task2-data/'
WORK_FOLDER = '/Users/Jakub/Documents/ETH/AML/task2/'

def balanced_accuracy_score(y_true, y_pred, sample_weight=None,adjusted=False):
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score

def prepare_data():
    # function to prepare the data & impute any missing values
    # load data
    x_train_data = pd.read_csv(WORK_FOLDER + "X_train.csv", dtype=np.float64)
    y_train_data = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
    x_test_data = pd.read_csv(WORK_FOLDER + "X_test.csv", dtype=np.float64)
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
    ## Scale data
    scaler = StandardScaler()
    scaler.fit(x_train_data)
    x_train_data = scaler.transform(x_train_data)
    x_test_data = scaler.transform(x_test_data)
    # convert to pandas dataframe
    x_train_data = pd.DataFrame(data=x_train_data)
    x_test_data = pd.DataFrame(data=x_test_data)
    y_train_data = pd.DataFrame(data=y_train_data)
    # return the data in a tuple
    return(x_train_data, y_train_data, x_test_data, id_data)


"""
Clean data:
    replace missing values (may want to change approach, here I just impute
    missing values using column medians - probably a better way)
    To begin with we check the initial number of NAs in data (replace with whichever dataset you want)
    # print(y_train_data.isna().sum())
@param 'MD_output': Output from Mahala1D
@param 'x_df': data frame with predictors
@output: data frame without outliers
@author: Leslie (reworked by Fede-> function delivers same result!)
"""
def impute_median(data_to_clean):
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    data_to_clean= imp.fit_transform(data_to_clean)
    data_to_clean = pd.DataFrame(data=data_to_clean)
    return data_to_clean

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



def svm_cv_function(number_runs = 5, kernel = "poly", degree=3, gamma=2, decision_function_shape="ovo", class_weight={0:6, 1:1, 2:6}):
	# REPEATED VALIDATION SET APPROACH
	BMAC = []
	for run in range(number_runs):
		print(run)
		X_train, X_test, y_train, y_test = train_test_split(x_train_data, y_train_data, test_size=0.25)
		# training a linear SVM classifier
		svm_model = SVC(kernel = kernel, gamma=gamma, degree=degree, C = 1, decision_function_shape=decision_function_shape, class_weight=class_weight).fit(X_train, y_train)
		y_pred = svm_model.predict(X_test)
		print(balanced_accuracy_score(y_test, y_pred))
		BMAC.append(balanced_accuracy_score(y_test, y_pred))
	print(np.mean(BMAC))

"""
Fitting Multinomial Logistic Regression
@param 'x_Tr':training predictors
@param 'y_Tr':training response
@param 'num_runs': Why included? The function already uses CV
@param 'num_estimators': Parameter is not used
@param 'solver': optimizer to use
@param 'class_weight': re-weighting classes that are less represented in the data
@param 'max_iter': max number of iterations (for what?)
@output: mean BMAC
@author: Leslie
"""
def logreg_cv_function(x_Tr, y_Tr, number_runs = 5, num_estimators=100, solver="lbfgs", class_weight={0:6, 1:1, 2:6}, max_iter=1000):
	#### REPEATED VALIDATION SET APPROACH
	BMAC = []
	for run in range(number_runs):
		print(run)
		X_train, X_test, y_train, y_test = train_test_split(x_Tr, y_Tr, test_size=0.1)
		clf = LogisticRegressionCV(cv=5, multi_class='multinomial', solver=solver, max_iter=max_iter, class_weight=class_weight).fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		BMAC.append(balanced_accuracy_score(y_test, y_pred))
	print(BMAC)



"""
Fitting Multinomial Logistic Regression and saving the predictions
FOR SVM (commented out)
@param 'kernel':
@param 'gamma':
@param 'degree':
@param 'decision_function_shape':
FOR MULTINOMIAL LOG.
@param 'solver': optimizer to use
@param 'class_weight': re-weighting classes that are less represented in the data
@param 'max_iter': max number of iterations (for what?)
@output: file containing the predicted values for the test 'y'
@author: Leslie
"""
def fit_final_model(kernel="poly", gamma=2, degree=3, decision_function_shape="ovo", class_weight={0:6, 1:1, 2:6},
	solver="lbfgs", max_iter=1000):
	# fit chosen model on full data and predict y
	final_model = LogisticRegressionCV(cv=5, multi_class='multinomial', solver=solver, max_iter=max_iter, class_weight=class_weight).fit(x_train_data, y_train_data)
	#final_model = SVC(kernel = kernel, gamma=gamma, degree=degree, C = 1, decision_function_shape=decision_function_shape, class_weight=class_weight).fit(x_train_data, y_train_data)
	y_pred_test = final_model.predict(x_test_data)
	y_pred_test = pd.DataFrame(data=y_pred_test)
	y_pred_test.columns = ["y"]
	y_predictions = y_pred_test["y"]
	## prepare results data frame
	y_test_pred = pd.DataFrame({"id": id_data, "y": y_predictions})
	## save results to csv
	y_test_pred.to_csv(WORK_FOLDER + "task2.csv", index=False)


"""
We go over an array (which we sorted beforehand), until the sum >= precision treshold desired
@param 'num_list': an array of numbers (previously sorted from largest to smallest!)
@param 'limit': what treshhold do we set?
@output: smallest value that a variable contributes to reach the "limit"
@author: stackexchange (modified by Fede)
"""
def sum_to_x(num_list, limit=0.9):
    i = 0
    while sum(num_list[:i]) < limit:
        i += 1
    return num_list[i]

"""
Variable selection using random forest
@param 'x_Tr': training data, predictors
@param 'y_Tr': training data, response
@param 'x_Te': test data, predictors
@param 'nr_trees': nr of trees for random forest
@param 'class_weights': same as those by Leslie
@param 'lm': what treshhold do we set (for importance criterion)
@output: data frames with important X for train and test data [moreover list of booleans with keep= True | False]
@author: https://chrisalbon.com/machine_learning/trees_and_forests/feature_selection_using_random_forest/ (modified by Fede)
"""
def varSelection(x_Tr, y_Tr, x_Te, n_trees=1000, crit="gini", class_weights={0:6, 1:1, 2:6}, lm=0.75):
    # Train the classifier with all the variables
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=0, n_jobs=-1,criterion=crit, class_weight=class_weights )
    clf.fit(x_Tr, y_Tr)
    # Which is the last "useful" variable?
    smallest_useful_contribution= sum_to_x(sorted(clf.feature_importances_, reverse=True), limit=lm)
    # Columns to keep
    keep=clf.feature_importances_ > smallest_useful_contribution
    # Selecting a subset of the predictors
    sfm = SelectFromModel(clf, threshold=smallest_useful_contribution)
    sfm.fit(x_Tr, y_Tr)
    X_important_train = sfm.transform(x_Tr)
    X_important_test = sfm.transform(x_Te)
    return X_important_train, X_important_test, keep

"""
Prediction using random forest
@param 'x_imp_train': training data, predictors (pre-selected features)
@param 'y_Tr': training data, response
@param 'X_imp_test': test data, predictors (pre-selected features)
@param 'y_Te': test data, response (we assume we have that--> the function will be called in a Cross-validation routine)
@param 'nr_trees': nr of trees for random forest
@param 'class_weights': same as those by Leslie
@output: Score of predictions
@author: https://chrisalbon.com/machine_learning/trees_and_forests/feature_selection_using_random_forest/ (modified by Fede)
"""
def randForPred(x_imp_train, X_imp_test, y_Tr, y_Te, n_trees=1000, class_weights={0:6, 1:1, 2:6}):
    # New classifier using merely "useful" variables
    clf_important = RandomForestClassifier(n_estimators=n_trees, random_state=0, n_jobs=-1, class_weight=class_weights)
    clf_important.fit(x_imp_train, y_Tr)
    y_pred = clf_important.predict(X_imp_test)
    return balanced_accuracy_score(y_Te, y_pred)

"""
CV_method
routine for variable selection.
@param 'nr__CV'
@param 'x_data': whole training set predictors
@param 'y_data': whole training set response
@output: list with variables to keep
@author: modified version of hand-designed CV by Leslie
"""
def CV_method(nr_CV, x_data, y_data):
    #Saving Key parameters for the methods implemented
    output= []
    p=len(x_data.columns)
    for i in nr_CV:
        # CV strategy, the same across all methods we apply
        x_train = x_data[index!=i]
        y_train = y_data[index!=i]
        x_test = x_data[index==i]
        y_test = y_data[index==i]
        VarSel= varSelection(x_Tr= x_train, y_Tr=y_train, x_Te=x_test)
        X_train_imp=VarSel[0]
        X_test_imp= VarSel[1]
        Cols_to_keep=VarSel[2]
        output.append(Cols_to_keep)
    keep=[]
    for a in range(0,p):
        if output[1][a]==True or output[2][a]==True or  output[3][a]==True or  output[4][a]==True or \
        output[5][a]==True or output[6][a]==True or  output[7][a]==True or  output[8][a]==True or \
        output[9][a]==True or output[0][a]==True :
            keep.append(True)
        else:
            keep.append(False)
    return keep

"""
CV_randFor
routine for variable selection.
@param 'nr__CV'
@param 'x_data': training set predictors (with pre-filtered features!)
@param 'y_data': whole training set response
@output: list with scores for
@author: modified version of hand-designed CV by Leslie
"""

def CV_randFor(nr_CV, x_data, y_data):
    #Saving Key parameters for the methods implemented
    output= []
    for i in nr_CV:
        # CV strategy, the same across all methods we apply
        x_train = x_data[index!=i]
        y_train = y_data[index!=i]
        x_test = x_data[index==i]
        y_test = y_data[index==i]
        res= randForPred(x_imp_train=x_train, X_imp_test=x_test, y_Tr=y_train, y_Te=y_test)
        output.append(res)
    return output


"""
###############################################
# RUN FUNCTIONS
###############################################
"""

"""
Prepare the data & return in usable format +
Cleaning the data (remove outliers and impute using median)
"""

x_train_data, y_train_data, x_test_data, id_data = prepare_data()

resOut_train=Mahala1D(x_train_data)
resOut_test=Mahala1D(x_test_data)

x_train_data=Outlier_remove(resOut_train,x_train_data)
x_test_data=Outlier_remove(resOut_test,x_test_data)

x_train_data=impute_median(x_train_data)
x_test_data=impute_median(x_test_data)

"""
CROSS VALIDATION:
    1. One routine for variable selection
    2. Another for random forest evaluation (capacity to generalize well?)
"""
# create outer folds manually (sorry this is so ugly...still adjusting to python)
nr_obs= len(x_train_data)
kfolds = np.repeat(1,nr_obs/10).tolist() + np.repeat(2,nr_obs/10).tolist() + np.repeat(3,nr_obs/10).tolist() + np.repeat(4,nr_obs/10).tolist() + np.repeat(5,nr_obs/10).tolist() + np.repeat(6,nr_obs/10).tolist() + np.repeat(7,nr_obs/10).tolist() + np.repeat(8,nr_obs/10).tolist() + np.repeat(9,nr_obs/10).tolist() + np.repeat(10,nr_obs/10).tolist()
index = np.random.choice(kfolds, size=len(x_train_data), replace=False)
iter_nr = range(1, max(kfolds)+1)

# y_train_data=np.ravel(y_train_data)

# Variable selection
to_keep=CV_method(nr_CV=iter_nr, x_data=x_train_data, y_data=y_train_data)
x_relevant_train=x_train_data.iloc[:, to_keep]
x_relevant_test=x_test_data.iloc[:, to_keep]


"""
COMPARING DIFFERENT METHODS' PERFORMANCE
run CV on different parameters (which ones?)
--> Multiple Logistic without variable selection
"""
# Multiple Logistic regression, with ALL predictors
logreg_cv_function(x_Tr=x_train_data, y_Tr=y_train_data, number_runs=1)
# random forest, with ALL predictors
CV_randFor(nr_CV=iter_nr, x_data=x_train_data, y_data=y_train_data)
# Support Vector Machines
svm_cv_function(number_runs=5, degree=3)


# random forest, with feature selection
CV_randFor(nr_CV=iter_nr, x_data=x_relevant_train, y_data=y_train_data)
# Multiple Logistic regression, with pre-selected variables
logreg_cv_function(x_Tr=x_relevant_train, y_Tr=y_train_data, number_runs=1)


"""
refit model to full data and save to csv
"""
final_model = LogisticRegressionCV(cv=5, multi_class='multinomial', max_iter=1000, class_weight={0:6, 1:1, 2:6}).fit(x_train_data, y_train_data)

y_pred_test = final_model.predict(x_test_data)
y_pred_test = pd.DataFrame(data=y_pred_test)
y_pred_test.columns = ["y"]
y_predictions = y_pred_test["y"]

## prepare results data frame
y_test_pred = pd.DataFrame({"id": id_data, "y": y_predictions})

## save results to csv
y_test_pred.to_csv(WORK_FOLDER + "task2_ML_b.csv", index=False)
