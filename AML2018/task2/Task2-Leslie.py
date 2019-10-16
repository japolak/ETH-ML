#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:27:35 2018

@author: leslie
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
import os
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import balanced_accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
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
"""


#### SET WORK FOLDER

# WORK_FOLDER = '/home/fred/Documents/ETH/HS2018/Advanced Machine Learning/Projects/Task2/task2-data/'
# WORK_FOLDER = FEDE
WORK_FOLDER = '/Users/Jakub/Documents/ETH/AML/task2/'

start = time. time()
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




def svm_cv_function(number_runs = 5, kernel = "poly", C=1, degree=3, gamma=2, decision_function_shape="ovo", class_weight={0:6, 1:1, 2:6}):
	#### REPEATED VALIDATION SET APPROACH
	BMAC = []
	for run in range(number_runs):
		print(run)
		X_train, X_test, y_train, y_test = train_test_split(x_train_data, y_train_data, test_size=0.1)
		# training a linear SVM classifier
		svm_model = SVC(kernel = kernel, gamma=gamma, degree=degree, C = C, probability=True, decision_function_shape=decision_function_shape, class_weight=class_weight).fit(X_train, y_train)
		probabil = svm_model.predict_proba(X_test)
		print(probabil.shape)
		print(probabil[1:5,1:5])
		pba = pd.DataFrame({'Class0': probabil[:,0], 'Class1': probabil[:,1], 'Class2': probabil[:,2] })
		pba.to_csv(WORK_FOLDER + "prob_svm.csv", index=False)
		# y_pred = svm_model.predict(X_test)
		# print(balanced_accuracy_score(y_test, y_pred))
		# BMAC.append(balanced_accuracy_score(y_test, y_pred))
	# print(np.mean(BMAC))

def final_model_probs(penalty="l2", class_weight={0:6, 1:1, 2:6}, max_iter=1000):

	#svm
	svm_model = SVC(kernel = 'rbf', gamma=0.001, C = 1, probability=True, decision_function_shape="ovo", class_weight=class_weight).fit(x_train_data, y_train_data)
	svm_prob = svm_model.predict_proba(x_test_data)

	# multinormial
	log_model = LogisticRegression(C=0.001, multi_class='multinomial', penalty=penalty, solver='sag', max_iter=max_iter, class_weight=class_weight).fit(x_train_data, y_train_data)
	log_prob = log_model.predict_proba(x_test_data)

	# lda
	lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=.9, priors=[1./8, 3./4, 1./8]).fit(x_train_data, y_train_data)
	lda_prob = lda_model.predict_proba(x_test_data)


	best_class = []
	for i in range(len(x_test_data)):

		if ((max(svm_prob[i,0], log_prob[i,0], lda_prob[i,0]) > max(svm_prob[i,1], log_prob[i,1], lda_prob[i,1])) and
			(max(svm_prob[i,0], log_prob[i,0], lda_prob[i,0]) > max(svm_prob[i,2], log_prob[i,2], lda_prob[i,2]))):
			best_class.append(0)
		if ((max(svm_prob[i,1], log_prob[i,1], lda_prob[i,1]) > max(svm_prob[i,0], log_prob[i,0], lda_prob[i,0])) and
			(max(svm_prob[i,1], log_prob[i,1], lda_prob[i,1]) > max(svm_prob[i,2], log_prob[i,2], lda_prob[i,2]))):
			best_class.append(1)
		if ((max(svm_prob[i,2], log_prob[i,2], lda_prob[i,2]) > max(svm_prob[i,0], log_prob[i,0], lda_prob[i,0])) and
			(max(svm_prob[i,2], log_prob[i,2], lda_prob[i,2]) > max(svm_prob[i,1], log_prob[i,1], lda_prob[i,1]))):
			best_class.append(2)

	y_test_pred = pd.DataFrame({"id": id_data, "y": best_class})

	## save results to csv
	y_test_pred.to_csv(WORK_FOLDER + "task2.csv", index=False)


def final_model_probs2(weights, penalty="l2", class_weight={0:6, 1:1, 2:6}, max_iter=1000):
    tot=sum(weights.values())
    #tot=weights['LDA']+ weights['MN']+ weights['SVM']
    #svm
    svm_model = SVC(kernel = 'rbf', gamma=0.001, C = 1, probability=True, decision_function_shape="ovo", class_weight=class_weight).fit(x_train_data, y_train_data)
    svm_prob = svm_model.predict_proba(x_test_data)
    # multinormial
    log_model = LogisticRegression(C=0.001, multi_class='multinomial', penalty=penalty, solver='sag', max_iter=max_iter, class_weight=class_weight).fit(x_train_data, y_train_data)
    log_prob = log_model.predict_proba(x_test_data)
    # lda
    lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=.9, priors=[1./8, 3./4, 1./8]).fit(x_train_data, y_train_data)
    lda_prob = lda_model.predict_proba(x_test_data)
    # neural nets
    nn_model = MLPClassifier(solver='lbfgs', activation='tanh', alpha=1e-6, learning_rate = 'adaptive', learning_rate_init =.0001, max_iter=50000,
			hidden_layer_sizes=(200,150)).fit(x_train_data,y_train_data)
    nn_prob = nn_model.predict_proba(x_test_data)
    # adaboost
    ada_model =AdaBoostClassifier(DecisionTreeClassifier(class_weight={0:6, 1:1, 2:6}, max_depth=2),n_estimators=600,learning_rate=0.0005).fit(x_train_data, y_train_data)
    ada_prob = ada_model.predict_proba(x_test_data)
    best_class = []
    for i in range(len(x_test_data)):
        prob0 = weights['SVM']/tot*svm_prob[i,0] + weights['MN']/tot*log_prob[i,0] + weights['LDA']/tot*lda_prob[i,0] + weights['NN']/tot*nn_prob[i,0] + weights['ADABOOST']/tot*ada_prob[i,0]

        prob1 = weights['SVM']/tot*svm_prob[i,1] + weights['MN']/tot*log_prob[i,1] + weights['LDA']/tot*lda_prob[i,1] + weights['NN']/tot*nn_prob[i,1] + weights['ADABOOST']/tot*ada_prob[i,1]

        prob2 = weights['SVM']/tot*svm_prob[i,2] + weights['MN']/tot*log_prob[i,2] + weights['LDA']/tot*lda_prob[i,2] + weights['NN']/tot*nn_prob[i,2] + weights['ADABOOST']/tot*ada_prob[i,2]

        if ((prob0>prob1) and (prob0>prob2)):
            best_class.append(0)
        elif ((prob1>prob0) and (prob1>prob2)):
            best_class.append(1)
        elif ((prob2>prob0) and (prob2>prob1)):
            best_class.append(2)
	y_test_pred = pd.DataFrame({"id": id_data, "y": best_class})
	## save results to csv
	y_test_pred.to_csv(WORK_FOLDER + "task2.csv", index=False)


def AdaProbs():
	ada_model =AdaBoostClassifier(DecisionTreeClassifier(class_weight={0:6, 1:1, 2:6}, max_depth=2),n_estimators=600,learning_rate=0.0005).fit(x_train_data, y_train_data)
	ada_prob = ada_model.predict_proba(x_test_data)
	ada_prob = pd.DataFrame({'Class0': ada_prob[:,0], 'Class1': ada_prob[:,1], 'Class2': ada_prob[:,2] })
	ada_prob.to_csv(WORK_FOLDER + "prob_ada.csv", index=False)


def logreg_cv_function(number_runs = 5, num_estimators=100, penalty="l2", learning_rate = 1, solver="saga", class_weight={0:6, 1:1, 2:6}, max_iter=1000):
	#### REPEATED VALIDATION SET APPROACH
	BMAC = []

	for run in range(number_runs):
		print(run)
		X_train, X_test, y_train, y_test = train_test_split(x_train_data, y_train_data, test_size=0.25)
		clf=AdaBoostClassifier(DecisionTreeClassifier(class_weight={0:6, 1:1, 2:6}, max_depth=2),n_estimators=600,learning_rate=learning_rate).fit(X_train, y_train)

		# clf = LogisticRegressionCV(cv=5, scoring="balanced_accuracy", multi_class='multinomial', penalty=penalty, solver=solver, max_iter=max_iter, class_weight=class_weight).fit(X_train, y_train)
		# clf = MLPClassifier(solver='lbfgs', activation='tanh', alpha=0.000001, learning_rate = 'adaptive', learning_rate_init =.0001, max_iter=50000,
		# 	hidden_layer_sizes=(150,150,150)).fit(X_train,y_train)
		print("still")
		y_pred = clf.predict(X_test)
		# print(balanced_accuracy_score(y_test, y_pred))
		BMAC.append(balanced_accuracy_score(y_test, y_pred))

	print(np.mean(BMAC))


def fit_final_model(kernel="poly", n_trees=50, C=1, crit="gini", gamma=2, degree=3, penalty="l2", decision_function_shape="ovo", class_weight={0:6, 1:1, 2:6},
	solver="lbfgs", max_iter=1000):

	# fit chosen model on full data and predict y
	# final_model = LogisticRegressionCV(cv=5, multi_class='multinomial', scoring="balanced_accuracy", penalty=penalty, solver=solver, max_iter=max_iter, class_weight=class_weight).fit(x_train_data, y_train_data)
	final_model = SVC(kernel = kernel, gamma=gamma, degree=degree, C = 1, decision_function_shape=decision_function_shape, class_weight=class_weight).fit(x_train_data, y_train_data)
	#final_model = LogisticRegression(C=0.001, multi_class='multinomial', penalty=penalty, solver=solver, max_iter=max_iter, class_weight=class_weight).fit(x_train_data, y_train_data)
	# final_model3 = MLPClassifier(solver='lbfgs', activation='tanh', alpha=0.01, learning_rate = 'adaptive', learning_rate_init =.0001, max_iter=50000,
			# hidden_layer_sizes=(200,150)).fit(x_train_data,y_train_data)
	# final_model4 = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1,criterion=crit, class_weight=class_weight).fit(x_train_data,y_train_data)
	end = time. time()
	print(start-end)

	y_pred_test = final_model.predict(x_test_data)
	# y_pred_test2 = final_model2.predict(x_test_data)
	# y_pred_test3 = final_model3.predict(x_test_data)
	# y_pred_test4 = final_model4.predict(x_test_data)
	# y_pred_test = pd.DataFrame({"data1": y_pred_test1, "data2": y_pred_test2, "data3": y_pred_test3, "data4": y_pred_test4})
	y_pred_test = pd.DataFrame(data=y_pred_test)
	print("e")
	y_pred_test.columns = ["y"]
	y_predictions = y_pred_test["y"]

	## prepare results data frame
	y_test_pred = pd.DataFrame({"id": id_data, "y": y_predictions})

	## save results to csv
	y_test_pred.to_csv(WORK_FOLDER + "task2.csv", index=False)



###############################################
# RUN FUNCTIONS
###############################################

#weights contains the scores obtained on the public score!
# 1 : LDA with shrinkage 9 (result 26)
# 2 : Multinomial (LogisticRegression( multi_class=...)) -result 22
# 3 : SVM  -result 21-
# 4 : ada boost -result 28-
# 5 : NN -result 10-
"""
weights={'LDA': 0.687227460209, 'MN':0.700972564691,
         'SVM': 0.682049110019, 'ADABOOST': 0.612106861644,
         'NN': 0.596827775283 }
"""

weights={'LDA': 1./5, 'MN':1./5,
         'SVM':1./5, 'ADABOOST': 1./5,
         'NN': 1./5 }

# prepare the data & return in usable format
x_train_data, y_train_data, x_test_data, id_data = prepare_data()



# degrees_to_try = np.arange(1, 6)

# for i in range(len(degrees_to_try)):
# 	print(i+1)
# 	tmp_model = svm_cv_function(kernel="poly", decision_function_shape="ovr",number_runs=5, degree=degrees_to_try[i+1])




# # x_train_data, x_test_data, keep = varSelection(x_train_data, y_train_data, x_test_data, lm=0.6)

# end = time. time()
# print(end - start)
# # run CV on different parameters
start = time. time()
# learns = [0.0001, 0.00001, 0.000001, 0.0005]
# for i in range(4):
# 	print(learns[i])
# 	logreg_cv_function(number_runs=1, learning_rate = learns[i],class_weight={0:6, 1:1, 2:6})
final_model_probs2(weights=weights)
# AdaProbs()
# svm_cv_function(number_runs=1, gamma=0.0009, C=1, kernel="rbf", decision_function_shape="ovo")
# {'kernel': 'rbf', 'C': 100, 'decision_function_shape': 'ovo', 'gamma': 0.0001}


# # refit model to full data and save to csv
# start = time. time()
# fit_final_model(gamma=0.0009, kernel="rbf", decision_function_shape="ovo")
# end = time. time()
# print(end - start)

end = time. time()
print(end - start)
