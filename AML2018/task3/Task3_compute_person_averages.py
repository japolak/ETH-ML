import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score
import os
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
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
import biosppy.signals.ecg as ecg
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

WORK_FOLDER = '/home/leslie/Desktop/AML/Task3/aml/'

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
    # scaler = StandardScaler()  
    # scaler.fit(x_train_data)  
    # x_train_data = scaler.transform(x_train_data)
    # x_test_data = scaler.transform(x_test_data)

    # convert to pandas dataframe
    x_train_data = pd.DataFrame(data=x_train_data)
    x_test_data = pd.DataFrame(data=x_test_data)
    y_train_data = pd.DataFrame(data=y_train_data)
    
    # return the data in a tuple 
    return(x_train_data, y_train_data, x_test_data, id_data)



x_train_data, y_train_data, x_test_data, id_data = prepare_data()
# create a list of the patient data
row_data = []
# y_list = []
for row in range(5117):	
	row_data.append(list(((np.array(x_train_data.iloc[row,:]))[~np.isnan(np.array(x_train_data.iloc[row,:]))])))
	# y_list.append(list(np.array(y_train_data.iloc[row,1])))

test_data = []
for test_row in range(3411):
	test_data.append(list(((np.array(x_test_data.iloc[test_row,:]))[~np.isnan(np.array(x_test_data.iloc[test_row,:]))])))
# print(y_list)

# print(row_data[1])


# this extracts the heart beats into e. ndarray (nrows, 180)
# now need to take average for each point

def rr(rr_list):
	num_peaks = len(rr_list)
	distances = []
	for i in range(num_peaks-1):
		distances.append(rr_list[i+1] - rr_list[i])
	avg = np.mean(np.array(distances))
	std = np.std(np.array(distances))
	skew = stats.skew(np.array(distances))

	return(avg, std, skew)


def average_heartbeat_per_person(patient_list):

    # function that extracts the heartbeat templates from the data, then returns an average template for each person
    # returns each patient average, each  patient is a row with     the 180 data points as columns
# 
    average_heartbeat_matrix = np.array([1] * (180*5 + 10))
    # all_heartbeats = np.array([1] * (180*7))
    all_heartbeats_class = []
    patient_number = []
    no_7 = []

    for i in range(len(patient_list)):
    # for i in range(10):
        ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal=np.array(patient_list[i]), sampling_rate = 300, show=False)   
        # rpeak
        # s0 = ecg.hamilton_segmenter(signal = np.array(patient_list[i]), sampling_rate=300)[0]
        # rpeaks0 = ecg.gamboa_segmenter(signal = np.array(patient_list[i]), sampling_rate=300)[0]
        # new_templates, rpeaks = ecg.extract_heartbeats(signal = np.array(patient_list[i]), sampling_rate=300, rpeaks=rpeaks0)
        # person_7_heartbeats_con = []
        
        features = list(np.quantile(templates, 0.1, axis=0)) + list(np.quantile(templates, 0.25, axis=0)) + list(np.quantile(templates, 0.5, axis=0)) + list(np.mean(templates, axis=0)) + list(np.quantile(templates, 0.75, axis=0)) 
        new = (np.mean(templates[:,60]))
        std = np.std(templates[:,60])
        skew = stats.skew(templates[:,60])
        rr_avg, rr_std, rr_skew = rr(list(rpeaks))
        
        patient_avg = np.mean(np.array(patient_list[i]))
        patient_med = np.quantile(np.array(patient_list[i]), 0.5)
        patient_std = np.std(np.array(patient_list[i]))
        patient_skew = stats.skew(np.array(patient_list[i]))

        features.append(new)
        features.append(std)
        features.append(skew)
        features.append(rr_avg)
        features.append(rr_std)
        features.append(rr_skew)
        features.append(patient_avg)
        features.append(patient_med)
        features.append(patient_std)
        features.append(patient_skew)



        
        
        person_average = features
        
        # if len(templates) > 6:
        
        #     person_7_heartbeats = np.asarray(templates)[0:7,:]
    
        #     for row in range(7):
        # 	    person_7_heartbeats_con += list(person_7_heartbeats[row,:])

        #     all_heartbeats = np.vstack((all_heartbeats, np.asarray(person_7_heartbeats_con)))
        # else:
        # 	no_7.append(i)

        # to get quantiles
        # person_average = list(np.quantile(templates, 0.1, axis=0)) + list(np.quantile(templates, 0.25, axis=0)) + list(np.mean(templates, axis=0)) + list(np.quantile(templates, 0.75, axis=0)) + list(np.quantile(templates, 0.9, axis=0))
        # print(len(person_average))
        average_heartbeat_matrix = np.vstack((average_heartbeat_matrix, np.asarray(person_average)))
        
        print(i)
    

    print(average_heartbeat_matrix.shape)
    
    average_heartbeat_matrix = average_heartbeat_matrix[1:,:]
    # all_heartbeats = all_heartbeats[1:,:]
    return(average_heartbeat_matrix)


all_heartbeats = average_heartbeat_per_person(row_data)
test_heartbeats = average_heartbeat_per_person(test_data)

# row_len = pd.DataFrame(data=person_averages)
all_heartbeat_samples = pd.DataFrame(data=all_heartbeats)
test_heartbeat_samples = pd.DataFrame(data=test_heartbeats)
# all_heartbeats_class = pd.DataFrame(data=all_heartbeats_class)

## save results to csv
# row_len.to_csv(WORK_FOLDER + "ecg_filtered_rowlen.csv", index=False)
all_heartbeat_samples.to_csv(WORK_FOLDER + "1255759_10train_quantile_heartbeat_samples.csv", index=False)
test_heartbeat_samples.to_csv(WORK_FOLDER + "1255759_10test_quantile_heartbeat_samples.csv", index=False)
# all_heartbeats_class.to_csv(WORK_FOLDER + "7_heart_beat_class.csv", index=False)
