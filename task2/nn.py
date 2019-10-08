# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a main script file for Intro to Machine Learning 2019.
@title: task2
@author: japolak
"""


import numpy as np 
import pandas as pd 

import keras # Imports keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, advanced_activations
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold, train_test_split
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
train = train.drop(['x1','x2','x4','x5','x6','x8','x16','x17','x20','x19','x18'],axis =1)
test = test.drop(['x1','x2','x4','x5','x6','x8','x16','x17','x20','x19','x18'],axis =1)

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
nvalid = 0 # 0 means no validation set (ratio!)
x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=nvalid,random_state=1)
if nvalid ==0 : del x_valid, y_valid

# Some useful parameters 
ntrain = train.shape[0]
ntest = test.shape[0]
nclasses = len(np.unique(train_labels['y']))
nseed = 0 
nfolds = 5
skf = StratifiedKFold(n_splits= nfolds, shuffle=True, random_state=nseed)
# nweights = ntrain / (nclasses * np.bincount(train_labels['y']))
#nclasscounts = np.bincount(train_labels['y'])
#nclassweights = ntrain / (nclass * np.bincount(train_labels['y'])) 


#%% Cross Validation training

acc_scores=np.zeros(nfolds,)

for i, (train_index, test_index) in enumerate(skf.split(x_train,y_train)):
    fold_x_train, fold_x_test = x_train[train_index], x_train[test_index]
    fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]
    
    fold_y_train_mat = to_categorical(fold_y_train, num_classes=3)
    fold_y_test_mat = to_categorical(fold_y_test, num_classes=3)
        
    model_nn = Sequential()
    model_nn.add(Dense(50, input_dim=17, kernel_initializer='random_uniform', bias_initializer='zeros' ))
    model_nn.add(BatchNormalization())
    model_nn.add(keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
    model_nn.add(Dropout(0.5))
    model_nn.add(Dense(50, kernel_initializer='random_uniform', bias_initializer='zeros' ))
    model_nn.add(BatchNormalization())
    model_nn.add(keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
    model_nn.add(Dropout(0.5))
    model_nn.add(Dense(3,))
    model_nn.add(keras.layers.Softmax(axis=-1))
    
    model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model_nn.fit(fold_x_train, fold_y_train_mat, epochs=400, batch_size=200, verbose=2
                 ,validation_data=(fold_x_test,fold_y_test_mat))
    
    acc_scores[i]= model_nn.evaluate(fold_x_test, fold_y_test_mat, batch_size=100, verbose=0)[1]
    print(acc_scores[i])

acc = round(np.mean(acc_scores),4)



#%% Validation set training

y_train_mat = to_categorical(y_train, num_classes=3)
if nvalid>0 : y_valid_mat = to_categorical(y_valid, num_classes=3) 


model_nn = Sequential()
model_nn.add(Dense(50, input_dim=17, kernel_initializer='random_uniform', bias_initializer='zeros' ))
model_nn.add(BatchNormalization())
model_nn.add(keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
model_nn.add(Dropout(0.5))
model_nn.add(Dense(50, kernel_initializer='random_uniform', bias_initializer='zeros' ))
model_nn.add(BatchNormalization())
model_nn.add(keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
model_nn.add(Dropout(0.5))
model_nn.add(Dense(3,))
model_nn.add(keras.layers.Softmax(axis=-1))

model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_nn.fit(x_train, y_train_mat, epochs=400, batch_size=200, verbose=2 
             ,validation_data=(x_valid,y_valid_mat)
             )
acc_valid = model_nn.evaluate(x_valid, y_valid_mat, batch_size=200, verbose=1)[1]


if nvalid>0:
    y_valid_pred = model_nn.predict(x_valid).argmax(axis=-1)
    acc = accuracy_score(y_valid, y_valid_pred)

#%% Final Model Training
y_train_mat = to_categorical(y_train, num_classes=3)

model_nn = Sequential()
model_nn.add(Dense(100, input_dim=9, kernel_initializer='random_uniform', bias_initializer='zeros' ))
model_nn.add(BatchNormalization())
model_nn.add(keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
model_nn.add(Dropout(0.5))
model_nn.add(Dense(100, kernel_initializer='random_uniform', bias_initializer='zeros' ))
model_nn.add(BatchNormalization())
model_nn.add(keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0))
model_nn.add(Dropout(0.5))
model_nn.add(Dense(3,))
model_nn.add(keras.layers.Softmax(axis=-1))

model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_nn.fit(x_train, y_train_mat, epochs=500, batch_size=200, verbose=2 )

y_test_pred = model_nn.predict(x_test).argmax(axis=-1)

def save_solution(csv_file,solution):
	with open(csv_file, 'w') as csv:
            df = pd.DataFrame.from_dict({'Id':range(2000,2000+len(solution)),'y': solution})
            df.to_csv(csv,index = False)

save_solution('solution.csv', y_test_pred )
