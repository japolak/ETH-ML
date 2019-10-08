# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a main script file for Intro to Machine Learning 2019.
@title: task3
@author: japolak
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from time import time

import keras # Imports keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#%%
from tensorflow import ConfigProto, Session

# Set number of cores to be used
config = ConfigProto(device_count={"CPU": 8})
keras.backend.set_session(Session(config=config))


#%% Preprocess Data

print('Running...')
# Read data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Modify data
train_labels = pd.DataFrame(train['y'])
train = train.drop(['y'], axis=1)

# Load data
x_train = train.values 
x_test = test.values  
y_train = train_labels.values.ravel()

# Utils functions
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):        
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))        

#%% Validation Set Performance
        
# Initializing Data    
x_train = train.values 
x_test = test.values  
y_train = train_labels.values.ravel()

# Scaling the data
if True :  
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)   

# Create validation set
nvalid = 0

# Some useful parameters 
ntrain = train.shape[0]
ntest = test.shape[0]
nfeatures = train.shape[1]
nclass = len(np.unique(train_labels['y']))
nseed = 0 
nfolds = 5
skf = StratifiedKFold(n_splits= nfolds, shuffle=True, random_state=nseed)
    
# Neural Net Architecture
nepoch = 200
nbatchsize = 32
nnodes = nfeatures*8
ndropout=0.5


model_nn = Sequential()
model_nn.add(Dense(int(nnodes), input_dim=nfeatures, kernel_initializer='random_uniform', bias_initializer='zeros' ))
model_nn.add(BatchNormalization())
model_nn.add(Activation('relu'))
model_nn.add(Dropout(ndropout))
model_nn.add(Dense(int(nnodes/2), kernel_initializer='random_uniform', bias_initializer='zeros' ))
model_nn.add(BatchNormalization())
model_nn.add(Activation('relu'))
model_nn.add(Dropout(ndropout))
model_nn.add(Dense(int(nnodes/4), kernel_initializer='random_uniform', bias_initializer='zeros' ))
model_nn.add(BatchNormalization())
model_nn.add(Activation('relu'))
model_nn.add(Dropout(ndropout))
model_nn.add(Dense(int(nnodes/8), kernel_initializer='random_uniform', bias_initializer='zeros' ))
model_nn.add(BatchNormalization())
model_nn.add(Activation('relu'))
model_nn.add(Dropout(ndropout))
model_nn.add(Dense(nclass))
model_nn.add(Activation('softmax'))

model_nn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


if nvalid>0:
    x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size = nvalid, random_state=nseed, stratify = y_train)
    y_train_mat = to_categorical(y_train, num_classes=nclass)
    y_valid_mat = to_categorical(y_valid, num_classes=nclass) 

    history = LossHistory() 
    tb= TensorBoard(log_dir='./logs'.format(time()))
    
    model_nn.fit(
                 x_train, y_train_mat, verbose=2, shuffle=True,
                 epochs=nepoch, batch_size=nbatchsize,
                 callbacks=[history,tb]
                 ,validation_data=(x_valid,y_valid_mat)
                 )
    
    y_valid_pred = model_nn.predict(x_valid).argmax(axis=-1)
    acc = accuracy_score(y_valid, y_valid_pred) 
    print('\n Validation Accuracy:',acc)

    nfrom = 0
    plt.plot(np.linspace(nfrom,nepoch,nepoch-nfrom),history.val_loss[nfrom:])
    plt.show()
    print('\n Minimum Loss',round(min(history.val_loss),4),'at epoch',np.array(history.val_loss).argmin())

    plt.plot(np.linspace(nfrom,nepoch,nepoch-nfrom),history.val_acc[nfrom:])
    plt.show()
    print('\n Maximum Accuracy',round(max(history.val_acc),4),'at epoch',np.array(history.val_acc).argmax())

if nvalid==0:
    y_train_mat = to_categorical(y_train, num_classes=nclass)
    
    model_nn.fit(x_train, y_train, verbose=2, 
                 epochs=nepoch, batch_size=nbatchsize)
    
    y_test_pred = model_nn.predict(x_test).argmax(axis=-1)
    
    def save_solution(csv_file,solution):
    	with open(csv_file, 'w') as csv:
            df = pd.DataFrame.from_dict({'Id':range(ntrain,ntrain+len(solution)),'y': solution})
            df.to_csv(csv,index = False)
    
    save_solution('solution.csv', y_test_pred )
    