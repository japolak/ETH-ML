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


#%% Preprocess Data

# Read data
train = pd.read_hdf("train.h5", "train")
test = pd.read_hdf("test.h5", "test")

print('Running...')

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
nvalid = 0.3

# Some useful parameters 
ntrain = train.shape[0]
ntest = test.shape[0]
nfeatures = train.shape[1]
nclass = len(np.unique(train_labels['y']))
nseed = 0 
nfolds = 5
skf = StratifiedKFold(n_splits= nfolds, shuffle=True, random_state=nseed)
    
# Neural Net Architecture parameters
''' 
SET PARAMETERS HERE
'''
ncompress = 2
nspan = 8
nnodes = nfeatures*nspan
nlayers = 4 
ndrop = .5
nreg = .0001
nbatchsize = 32
nepoch = 200


nname = 'NN9'   
if nlayers == 4:
    a = ','+str(int(nnodes/ncompress**3))
else: 
    a = ''

print('Model Architecture:')
print(nname,
      ' (',int(nnodes/ncompress**0),',',int(nnodes/ncompress**1),',',int(nnodes/ncompress**2),a,')',
      ' comp ', ncompress,
      ' span ', nspan,
      ' layers ', nlayers,
      ' dropout ', ndrop,
      ' regul ', nreg,
      ' batch ', nbatchsize,
      ' epoch ', nepoch,
      sep='')
del a

'''
COPY MODEL SPECIFICATION
NN1 (960,480,240,120) comp 2 span 8 layers 4 dropout 0.5 batch 128 epoch 200 (X overfit)
NN2 (960,480,240,120) comp 2 span 8 layers 4 dropout 0.5 batch 32 epoch 200 
NN3 (960,480,240,120) comp 2 span 8 layers 4 dropout 0.5 batch 16 epoch 200 (X needs more epochs)
NN4 (960,480,240,120) comp 2 span 8 layers 4 dropout 0.5 regularizer 0.01 batch 32 epoch 200
NN5 (960,320,106) comp 3 span 8 layers 3 dropout 0.5 batch 16 epoch 200 (X needs more epochs)
NN6 (960,320,106) comp 3 span 8 layers 3 dropout 0.5 batch 32 epoch 200
NN7 (960,320,106) comp 3 span 8 layers 3 dropout 0.25 batch 32 epoch 200 (X overfit)
NN8 (1800,600,200,66) comp 3 span 15 layers 4 dropout 0.5 batch 32 epoch 200
'''

model_nn = Sequential()
model_nn.add(Dense(int(nnodes/ncompress**0), input_dim=nfeatures, 
                   kernel_initializer='random_uniform', bias_initializer='zeros'))
model_nn.add(BatchNormalization())
model_nn.add(Activation('relu'))
model_nn.add(Dropout(ndrop))
model_nn.add(Dense(int(nnodes/ncompress**1),
                   kernel_initializer='random_uniform', bias_initializer='zeros'))
model_nn.add(BatchNormalization())
model_nn.add(Activation('relu'))
model_nn.add(Dropout(ndrop))
model_nn.add(Dense(int(nnodes/ncompress**2), 
                   kernel_initializer='random_uniform', bias_initializer='zeros'))
model_nn.add(BatchNormalization())
model_nn.add(Activation('relu'))
model_nn.add(Dropout(ndrop))
if nlayers == 4 :
    model_nn.add(Dense(int(nnodes/ncompress**3), 
                   kernel_initializer='random_uniform', bias_initializer='zeros'))
    model_nn.add(BatchNormalization())
    model_nn.add(Activation('relu'))
    model_nn.add(Dropout(ndrop))
model_nn.add(Dense(nclass))
model_nn.add(Activation('softmax'))

model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  
#%% Fit model 

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
    
    model_nn.fit(x_train, y_train_mat, verbose=2, 
                 epochs=nepoch, batch_size=nbatchsize)
    
    y_test_pred = model_nn.predict(x_test).argmax(axis=-1)
    
    def save_solution(csv_file,solution):
    	with open(csv_file, 'w') as csv:
            df = pd.DataFrame.from_dict({'Id':range(ntrain,ntrain+len(solution)),'y': solution})
            df.to_csv(csv,index = False)
    
    save_solution('solution.csv', y_test_pred )

# Save val score result
hist = pd.DataFrame()
hist[nname+' loss']=np.array(history.val_loss)
hist[nname+' acc']=np.array(history.val_acc)
hist.to_csv('./logs/'+nname+'.csv',index=False)

#%%
hist_loss = pd.DataFrame(np.linspace(0,nepoch,nepoch),columns=['epoch'])
nfrom = 50
nmodels = 8
nlistmodels = [2,6,8]
for i in nlistmodels: # range(nmodels):
    nreadname = 'NN'+str(i)
    hist_loss[nreadname]= pd.read_csv('logs/'+nreadname+'.csv',usecols=[0])
    plt.plot(np.linspace(nfrom,nepoch,nepoch-nfrom),hist_loss[nreadname][nfrom:])
    plt.legend()
plt.show()
