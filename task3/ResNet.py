#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ResNet full pre-activation

print('\n Initializing...')
import numpy as np 
import pandas as pd 
from copy import copy
import keras 
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, add
from keras.utils.vis_utils import plot_model
from keras.constraints import max_norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%% Read Data
print('Reading data...')
# Read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Modify data
train_labels = pd.DataFrame(train['y'])
train = train.drop(['y'], axis=1)



#%%  
        
def initializeData(train=train, train_labels=train_labels, test=test):  
    x_train = train.values 
    x_test = test.values  
    y_train = train_labels.values.ravel()
    # Scale data
    if True :  
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)  
    
    return x_train,y_train,x_test     

def relu(x):
    return Activation('relu')(x) 

def createResNet(ResNetType='OG1',
                 neurons_first=120,
                 neurons_last=5,
                 nodes=512,
                 init='glorot_normal',
                 c_maxnorm=5,
                 activation='relu',
                 dropout=0.5,
                 blocks=2,
                 method='adam',
                 loss='sparse_categorical_crossentropy',
                 accuracy='accuracy',
                 ):
    
    activation = Activation(activation)
    act_idx = 1
    nname = 'ResNet-'+ResNetType+'-'+str(blocks)+'-'+str(nodes)
    print('\n ------------- \n Architecture:',nname,' \n ------------- \n')
    
    # Input layer
    inp = Input(shape=(neurons_first,))
    # First layer
    ly = Dense(nodes, kernel_initializer=init,
               kernel_constraint=max_norm(c_maxnorm, axis=0),
               use_bias=False)(inp)
    
    if ResNetType == 'OG1':
        ly = BatchNormalization()(ly)
        ly = relu(ly)
        ly = Dropout(dropout)(ly)
        # Middle layers
        for i in range(blocks):
            middle = Dense(nodes, kernel_initializer=init, 
                           kernel_constraint=max_norm(c_maxnorm, axis=0),
                           use_bias=False)(ly)
            middle = BatchNormalization()(middle)
            middle = relu(middle)
            middle = Dropout(dropout)(middle)
            ly = add([ly, middle])
            #act = copy(activation); act.name = act.name+'_'+str(act_idx); act_idx += 1
            #ly = act(ly) 
            
    if ResNetType == 'PA1':

        # Middle layers
        for i in range(blocks):
            middle = BatchNormalization()(ly)
            act = copy(activation); act.name = act.name+'_'+str(act_idx); act_idx += 1
            middle = act(middle)
            middle = Dropout(dropout)(middle)
            middle = Dense(nodes, kernel_initializer=init, 
                           kernel_constraint=max_norm(c_maxnorm, axis=0),
                           use_bias=False)(middle)
            ly = add([ly, middle])
            #act = copy(activation); act.name = act.name+'_'+str(act_idx); act_idx += 1
            #ly = act(ly) 
        
    
    
    if ResNetType == 'OG2':
        ly = BatchNormalization()(ly)
        act = copy(activation); act.name = act.name+'_'+str(act_idx); act_idx += 1
        ly = act(ly)
        ly = Dropout(dropout)(ly)
        # Middle layers
        for i in range(blocks):
            middle = Dense(nodes, kernel_initializer=init, 
                           kernel_constraint=max_norm(c_maxnorm, axis=0),
                           use_bias=False)(ly)
            middle = BatchNormalization()(middle)
            act = copy(activation); act.name = act.name+'_'+str(act_idx); act_idx += 1
            middle = act(middle)
            middle = Dropout(dropout)(middle)
            middle = Dense(nodes, kernel_initializer=init, 
                           kernel_constraint=max_norm(c_maxnorm, axis=0),
                           use_bias=False)(middle)
            middle = BatchNormalization()(middle)
            middle = Dropout(dropout)(middle)
            ly = add([ly, middle])
            act = copy(activation); act.name = act.name+'_'+str(act_idx); act_idx += 1
            ly = act(ly)    
            
            
    if ResNetType == 'PA2':
        # Middle layers
        for i in range(blocks):
            middle = BatchNormalization()(ly)
            act = copy(activation); act.name = act.name+'_'+str(act_idx); act_idx += 1
            middle = act(middle)
            middle = Dropout(dropout)(middle)
            middle = Dense(nodes, kernel_initializer=init, 
                           kernel_constraint=max_norm(c_maxnorm, axis=0),
                           use_bias=False)(middle)
            middle = BatchNormalization()(middle)
            act = copy(activation); act.name = act.name+'_'+str(act_idx); act_idx += 1
            middle = act(middle)
            middle = Dropout(dropout)(middle)
            middle = Dense(nodes, kernel_initializer=init, 
                           kernel_constraint=max_norm(c_maxnorm, axis=0),
                           use_bias=False)(middle)
            ly = add([ly, middle])
        ly = BatchNormalization()(ly)
        act = copy(activation); act.name = act.name+'_'+str(act_idx); act_idx += 1
        ly = act(ly)
        ly = Dropout(dropout)(ly)  
    
    # Last layer
    out = Dense(neurons_last, kernel_initializer=init, 
               kernel_constraint=max_norm(c_maxnorm, axis=0),
               use_bias=False)(ly)
    out = BatchNormalization()(out)
    act_idx += 1
    # Output layer
    out = Activation('softmax')(out)
    # Create model
    nn = Model(inputs=inp, outputs=out)
    # Compile
    nn.compile(optimizer=method, loss=loss, metrics=[accuracy])

    # Create model
    nn = Model(inputs=inp, outputs=out)
    # Compile
    nn.compile(optimizer=method, loss=loss, metrics=[accuracy])
    return nn,nname

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):        
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))   

#%% 
nepoch =500
nbatchsize = 32
ntrain = train.shape[0]
ntest = test.shape[0]

# Create Network Architecture
nn,nname = createResNet(ResNetType='OG1',blocks=2,nodes=512)
if True : nn.summary();  plot_model(nn, to_file=nname+'.png', show_shapes=True)    
# Initialize Data
x_train, y_train, x_test = initializeData(train,train_labels,test)
nvalid = 0.3 ;  nseed=0
if nvalid >0:
    x_train, x_valid, y_train, y_valid = train_test_split(
                x_train, y_train, test_size = nvalid, random_state=nseed, stratify = y_train)
    history = LossHistory() 
    print('Training network...')
    nn.fit(x_train, y_train, verbose=2, shuffle=True,
           epochs=nepoch, batch_size=nbatchsize,
           callbacks=[history], validation_data=(x_valid,y_valid))
    print('\n Minimum Loss',round(min(history.val_loss),4),'at epoch',np.array(history.val_loss).argmin()+1)
    print('\n Maximum Accuracy',round(max(history.val_acc),4),'at epoch',np.array(history.val_acc).argmax()+1)
    # Save Results
    if True:
        hist = pd.DataFrame()
        hist[nname+' loss']=np.array(history.val_loss) ; hist[nname+' acc']=np.array(history.val_acc)
        hist.to_csv('./logs/'+nname+'.csv',index=False) 
        print('\n Results Saved!')
    print('\n Done!')
if nvalid ==0:
    nn.fit(x_train, y_train, verbose=2, 
                 epochs=nepoch, batch_size=nbatchsize)
    y_test_pred = nn.predict(x_test).argmax(axis=-1)
    def save_solution(csv_file,solution):
    	with open(csv_file, 'w') as csv:
            df = pd.DataFrame.from_dict({'Id':range(ntrain,ntrain+len(solution)),'y': solution})
            df.to_csv(csv,index = False)
    save_solution('solution.csv', y_test_pred )
        