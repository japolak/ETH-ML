#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ResNet full pre-activation

print('\n Initializing...')
import numpy as np 
import pandas as pd 
import keras 
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, add
from keras.utils.vis_utils import plot_model
from keras.constraints import max_norm
from sklearn.preprocessing import StandardScaler
from tensorflow import ConfigProto, Session
# Set number of cores to be used
config = ConfigProto(device_count={"CPU": 12})
keras.backend.set_session(Session(config=config))

#%% Read Data
train_labeled = pd.read_csv('train_labeled.csv')
train_unlabeled = pd.read_csv('train_unlabeled.csv')
test = pd.read_csv('test.csv')

train_labels = pd.DataFrame(train_labeled['y'])
train_labeled = train_labeled.drop(['y'], axis=1)

# Data Information
ntrain = train_labeled.shape[0]
ntrain_unlabeled = train_unlabeled.shape[0]
ntest = test.shape[0]
nfeatures = train_labeled.shape[1]
nclasses = len(np.unique(train_labels['y']))
nweights = ntrain / (nclasses * np.bincount(train_labels['y']))

#%%  
        
def InitializeData(train=train_labeled, train_labels=train_labels, test=test):  
    x_train = train.values 
    x_test = test.values  
    y_train = train_labels.values.ravel()
    # Scale data
    if True :  
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)  
    return x_train,y_train,x_test      

def save_solution(csv_file,solution):
    with open(csv_file, 'w') as csv:
        df = pd.DataFrame.from_dict({'Id':range(ntrain+ntrain_unlabeled,ntrain+ntrain_unlabeled+len(solution)),'y': solution})
        df.to_csv(csv,index = False)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):        
        self.val_loss = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc')) 
    
def CreateNeuralNet(nnType='OG1',
                 neuronsFirst=nfeatures,
                 neuronsLast=nclasses,
                 nodes=512,
                 init='glorot_normal',
                 maxNorm=5,
                 activation='relu',
                 dropout=0.5,
                 blocks=2,
                 useBias=False,
                 optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 accuracy='accuracy',
                 ):
    
    def ActivationRelu(x):
        return Activation(activation)(x)
    
    def DenseLayer(x, nodes=nodes):
        return Dense(nodes, kernel_initializer=init,
               kernel_constraint=max_norm(maxNorm, axis=0),
               use_bias=useBias)(x)


    nname = nnType+'-'+str(blocks)+'-'+str(nodes)
    print('\n ------------- \n Architecture:',nname,' \n ------------- \n')
    
    # Input layer
    inp = Input(shape=(neuronsFirst,))
'''
    # First layer
    ly = Dense(nodes, kernel_initializer=init,
               kernel_constraint=max_norm(maxNorm, axis=0),
               use_bias=useBias)(inp)
'''
    if nnType == 'LadderNet':
        h = inp
        W = DenseLayer(h)
        z = BatchNormalization()(W)
        h = ActivationRelu(z)
        W = DenseLayer(h)
        z = BatchNormalization()(W)
        h = Activation('softmax')(z)
        out = h
    
    # Middle layers
    if nnType == 'Seq': #Sequential model
        ly = BatchNormalization()(ly)
        ly = ActivationRelu(ly)
        ly = Dropout(dropout)(ly)
        for i in range(blocks):
            mid = DenseLayer(ly)
            mid = BatchNormalization()(mid)
            mid = ActivationRelu(mid)
            ly = Dropout(dropout)(mid)
                  
    if nnType == 'SeqCone':
        ly = BatchNormalization()(ly)
        ly = ActivationRelu(ly)
        ly = Dropout(dropout)(ly)
        for i in range(blocks):
            mid = DenseLayer(ly,int(nodes/(2**(i+1))))
            mid = BatchNormalization()(mid)
            mid = ActivationRelu(mid)
            ly = Dropout(dropout)(mid)
            
    if nnType == 'ResNet':
        ly = BatchNormalization()(ly)
        ly = ActivationRelu(ly)
        ly = Dropout(dropout)(ly)
        for i in range(blocks):
            middle = DenseLayer(ly)
            middle = BatchNormalization()(middle)
            middle = ActivationRelu(middle)
            middle = Dropout(dropout)(middle)
            ly = add([ly, middle])  
            ly = ActivationRelu(ly)
        
    if nnType == 'ResNet-OG':
        ly = BatchNormalization()(ly)
        ly = ActivationRelu(ly)
        ly = Dropout(dropout)(ly)
        for i in range(blocks):
            middle = DenseLayer(ly)
            middle = BatchNormalization()(middle)
            middle = ActivationRelu(middle)
            middle = Dropout(dropout)(middle)
            ly = add([ly, middle])

    if nnType == 'ResNet-PA':
        for i in range(blocks):
            middle = BatchNormalization()(ly)
            middle = ActivationRelu(middle)
            middle = Dropout(dropout)(middle)
            middle = DenseLayer(middle)
            ly = add([ly, middle])
        
    if nnType == 'ResNet-OGx2':
        ly = BatchNormalization()(ly)
        ly = ActivationRelu(ly)
        ly = Dropout(dropout)(ly)
        for i in range(blocks):
            middle = DenseLayer(ly)
            middle = BatchNormalization()(middle)
            middle = ActivationRelu(middle)
            middle = Dropout(dropout)(middle)
            middle = DenseLayer(middle)
            middle = BatchNormalization()(middle)
            middle = Dropout(dropout)(middle)
            ly = add([ly, middle])
            ly = ActivationRelu(ly)    
            
    if nnType == 'ResNet-PAx2':
        for i in range(blocks):
            middle = BatchNormalization()(ly)
            middle = ActivationRelu(middle)
            middle = Dropout(dropout)(middle)
            middle = DenseLayer(middle)
            middle = BatchNormalization()(middle)
            middle = ActivationRelu(middle)
            middle = Dropout(dropout)(middle)
            middle = DenseLayer(middle)
            ly = add([ly, middle])
        ly = BatchNormalization()(ly)
        ly = ActivationRelu(ly)
        ly = Dropout(dropout)(ly)  
    
    # Last layer
    ly = Dense(neuronsLast, kernel_initializer=init, 
               kernel_constraint=max_norm(maxNorm, axis=0),
               use_bias=useBias)(ly)
    ly = BatchNormalization()(ly)
    # Output layer
    out = Activation('softmax')(ly)
    # Create model
    nn = Model(inputs=inp, outputs=out)
    # Compile
    nn.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
    return nn,nname  

#%% Create Network Architecture

nn,nname = CreateNeuralNet(nnType='SeqCone',blocks=3,nodes=1024, useBias=True)
if False : nn.summary();  plot_model(nn, to_file='./logs/'+nname+'.png', show_shapes=True, show_layer_names=False)  

#%%
nepoch =200
nbatchsize = 32
nvalidation = 0
history = LossHistory() 

x_train, y_train, x_test = InitializeData(train_labeled,train_labels,test)
nn.fit(x_train, y_train, verbose = 2, shuffle = True,
       epochs = nepoch, batch_size = nbatchsize,
       callbacks=[history], validation_split=nvalidation )

# Save Results
if nvalidation > 0:
    print('\n Minimum Loss',round(min(history.val_loss),4),'at epoch',np.array(history.val_loss).argmin()+1)
    print('\n Maximum Accuracy',round(max(history.val_acc),4),'at epoch',np.array(history.val_acc).argmax()+1)
    hist = pd.DataFrame()
    hist[nname+' loss']=np.array(history.val_loss) ; hist[nname+' acc']=np.array(history.val_acc)
    hist.to_csv('./logs/'+nname+'.csv',index=False) 
    print('\n Logs Saved!')

if nvalidation == 0:
    y_test_pred = nn.predict(x_test).argmax(axis=-1)
    save_solution('solution'+nname+'.csv', y_test_pred )
    print('\n Solution Saved!')
    