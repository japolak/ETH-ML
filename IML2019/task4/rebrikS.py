# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ResNet full pre-activation

print('\n Initializing...')
import numpy as np 
import pandas as pd 
import keras 
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input, add, GaussianNoise, concatenate
from keras.utils.vis_utils import plot_model
from keras.constraints import max_norm
from sklearn.preprocessing import StandardScaler
from tensorflow import ConfigProto, Session
from keras.utils.np_utils import to_categorical
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
def InitializeData(train=train_labeled, train_labels=train_labels, test=test, unlabeled = train_unlabeled):  
    x_train = train.values 
    x_test = test.values  
    y_train = train_labels.values.ravel()
    x_unlabeled = unlabeled.values
    # Scale data
    if True :  
        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)  
        x_unlabeled = scaler.fit_transform(x_unlabeled)
    return x_train,y_train,x_test,x_unlabeled

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
                 loss='categorical_crossentropy',
                 accuracy='accuracy',
                 stddev=0.3,
                 addNoise=True
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
    inp_unlabeled = Input(shape=(neuronsFirst,))
    inp_labeled = Input(shape=(neuronsFirst,))

    
    h0 = inp_labeled
    n0 = GaussianNoise(stddev)(inp_unlabeled)
    hh0= n0
    
    merge0 = concatenate([h0,hh0])
    zpre1 = DenseLayer(merge0)
    z1 = BatchNormalization()(zpre1)
    n1 = GaussianNoise(stddev)(z1)
    h1 = ActivationRelu(n1)
    
    zpre2 = DenseLayer(h1,nodes=neuronsLast)
    z2 = BatchNormalization()(zpre2)
    n2 = GaussianNoise(stddev)(z2)
    out_corrupted = Activation('softmax')(n2)
        
   
    # Create model
    nn = Model(inputs=[inp_labeled,inp_unlabeled], outputs=out_corrupted)
    # Compile
    nn.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
    return nn,nname  

#%% Create Network Architecture

nn,nname = CreateNeuralNet(nnType='LadderNet',blocks=2,nodes=512, useBias=False)
if True : nn.summary();  plot_model(nn, to_file='./logs/'+nname+'.png', show_shapes=True, show_layer_names=False)  

#%%
nepoch =3
nbatchsize = 32
nvalidation = 0.3
history = LossHistory() 

x_train, y_train, x_test, x_unlabeled = InitializeData()
y_train_mat = to_categorical(y_train, num_classes=nclasses)

x_unlabeled=x_unlabeled[:ntrain]

nn.fit(x_train, y_train_mat, verbose = 2, shuffle = True,
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
    