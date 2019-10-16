#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import keras 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Layer
import tensorflow as tf

# Set number of cores to be used
from tensorflow import ConfigProto, Session
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
    from sklearn.preprocessing import StandardScaler
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
        


class addGamma( Layer ):
    # Gamma parameters
    def __init__(self  , **kwargs):
        super(addGamma, self).__init__(**kwargs)
        
    def build(self, input_shape):
        if self.built:
            return
        self.gamma = self.add_weight(name='gamma', shape= input_shape[1:], initializer='ones', trainable=True)
        self.built = True
        super(addGamma, self).build(input_shape)  
        
    def call(self, x , training=None):
        return tf.add( x , self.gamma )


def createNeuralNetwork(nnType='NN',
                 arch = None,
                 neuronsFirst=nfeatures,
                 neuronsLast=nclasses,
                 nodes = 512,
                 layers = 2,
                 init='glorot_normal',
                 maxNorm=5,
                 activation='relu',
                 dropout=0.5,
                 useBias=True,
                 optimizer='adam',
                 loss='categorical_crossentropy',
                 accuracy='accuracy',
                 ):
    
    from keras.models import Model
    from keras.layers import Dense, Dropout, Activation, BatchNormalization, Input
    from keras.constraints import max_norm
    
    def activationFunction(x):
        return Activation(activation)(x)
    
    def denseLayer(x, nodes=nodes):
        return Dense(nodes, kernel_initializer=init,
               kernel_constraint=max_norm(maxNorm, axis=0),
               use_bias=useBias)(x)
        
    def getArchitecture(n1=neuronsFirst,nl=neuronsLast,l=layers,n=nodes):
        arch = [n1]
        for i in range(l):
            arch.append(n)
        arch.append(nl)
        return arch
      
    # Neural Network name
    if arch is None: arch = getArchitecture() 
    nname = nnType+'-'+(','.join(map(str,arch)))
    print('\n ------------- \n Architecture:',nname,' \n ------------- \n')
    
    # Parameters
    L = len(arch) - 1
    gammas =[ addGamma() for l in range(L)]  
    
    # Input layer
    inp = Input(shape=(neuronsFirst,))
    ly = inp
    
    # Middle layers
    for i in range(1,L+1):
        ly = denseLayer(ly,nodes=arch[i])
        ly = BatchNormalization()(ly)
        if i == L:
            ly = Activation('softmax')(gammas[i-1](ly))
        else:
            ly = activationFunction(gammas[i-1](ly))
            ly = Dropout(dropout)(ly)
    
    # Output layers
    out = Activation('softmax')(ly)
    
    # Connect Model
    nn = Model(inputs=inp, outputs=out)
    
    # Compile Model
    nn.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])
    
    return nn, nname

def createOldNeuralNetwork():
    
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, BatchNormalization
    nnodes = 139*12
    ndropout=0.5
    
    
    model_nn = Sequential()
    model_nn.add(Dense(int(1024), input_dim=139, kernel_initializer='random_uniform', bias_initializer='zeros' ))
    model_nn.add(BatchNormalization())
    model_nn.add(Activation('relu'))
    model_nn.add(Dropout(ndropout))
    model_nn.add(Dense(int(1024), kernel_initializer='random_uniform', bias_initializer='zeros' ))
    model_nn.add(BatchNormalization())
    model_nn.add(Activation('relu'))
    model_nn.add(Dropout(ndropout))
    model_nn.add(Dense(int(512), kernel_initializer='random_uniform', bias_initializer='zeros' ))
    model_nn.add(BatchNormalization())
    model_nn.add(Activation('relu'))
    model_nn.add(Dropout(ndropout))
    model_nn.add(Dense(int(256), kernel_initializer='random_uniform', bias_initializer='zeros' ))
    model_nn.add(BatchNormalization())
    model_nn.add(Activation('relu'))
    model_nn.add(Dropout(ndropout))
    model_nn.add(Dense(int(128), kernel_initializer='random_uniform', bias_initializer='zeros' ))
    model_nn.add(BatchNormalization())
    model_nn.add(Activation('relu'))
    model_nn.add(Dropout(ndropout))
    model_nn.add(Dense(10))
    model_nn.add(Activation('softmax'))
    
    model_nn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    nname = 'old'
    
    return model_nn, nname
#%% Create model
#nn, nname = createNeuralNetwork(arch=[139,1024,512,256,128,10])
nn, nname = createOldNeuralNetwork()
if True : nn.summary(); #keras.utils.plot_model(nn, to_file='./logs/'+nname+'.png', show_shapes=True, show_layer_names=False)  


#%%
    
def schedule(epoch,learning_rate):
    if(epoch==50 or epoch==100):
        learning_rate = learning_rate/2
    return learning_rate

nepoch =150
nbatchsize = 32
nvalidation = 0.5
history = LossHistory() 
lrate = keras.callbacks.LearningRateScheduler(schedule, verbose=0)


x_train, y_train, x_test, x_unlabeled = InitializeData()
x_train,x_valid,y_train,y_valid = train_test_split(x_train,y_train,test_size=nvalidation,random_state=42)

X_T = x_train
Y_T = keras.utils.np_utils.to_categorical(y_train, num_classes=nclasses)
X_U = x_unlabeled


y_train_mat = keras.utils.np_utils.to_categorical(y_train, num_classes=nclasses)
y_valid_mat = keras.utils.np_utils.to_categorical(y_valid, num_classes=nclasses)

nn.fit(x_train, y_train_mat, verbose = 2, shuffle = True ,epochs = nepoch
       , batch_size = nbatchsize,
       callbacks=[history], validation_data=(x_valid,y_valid_mat) )
#%%
## Save Results
#if nvalidation > 0:
#    print('\n Minimum Loss',round(min(history.val_loss),4),'at epoch',np.array(history.val_loss).argmin()+1)
#    print('\n Maximum Accuracy',round(max(history.val_acc),4),'at epoch',np.array(history.val_acc).argmax()+1)
#    hist = pd.DataFrame()
#    hist[nname+' loss']=np.array(history.val_loss) ; hist[nname+' acc']=np.array(history.val_acc)
#    hist.to_csv('./logs/'+nname+'.csv',index=False) 
#    print('\n Logs Saved!')

treshold_list = [0.99,0.98,0.96,0.90,0]
for i in range(5):
    
    treshold = treshold_list[i]
    print('\n Treshold:',treshold,'\n')
    X_U_prob = nn.predict(X_U).max(axis=1)
    X_U_pred = nn.predict(X_U).argmax(axis=-1)
    print('\n Max probaility:',X_U_prob.max(axis=-1),'\n')
    index = X_U_prob > treshold
    print('\n Taking:',index.sum(),'/',len(X_U_prob),'\n')
    to_drop = np.where(index)[0]
    X_toadd = X_U[index]
    Y_toadd = keras.utils.np_utils.to_categorical(X_U_pred[index],num_classes=nclasses)
    X_previous = X_T
    Y_previous = Y_T
    X_T = np.concatenate([X_previous,X_toadd],axis=0)
    Y_T = np.concatenate([Y_previous,Y_toadd],axis=0)
    df_X_U = pd.DataFrame(X_U)
    df_X_U.drop(df_X_U.index[to_drop], axis=0,inplace=True)
    X_U = df_X_U
    nn.fit(X_T,Y_T,verbose = 2, shuffle = True ,epochs = 50, batch_size = nbatchsize,
           callbacks=[history], validation_data=(x_valid,y_valid_mat))

X_V = x_valid
Y_V = keras.utils.np_utils.to_categorical(y_valid,num_classes=nclasses)

X_T = np.concatenate([X_T,X_V],axis=0)
Y_T = np.concatenate([Y_T,Y_V],axis=0)    
nn.fit(X_T,Y_T,verbose = 2, shuffle = True ,epochs = 200, batch_size = nbatchsize, callbacks =[lrate])
y_test_pred = nn.predict(x_test).argmax(axis=-1)
save_solution('solution'+nname+'.csv', y_test_pred )
print('\n Solution Saved!')

#if nvalidation == 0:
#    y_test_pred = nn.predict(x_test).argmax(axis=-1)
#    save_solution('solution'+nname+'.csv', y_test_pred )
#    print('\n Solution Saved!')
    

