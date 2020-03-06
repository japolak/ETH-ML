#!/usr/bin/env python

import pandas as pd
import numpy as np
import os as os
from sklearn.decomposition import PCA

import math
from scipy import stats
from copy import copy as copy
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import BatchNormalization
from sklearn.semi_supervised import LabelSpreading

from tensorflow import ConfigProto, Session
# Set number of cores to be used
config = ConfigProto(device_count={"CPU": 8})
keras.backend.set_session(Session(config=config))


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()
    
    
#%%
np.random.seed(2019)


#%% Read Data
train_labeled = pd.read_csv('train_labeled.csv')
train_unlabeled = pd.read_csv('train_unlabeled.csv')
test = pd.read_csv('test.csv')

df_trainL = train_labeled
df_trainUL = train_unlabeled
df_test = test

nn = 9000
nnn = 100
df_trainL = df_trainL.iloc[0:nn]
#df_trainUL = df_trainUL.iloc[0:nnn]


XL_unscl=df_trainL.iloc[:,1:df_trainL.shape[1]]
XUL_unscl = df_trainUL
Y1 = df_trainL.iloc[:, 0]

X_unscl = pd.concat([XL_unscl, XUL_unscl, df_test])

scl = StandardScaler()

X = pd.DataFrame(scl.fit_transform(X_unscl))
pca_model = PCA(0.999)
pca_model.fit(X)

X = pd.DataFrame(pca_model.transform(X))


features = X.shape[1]
L = X.iloc[0:nn]
UL = X.iloc[nn:df_trainUL.shape[0] + nn]
df_test = X.iloc[df_trainUL.shape[0] + nn: X.shape[0]]
labels = np.ones((UL.shape[0], 1))
labels = pd.DataFrame(labels*-1)
Y = pd.concat([Y1, labels])

n_folds = 10
i = 0

kfold = model_selection.StratifiedKFold(n_splits=n_folds, random_state=i)
kfold2 = model_selection.StratifiedKFold(n_splits=2, random_state=2020)
np.random.seed(2019)


X_train= L
Y_train= Y1

#X_test = df_test.iloc[:, 0:120]

scl = StandardScaler()
X_train = pd.DataFrame(scl.fit_transform(X_train))
#X_test = pd.DataFrame(scl.fit_transform(X_test))
nclass = 10
nvalid = 0.2

#%%
import time
   
n_folds = 10
acc = np.zeros(n_folds)
elapse = np.zeros(n_folds)

kfold = model_selection.StratifiedKFold(n_splits=n_folds, random_state=2019)

kernel_vec = ['rbf']

C_vec = np.logspace(-1, 2, num = 5)
gamma_vec = np.logspace(-1, 1, num = 3)

score = np.zeros((25, 3))
score_indx = 0
adds = 3
acc_vec = np.zeros((adds,1 ))
L = L
Y = Y1

tauSVC = 0.4
tauNN = 0.4
probCutoff = 0.9

C = 5
gamma = 0.00993642

X_UL = copy(UL)

num_to_add = int(np.ceil(len(X_UL)/adds))

j = 0
klist = list(kfold.split(L,Y))
train_index, test_index = klist[0]
    
X_train, X_test = L.iloc[train_index], L.iloc[test_index]
Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

klist2 = list(kfold2.split(X_train, Y_train))
L1_indx, L2_indx = klist2[0]

allData = True
if allData:
    X_L1, X_L2 = copy(X_train), copy(X_train)
    Y_L1, Y_L2 = copy(Y_train), copy(Y_train)
else:
    X_L1, X_L2 = X_train.iloc[L1_indx], X_train.iloc[L2_indx]
    Y_L1, Y_L2 = Y_train.iloc[L1_indx], Y_train.iloc[L2_indx]
    
n_NN = 5
k = 7
i = 0
X_L = L
Y_L = Y
print('allData = ', allData)
for i in range(3):
    ticks1 = time.time()

    NN = Sequential()
    NN.add(Dense(900, input_dim=features, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
    NN.add(BatchNormalization())
    NN.add(Dropout(k/10))
    NN.add(Dense(500, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
    NN.add(BatchNormalization())
    NN.add(Dropout(k/10))
    NN.add(Dense(200, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
    NN.add(BatchNormalization())
    NN.add(Dropout(k/10))
    NN.add(Dense(100, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
    NN.add(BatchNormalization())
    NN.add(Dropout(k/10))
    NN.add(Dense(10, activation=tf.nn.softmax, kernel_initializer='random_normal', bias_initializer='zeros'))
    NN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    Y_L_mat = to_categorical(Y_L, num_classes=nclass)
    
    NN.fit(X_L.values, Y_L_mat, epochs = 100, verbose = 2)
    pred_probNN = NN.predict(X_UL.values)
    predNN = pred_probNN.argmax(axis = 1)
    predNN_prob = pred_probNN.max(axis = 1)
    
    pred_prob_max = pred_probNN.max(axis = 1)
    
    to_add = np.where(pred_probNN > 0.95)[0]

    X_to_add = X_UL.iloc[to_add]
    Y_to_add = pd.DataFrame(predNN[to_add])
    
    X_L1 = pd.concat([X_L1,X_to_add])
    Y_L1 = pd.concat([Y_L1,Y_to_add])
    
    # Drop from the unlabeled date the obervations the we labeled
    X_UL.drop(X_UL.index[to_add], axis = 0, inplace = True)
    
    numNN = len(to_add)

NN = Sequential()
NN.add(Dense(900, input_dim=features, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
NN.add(BatchNormalization())
NN.add(Dropout(k/10))
NN.add(Dense(500, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
NN.add(BatchNormalization())
NN.add(Dropout(k/10))
NN.add(Dense(200, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
NN.add(BatchNormalization())
NN.add(Dropout(k/10))
NN.add(Dense(100, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
NN.add(BatchNormalization())
NN.add(Dropout(k/10))
NN.add(Dense(10, activation=tf.nn.softmax, kernel_initializer='random_normal', bias_initializer='zeros'))
NN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Y_L1_mat = to_categorical(Y_L1, num_classes=nclass)

NN.fit(X_L1.values, Y_L1_mat, epochs = 100, verbose = 2)
pred_prob = NN.predict(X_test.values)
pred = pred_prob.argmax(axis = 1)


pred = pd.DataFrame(np.stack((df_test.index+30000, pred), axis = 1))
pred.columns = ['Id', 'y']
pred.to_csv('output_AloMichalMateij.csv', index=False)




    # Test Initial Model
#    NN = Sequential()
#    NN.add(Dense(900, input_dim=features, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
#    NN.add(BatchNormalization())
#    NN.add(Dropout(k/10))
#    NN.add(Dense(500, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
#    NN.add(BatchNormalization())
#    NN.add(Dropout(k/10))
#    NN.add(Dense(200, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
#    NN.add(BatchNormalization())
#    NN.add(Dropout(k/10))
#    NN.add(Dense(100, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
#    NN.add(BatchNormalization())
#    NN.add(Dropout(k/10))
#    NN.add(Dense(10, activation=tf.nn.softmax, kernel_initializer='random_normal', bias_initializer='zeros'))
#    NN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#    Y_L1_mat = to_categorical(Y_L1, num_classes=nclass)
#    
#    NN.fit(X_L1.values, Y_L1_mat, epochs = 200, verbose=1)
#    pred_prob = NN.predict(X_test.values)
#    pred = pred_prob.argmax(axis = 1)
#    
#    pred_prob_max = pred_prob.max(axis = 1)
#    
#    to_add = np.where(pred_prob > 0.9)[0]
#
#    X_to_add = X_UL.iloc[to_add]
#    Y_to_add = pd.DataFrame(pred_prob[to_add].argmax(axis = 1))
#    
#    X_train = pd.concat([X_train,X_to_add])
#    Y_train = pd.concat([Y_train,Y_to_add])
#    
#    # Drop from the unlabeled date the obervations the we labeled
#    X_UL.drop(X_UL.index[to_add], axis = 0, inplace = True)
#    
#    
#    
#    accNN = accuracy_score(Y_test, predNN)
#    
#
#    acc_vec[i] = accNN
#    tocks = time.time()
#    print('[', i, ']', accNN, numNN,  'seconds:',  round(tocks - ticks1))
#    if numNN ==0:
#        break 
  


#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################


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
    

