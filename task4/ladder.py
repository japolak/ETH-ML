# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

from keras.datasets import mnist
import keras
import random
from sklearn.metrics import accuracy_score
from keras.utils.vis_utils import plot_model

from laddernetwork import get_ladder_network_fc
#%%


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



x_train, y_train, x_test, x_unlabeled = InitializeData()
nvalidation = 0.5

x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size = nvalidation, random_state=0, stratify = y_train)

y_train_mat = to_categorical(y_train, num_classes=nclasses)
y_valid_mat = to_categorical(y_valid, num_classes=nclasses) 

#%%


# get the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.0
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.0

y_train = keras.utils.to_categorical( y_train )
y_test = keras.utils.to_categorical( y_test )


# only select 100 training samples 
idxs_annot = list(range( x_train.shape[0]))
random.seed(0)
random.shuffle( idxs_annot )
idxs_annot = idxs_annot[ :100 ]

x_train_unlabeled = x_train
x_train_labeled = x_train[ idxs_annot ]
y_train_labeled = y_train[ idxs_annot  ]


n_rep = int(x_train_unlabeled.shape[0] / x_train_labeled.shape[0])
x_train_labeled_rep = np.concatenate([x_train_labeled]*n_rep)
y_train_labeled_rep = np.concatenate([y_train_labeled]*n_rep)


# initialize the model 
inp_size = 28*28 # size of mnist dataset 
n_classes = 10
model = get_ladder_network_fc( layer_sizes = [ inp_size , 250, 250, n_classes ]  )

if True:
    model.summary()
    plot_model(model, to_file='arch.png', show_shapes=True, show_layer_names=True)  

# train the model for 100 epochs
for _ in range(1):
    model.fit([ x_train_labeled_rep , x_train_unlabeled   ] , y_train_labeled_rep , epochs=1)
    y_test_pr = model.test_model.predict(x_test , batch_size=100 )
    print("test accuracy" , accuracy_score(y_test.argmax(-1) , y_test_pr.argmax(-1)  ))

