# -*- coding: utf-8 -*-
import pandas as pd 

# Read Data
train_labeled = pd.read_hdf('train_labeled.h5', 'train')
train_unlabeled = pd.read_hdf('train_unlabeled.h5', 'train')
test = pd.read_hdf('test.h5', 'test')
# Save Data
train_labeled.to_csv('train_labeled.csv',index=False)
train_unlabeled.to_csv('train_unlabeled.csv',index=False)
test.to_csv('test.csv',index=False)
