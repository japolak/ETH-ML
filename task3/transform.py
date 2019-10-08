# -*- coding: utf-8 -*-
import pandas as pd 

# Read Data
train = pd.read_hdf('train.h5', 'train')
test = pd.read_hdf('test.h5', 'test')

# Save Data
train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)
