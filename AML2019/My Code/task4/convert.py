import os
import pandas as pd 
from utils import *

path = makePath()
name = 'test_emg_features'
X = importPickle(name)
print(X.shape)
X = removeZeros(X)
X = missingValues(X, opt='median')
print(X.shape)
missingValues(X)
saveData(X,name)
