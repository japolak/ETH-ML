#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import sklearn
import statsmodels
import numbers
import scipy
import warnings
#import biosppy
from utils import *
warnings.filterwarnings('ignore')



#%% Mice splitter
def miceSplitter(data,testMice=3):
    mice1 = np.array(range(0, 21600))
    mice2 = np.array(range(21600, 43200))
    mice3 = np.array(range(43200, 64800))

    if testMice == 1:
        data1 = data.iloc[mice1,:]
        data2 = data.drop(mice1,axis=0)
    elif testMice == 2:
        data1 = data.iloc[mice2, :]
        data2 = data.drop(mice2, axis=0)
    elif testMice == 3:
        data1 = data.iloc[mice3, :]
        data2 = data.drop(mice3,axis=0)
    return data1,data2


def signalFilter(data, cutoff=26):
    from scipy import signal
    sos = signal.butter(4, cutoff, 'lowpass', fs=128, output='sos')
    dataf = pd.DataFrame(signal.sosfilt(sos, data.values, axis=1), index=data.index)
    return dataf


#%%
path = makePath()
Xa = importData("train_eeg1")
#Xb = importData("train_eeg2")
#Xc = importData("train_emg")

Xaf = signalFilter(Xa)



#%%
Xa1,Xa2 = miceSplitter(Xa,1)
epoch = 0
sig = Xa1.iloc[epoch,:]
sos = signal.butter(4,26,'lowpass',fs=128,output='sos')
filt = signal.sosfilt(sos, sig)
if True:
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(sig) ; ax2.plot(filt)
    plt.tight_layout() ;  plt.show()


#%%
path = makePath()
y = importData("train_labels")

#%%
yc = y.copy().values.ravel()
n = len(yc)
c = len(np.unique(yc))
cc = np.bincount(yc)[1:]
cw = n / (c * np.bincount(yc)[1:])
cwn = np.bincount(yc)[1:] / n
m=int(n/3)
mice1 = np.array(range(0,21600))
mice2 = np.array(range(21600,43200))
mice3 = np.array(range(43200,64800))
y1 = yc[mice1]
y2 = yc[mice2]
y3 = yc[mice3]
cw1 = np.bincount(y1)[1:] / m
cw2 = np.bincount(y2)[1:] / m
cw3 = np.bincount(y3)[1:] / m
cw_obs = {1: 1, 2: 1, 3: 6}
# %%
series = y
i = 2
w = 1000
plt.plot(series[i*w:(i+1)*w])
#%%
yc = y.copy().values.ravel()
steps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
changes = []
for s in steps:
    change = 0
    for i in range(s, len(yc)-s):
        if np.array(yc[i] != yc[i-s:i]).all():
            if np.array(yc[i] != yc[i+s:i+s+s]).all():
        #        if np.array(yc[i-s:i] == yc[i+s:i+s+s]).all():
                    for j in range(s):
                        yc[i+j] = yc[i-1]
                    change +=1
    changes.append(change)


# %%
path = makePath()
X1 = importData("train_eeg1")
row = X1.iloc[0,:].values
signal = np.array([])
for i,row in X1.iloc[0:10,:].iterrows():
    signal = np.concatenate((signal,row.values),axis=0)
import biosppy.signals.emg as emg
out = emg.emg(signal=signal, sampling_rate=256,show=True)
out = emg.find_onsets(signal=signal,sampling_rate=256)
