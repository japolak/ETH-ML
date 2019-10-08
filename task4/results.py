# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 

nepoch = 500

df_loss = pd.DataFrame(np.linspace(1,nepoch,nepoch),columns=['epoch'])
df_acc = pd.DataFrame(np.linspace(1,nepoch,nepoch),columns=['epoch'])
#%%

directory = '/Users/Jakub/Documents/ETH/S19/IML/task4/logs/'

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        nname = filename[:-4]
        df_loss[nname] = pd.read_csv(directory+filename,usecols=[0])
        df_acc[nname] = pd.read_csv(directory+filename,usecols=[1])
        continue
    else:
        continue
#%%

nmodels = df_loss.shape[1]-1
nfrom = 30

for i in range(nmodels):   
    i=i+1
#    plt.plot(df_loss.iloc[nfrom:,0],df_loss.iloc[nfrom:,i],)
    plt.plot(df_acc.iloc[nfrom:,0],df_acc.iloc[nfrom:,i])
plt.legend()
plt.fig
plt.show()



