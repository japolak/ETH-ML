# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


#%%
nepoch=500
nfrom = 100
metric = 1  # 0 loss, 1 accuracy
cv = 4

df_mean = pd.DataFrame(np.linspace(0,nepoch,nepoch),columns=['epoch'])
df = pd.DataFrame(np.linspace(0,nepoch,nepoch),columns=['epoch'])

#plot cv
model = 'ResNN' # ResNN or NN
for i in range(cv):
    i+=1
    fold='CV'+str(i)
    filename=model+'1CV'+str(i)
    df[fold]=pd.read_csv('logs/'+filename+'.csv',usecols=[metric])
    plt.plot(np.linspace(nfrom,nepoch,nepoch-nfrom),df[fold][nfrom:])
plt.show()

df_mean[model]=df.iloc[:,1:].mean(axis=1)


plt.plot(np.linspace(nfrom,nepoch,nepoch-nfrom),df_mean[model][nfrom:])
plt.show()
#%%

mean = df_mean.drop(['epoch'],axis=1)
mean.to_csv('ResNet-A-5_1-512.csv',index=False)
#%%


modelnames = [
        'ResNet-OG1-2-512',#'ResNet-PA1-2-512',
        'ResNet-OG2-1-512',#'ResNet-PA2-1-512',
        'ResNet-OG1-4-512',#'ResNet-PA1-4-512'
#       ,'ResNet-OG2-2-512','ResNet-PA2-2-512'
              ]
nepoch = 700
nfrom = 100
metric = 0

df = pd.DataFrame(np.linspace(0,nepoch,nepoch),columns=['epoch'])
for i in modelnames:
    df[i]=pd.read_csv('logs/'+i+'.csv',usecols=[metric])
    plt.plot(np.linspace(nfrom,nepoch,nepoch-nfrom),df[i][nfrom:])
    plt.legend()
    
plt.show
