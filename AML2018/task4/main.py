%reset
import os
import copy
import numpy as np
import matplotlib as plt
from get_data import get_videos_from_folder,get_target_from_csv,extend_video,reshape_data
from utils import save_solution

#skvideo.setFFmpegPath('/Library/miniconda3/lib/python3.7/site-packages/ffmpeg')
dir_path = os.path.dirname(os.path.realpath('/Users/Jakub/Documents/ETH/AML/task4'))
train_folder = os.path.join(dir_path,"/Users/Jakub/Documents/ETH/AML/task4/train/")
test_folder = os.path.join(dir_path,"/Users/Jakub/Documents/ETH/AML/task4/test/")

train_target = os.path.join(dir_path,'/Users/Jakub/Documents/ETH/AML/task4/train_target.csv')
my_solution_file = os.path.join(dir_path,'/Users/Jakub/Documents/ETH/AML/task4/solution.csv')

y_train = get_target_from_csv(train_target)
x_train = get_videos_from_folder(train_folder)
x_test = get_videos_from_folder(test_folder)

#E=Extended videos
x_trainE=copy.copy(extend_video(x_train))
x_testE=copy.copy(extend_video(x_test))

#R=Reshaped videos
x_trainER=copy.copy(reshape_data(x_trainE))
x_testER=copy.copy(reshape_data(x_testE))
    
#%%
##### A couple of tests....
## Test for equality of extension
#%matplotlib inline  
#plt.pyplot.imshow(x_trainE[10][50,:,:], cmap="gray") # 10th video has only 49 frames so should equal to first
#plt.pyplot.imshow(x_train[10][0,:,:], cmap="gray") 
#(x_trainE[10][50,:,:]==x_train[10][0,:,:]).all()
#
## Test for equality of reshape
#plt.pyplot.imshow(x_trainER[10,50,:,:,0], cmap="gray") # 10th video has only 49 frames so should equal to first
#plt.pyplot.imshow(x_train[10][0,:,:], cmap="gray") 
#(x_trainER[10,:,:,:,0]==x_trainE[10][:,:,:]).all()
#
#
#plt.pyplot.imshow(x_trainER[10,208,:,:,0], cmap="gray")

    
    