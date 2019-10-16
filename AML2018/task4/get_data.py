import numpy as np
import skvideo.io
import os
import pandas as pd
import copy

dir_path = os.path.dirname(os.path.realpath('/Users/Jakub/Documents/ETH/AML/task4'))
#%%
def get_videos_from_folder(data_folder):
	'''
	get a list of video x wehre each video is a numpy array in the format [n_frames,width,height]
	with uint8 elements.
	argument: relative path to the data_folder from the source folder.
	'''
	data_folder = os.path.join(dir_path,data_folder)
	x = []
	file_names = []
	if os.path.isdir(data_folder):
		for dirpath, dirnames, filenames in os.walk(data_folder):
			for filename in filenames:
				file_path = os.path.join(dirpath, filename)
				statinfo = os.stat(file_path)
				if statinfo.st_size != 0:
					video = skvideo.io.vread(file_path, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
					x.append(video)
					file_names.append(int(filename.split(".")[0]))
	indices = sorted(range(len(file_names)), key=file_names.__getitem__)
	x = np.take(x,indices)
	return x

def get_target_from_csv(csv_file):
	'''
	get a numpy array y of labels. the order follows the id of video.
	argument: relative path to the csv_file from the source folder.
	'''
	csv_file = os.path.join(dir_path,csv_file)
	with open(csv_file, 'r') as csvfile:
		label_reader = pd.read_csv(csvfile)
		y = label_reader['y']
	y = np.array(y)
	return y

def extend_video(x_data,maxlength=209):
    '''
    Extend/repeat each video to equal length.
    Longest video was 209 frames, so set as default)
    '''
    x_dataE=copy.copy(x_data)
    for i in range(x_dataE.shape[0]):
        video=copy.copy(x_dataE[i])
        frame=video.shape[0]
        nvideo=np.zeros((maxlength,100,100),dtype=np.uint8)
        nvideo[0:frame,:,:]=copy.copy(video[0:frame,:,:])
        j = frame
        while j<maxlength:
            for k in range(frame):
                nvideo[j,:,:]=copy.copy(video[k,:,:])
                j += 1
                if j== (maxlength):
                    break
        x_dataE[i]=copy.copy(nvideo)
    return x_dataE

def reshape_data(x_data):
    '''
    Takes as an input x_train and copies the individual videos
    into the first dimension of the arrray and adds last dimension for color
    output: [video, frame, width, heigth, color]
    '''    
    data=copy.copy(x_data)
    N=x_data.shape[0]
    ndata=np.empty((N,209,100,100),dtype=np.uint8)
    for i in range (data.shape[0]):
        ndata[i,:,:,:]=data[i][:,:,:]
    nndata=np.reshape(ndata,[N,209,100,100,1])
    return nndata

'''
# Example of running it in main.py

import copy
###   E=Extend videos
x_trainE=copy.copy(extend_video(x_train))
x_testE=copy.copy(extend_video(x_test))
###   R=Reshape videos
x_trainER=copy.copy(reshape_data(x_trainE))
x_testER=copy.copy(reshape_data(x_testE))
'''



