import os
import tensorflow as tf	
import numpy as np
import copy
from tf_utils import input_fn_from_dataset,input_fn_frame_from_dataset,save_tf_record,prob_positive_class_from_prediction
from get_data import get_videos_from_folder,get_target_from_csv,extend_video,reshape_data
from utils import save_solution

#skvideo.setFFmpegPath('/Library/miniconda3/lib/python3.7/site-packages/ffmpeg')
dir_path = os.path.dirname(os.path.realpath('/Users/Jakub/Documents/ETH/AML/task4/'))
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

tf_record_dir = os.path.join(dir_path,'task4','tf_records/')
os.makedirs(tf_record_dir, exist_ok=True)

tf_record_train = os.path.join(tf_record_dir, 'train' + '.tfrecords')
tf_record_test = os.path.join(tf_record_dir, 'test' + '.tfrecords')

if not os.path.exists(tf_record_train):
    x_train = get_videos_from_folder(train_folder)
    y_train = get_target_from_csv(train_target)
    save_tf_record(x_train,tf_record_train,y = y_train)

if not os.path.exists(tf_record_test):
    x_test = get_videos_from_folder(test_folder)
    save_tf_record(x_test,tf_record_test)	

feature_col_video = tf.feature_column.numeric_column('video',shape=[212,100,100],dtype=tf.uint8)
feature_col_frame = tf.feature_column.numeric_column('frame',shape = [100,100],dtype=tf.uint8)†
batchsize_video = 1

estimator = tf.estimator.LinearClassifier(feature_columns = [feature_col_video],model_dir='tf_checkpoints')
estimator.train(input_fn=lambda: input_fn_from_dataset(tf_record_train, batch_size=batchsize_video),max_steps=1)
pred = estimator.predict(input_fn=lambda: input_fn_from_dataset(tf_record_test, batch_size=batchsize_video,num_epochs=1,shuffle = False))
dummy_solution = prob_positive_class_from_prediction(pred)
save_solution(my_solution_file,dummy_solution)