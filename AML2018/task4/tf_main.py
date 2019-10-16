#import sys
#sys.path.append(os.path.realpath("/home/fred/Documents/ETH/HS2018/Advanced_Machine_Learning/Projects/Task4/"))
# %reset
import os
import tensorflow as tf
import numpy as np
import sys
import copy
import pandas as pd
sys.path.append(os.path.realpath("/home/fred/Documents/ETH/HS2018/Advanced_Machine_Learning/Projects/Task4/"))
from tf_utils import input_fn_from_dataset,input_fn_frame_from_dataset,save_tf_record,prob_positive_class_from_prediction
from get_data import get_videos_from_folder,get_target_from_csv
from utils import save_solution
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

dir_path = os.path.realpath("/home/fred/Documents/ETH/HS2018/Advanced_Machine_Learning/Projects/Task4/")
train_folder = "task4-data/train/"
test_folder  = "task4-data/test"
train_target = "task4-data/train_target.csv"
my_solution_file = os.path.join(dir_path,'predictions/my_sol.csv')

"""
Load data
"""
x_train = get_videos_from_folder(train_folder)
y_train = get_target_from_csv(train_target)
x_test = get_videos_from_folder(test_folder)
"""
###############################################################################
###############################################################################
### Tensorflow
###############################################################################
###############################################################################
"""
"""
Extending the videos
"""
def extend_video(x_data,maxlength=209):
    x_dataE=x_data[:]
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
train_ext= extend_video(x_train)
test_ext= extend_video(x_test)
"""
Reshaping to new format
"""
# Train data
train_nr_videos=train_ext.shape[0]
new_x_train = np.empty((train_nr_videos, 209, 100, 100),dtype=np.uint8)
for k in range(train_nr_videos):
    new_x_train[k,:,:,:] =train_ext[k][:, :, :]       # as a 4 D tensor (1st dimension video number)
new_X_train=np.reshape(new_x_train, [train_nr_videos,209, 100, 100, 1]) # adding color channel (5th D)
# Test data
test_nr_videos=test_ext.shape[0]
new_x_test = np.empty((test_nr_videos, 209, 100, 100),dtype=np.uint8)
for k in range(test_nr_videos):
    new_x_test[k,:,:,:] =test_ext[k][:, :, :]       # as a 4 D tensor (1st dimension video number)
new_X_test=np.reshape(new_x_test, [test_nr_videos,209, 100, 100, 1]) # adding color channel (5th D)

"""
A couple of tests to see if we were successful
%matplotlib inline
plt.imshow(test_ext[10][50,:,:], cmap="gray") # plot 1st image's 2nd feature map
plt.imshow(new_X_test[10,50,:,:,0], cmap="gray")
plt.imshow(x_test[10][50,:,:], cmap="gray")
(test_ext[10][50,:,:]==new_X_test[10,50,:,:,0]).all()
(new_X_test[10,50,:,:,0]==x_test[10][50,:,:]).all()
"""

#########################
#########################
# Designing the graph
#########################
#########################
"""
General remarks:
    i. we work with videos in W/B. We thus have as an input a tensor
      of size (nr of frames -i.e. time-)x(height)x(width)
      --> it is 3D (like a picture with three colors...).
      ii. We pass a video at the time (with all the frames) and classify it

Ideas for the architecture:
    a. Dropout for regularization [see ex. book, JUST AT THE END]
    b. 3D Maps


"""


"""
PARAMETERS TO TUNE:
    Number of Maps per convolution:
        a. NrMaps_c1
        b. ...
"""
BASE_PATH =  "/home/fred/Documents/ETH/HS2018/Advanced_Machine_Learning/Projects/Task4/task4-data/"
LOG_PATH = os.path.join(BASE_PATH, "Logs/mymodel_final.ckpt")
tf.reset_default_graph()

"""
PLOTTING:

%matplotlib inline
plt.imshow(x_train[1][1], cmap="gray") # plot 1st image's 2nd feature map
X_batch =x_train[1]
X_batch = np.reshape(X_batch, [-1, 100, 100, 1])
plt.imshow(X_batch[1,:,:,0], cmap="gray")
"""
n_epochs = 50
batch_size = 2
learning_rate = 0.001
X = tf.placeholder(tf.float32, shape = (None,209,100,100,1), name = "X")
y = tf.placeholder(tf.int64, shape=(None), name = "y")

with tf.name_scope("cnn"):
  # passing the X...
  """
  Open Question:
     1. I think we should pass SEVERAL videos at the time (batch).
     The problem here is that they are all of different sizes
     (array needs same dimension?!).
     2. The convolution should be w.r.t. both height, width and time.
     At the beginning I thought it was the same as working "with
     multiple color channels".The problem is, again, that the time
     varies... SHOULD WE USE conv3d?

  Remark:
      we might as well start from an easy structure...
      The following is the "digits example"
  """
  # Convolutional Layer
  # input_layer = tf.reshape(X, [2, 209, 100, 100, 1])

  conv1 = tf.layers.conv3d(inputs=X, filters=32,
                             kernel_size=[3,3,3],
                             padding="same",
                             activation=tf.nn.relu)
    # Pooling Layer
  pool1 = tf.layers.max_pooling3d(inputs=conv1,   \
                                    pool_size=[2,2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer
  conv2 = tf.layers.conv3d( inputs=pool1, filters=36,
                             kernel_size=[3,3,3],
                             padding="same",
                             activation=tf.nn.relu)

  pool2 = tf.layers.max_pooling3d(inputs=conv2,\
                              pool_size=[2,2,2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [2, 25*25 * 36])
  dense = tf.layers.dense(inputs=pool2_flat,\
                            units= 25*25*36, activation=tf.nn.relu)

  dropout = tf.layers.dropout(inputs=dense, rate=0.5)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=2)
  # tf.nn.softmax(logits, name="softmax_tensor")



# Specify loss function
with tf.name_scope("loss"):
    xentropy = tf.losses.sparse_softmax_cross_entropy(labels=y, \
                                                      logits=logits)
    loss = tf.reduce_mean(xentropy, name = "loss")

# Specify gradient descent optimizer
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# Model evaluation
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Execution phase
with tf.Session() as sess:
    init.run()
    """
    Issue: I need to pass video after video because I have problems
    dealing with this thing's dimension! In particular:
    I believe I should pass an array with four dimensions (and add a
    5th one for the "color"). Having a "list" (a 1D array) with the
    different videos makes it difficult to pass more than one video at
    the time...
    """
    num_batches=len(new_X_train)/batch_size
    if len(new_X_train) % batch_size ==0:
        for batch_number in range(num_batches):
            starting_point=batch_number*batch_size
            ending_point=(batch_number+1)*batch_size
            X_batch=new_X_train[starting_point:ending_point,:,:,:,:]
            y_batch=y_train[starting_point:ending_point]
            X_batch = X_batch.astype(np.float32)
            y_batch = y_batch.astype(np.int64)
            sess.run(training_op,
                         feed_dict = {X: X_batch, y: y_batch})
    else:
       for batch_number in range(num_batches+1):
           if batch_number<=num_batches:
                starting_point=batch_number*batch_size
                ending_point=(batch_number+1)*batch_size
                X_batch=new_X_train[starting_point:ending_point,:,:,:,:]
                y_batch=y_train[starting_point:ending_point]
           else:
                starting_point=(batch_number+1)*batch_size+1
                ending_point=len(new_X_train)
                if starting_point<ending_point:
                    X_batch=new_X_train[starting_point:ending_point,:,:,:,:]
                    y_batch=y_train[starting_point:ending_point]
                elif starting_point==ending_point:
                    X_batch=new_X_train[starting_point,:,:,:,:]
                    y_batch=y_train[starting_point]
           X_batch = X_batch.astype(np.float32)
           y_batch = y_batch.astype(np.int64)
           sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
    # need to add a dimension (for color)
    save_path = saver.save(sess, LOG_PATH)

# roc_auc = roc_auc_score(y_true, prob_positive_class)

# Predict
# Predict

with tf.Session() as sess:
    saver.restore(sess, LOG_PATH)
    y_pred=[]
    for i in range(x_test.shape[0]):
        X_batch =x_test[i]
        nr_frames=X_batch.shape[0]
        # need to add a dimension (for color)
        X_batch = np.reshape(X_batch, [-1, 100, 100, 1])
        X_batch = X_batch.astype(np.float32)
        Z = logits.eval(feed_dict = {X: X_batch})
        y_pred.append(Z)

y_array= np.empty(shape=[69,2])
for i in range(69):
    norm= np.sum(abs(y_pred[i][:,:]))
    p0 = np.sum(abs(y_pred[i][:,0]))/norm
    p1 = np.sum(abs(y_pred[i][:,1]))/norm
    y_array[i,0]=p0
    y_array[i,1]=p1

id_data2 = np.arange(0, len(y_array))
results = pd.DataFrame({'id': id_data2, "y": y_array[:,1] })
results.to_csv(BASE_PATH + 'task4_cnn.csv', index = False)
