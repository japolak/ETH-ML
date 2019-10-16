#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 10:53:20 2017

@author: nicus
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import f1_score
import random
import time
import csv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
tf.logging.set_verbosity(tf.logging.ERROR)  # don't show warnings and messages
from sklearn import ensemble

start = time. time()
# Helper functions and classes

random.seed(1)
class Dataset:
    
    def __init__(self, data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._original_data = data
        self._num_examples = data.shape[0]
        pass
    
    def getData(self):
        return self._original_data
    
    @property
    def data(self):
        return self._data
    
    def next_batch(self, batch_size, shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexes
            self._data = self.data[idx]  # get list of 'num' random samples
            
        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = self.data[idx0]  # get list of 'num' random samples
            
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  # avoid the case where the #sample != integer times of batch_size
            end =  self._index_in_epoch
            data_new_part = self._data[start:end]
            full_data = np.concatenate((data_rest_part, data_new_part), axis = 0)
            return (full_data[:, 1:], full_data[:, 0].astype(int))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            full_data = self._data[start:end]
            return (full_data[:, 1:], full_data[:, 0].astype(int))
        
        
#%%

# Define constants
BASE_PATH =  "/home/leslie/Desktop/AML/Task3/aml/"   
LOG_PATH = os.path.join(BASE_PATH, "Logs/mymodel_final.ckpt")
# DATA_TRAINING = os.path.join(BASE_PATH, "X_train.csv")
# DATA_TEST = os.path.join(BASE_PATH, "X_test.csv")
# Y_TRAIN = = os.path.join(BASE_PATH, "y_train.csv")

WORK_FOLDER = '/home/leslie/Desktop/AML/Task3/aml/'


def load_person_average_data():
    # function to prepare the data & impute any missing values
    
    # load data 
    # person_averages = pd.read_csv(WORK_FOLDER + "hamilton_upsampled.csv", dtype=np.float64)
    # y_train_data = pd.read_csv(WORK_FOLDER + "y_hamilton_upsampled.csv", dtype=np.float64)
    # x_test_averages = pd.read_csv(WORK_FOLDER + 'test_avg_heartbeat.csv', dtype=np.float64)

    # person_averages = pd.read_csv(WORK_FOLDER + "1.25.mean.75.9train_quantile_heartbeat_samples.csv", dtype=np.float64)
    # y_train_data = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
    # x_test_averages = pd.read_csv(WORK_FOLDER + "1.25.mean.75.9test_quantile_heartbeat_samples.csv", dtype=np.float64)

    # person_averages = pd.read_csv(WORK_FOLDER + "train_quantile_heartbeat_samples.csv", dtype=np.float64)
    # y_train_data = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
    # x_test_averages = pd.read_csv(WORK_FOLDER + "test_quantile_heartbeat_samples.csv", dtype=np.float64)

    person_averages = pd.read_csv(WORK_FOLDER + "1255759_10train_quantile_heartbeat_samples.csv", dtype=np.float64)
    y_train_data = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
    x_test_averages = pd.read_csv(WORK_FOLDER + "1255759_10test_quantile_heartbeat_samples.csv", dtype=np.float64)


    # person_averages = pd.read_csv(WORK_FOLDER + "train10.csv", dtype=np.float64)
    # y_train_data = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
    # x_test_averages = pd.read_csv(WORK_FOLDER + "test10.csv", dtype=np.float64)

    # person_averages = pd.read_csv(WORK_FOLDER + "05.25.5.75.95train_quantile_heartbeat_samples.csv", dtype=np.float64)
    # y_train_data = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
    # x_test_averages = pd.read_csv(WORK_FOLDER + "05.25.5.75.95test_quantile_heartbeat_samples.csv", dtype=np.float64)
    

    # person_averages = pd.read_csv(WORK_FOLDER + "X_train_features.csv", dtype=np.float64, header=None)
    # y_train_data = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
    # x_test_averages = pd.read_csv(WORK_FOLDER + 'X_test_features.csv', dtype=np.float64, header=None) 

    # person_averages = pd.read_csv(WORK_FOLDER + "hamilton_filtered_rowlen.csv", dtype=np.float64)
    # y_train_data = pd.read_csv(WORK_FOLDER + "y_train.csv", dtype=np.float64)
    # x_test_averages = pd.read_csv(WORK_FOLDER + 'test_avg_heartbeat.csv', dtype=np.float64)

    # person_averages = pd.read_csv(WORK_FOLDER + "hamilton_all_heart_beat_samples.csv", dtype=np.float64)
    # y_train_data = pd.read_csv(WORK_FOLDER + "hamilton_all_heart_beat_class.csv", dtype=np.float64)
    # x_test_averages = pd.read_csv(WORK_FOLDER + 'test_all_heartbeat.csv', dtype=np.float64)
 

    # person_averages = pd.read_csv(WORK_FOLDER + "ecg_all_heart_beat_samples.csv", dtype=np.float64)
    # y_train_data = pd.read_csv(WORK_FOLDER + "ecg_all_heart_beat_class.csv", dtype=np.float64)
    # x_test_averages = pd.read_csv(WORK_FOLDER + 'ecg_test_all_heartbeat.csv', dtype=np.float64)


    # Remove ID column from x train
    col_names_x = person_averages.columns.get_values()
    col_names_x = col_names_x.tolist()
    features = col_names_x[0:]
    x_train_data = person_averages[features]
    
    # Remove id column from y train
    col_names_y = y_train_data.columns.get_values()
    col_names_y = col_names_y.tolist()
    response = col_names_y[1:]
    ids = col_names_y[0]
    id_data = y_train_data[ids]
    y_train_data = y_train_data[response]
    

    # ## Scale data
    scaler = StandardScaler()  
    scaler.fit(person_averages)  
    person_averages = scaler.transform(person_averages)
    x_test_averages = scaler.transform(x_test_averages)

    # convert to pandas dataframe
    person_averages = pd.DataFrame(data=person_averages)
    y_train_data = pd.DataFrame(data=y_train_data)
    x_test_averages = pd.DataFrame(data=x_test_averages)
    
    # return the data in a tuple 
    return(person_averages, y_train_data, x_test_averages, id_data)


# Import and prepare data
X1, y1, DATA_TEST, id_data = load_person_average_data()
X1.as_matrix().astype(np.float32)
y1.as_matrix().astype(np.int)
DATA_TEST.as_matrix().astype(np.float64)

# Define DNN constants
n_inputs = DATA_TEST.shape[1]
n_outputs = len(np.unique(y1))
n_epochs = 50
batch_size = 50
learning_rate = 0.0001

# Handler function to build the Dnn layers
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 * np.sqrt(2.0/(n_inputs + n_outputs))
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
        W = tf.Variable(init, name = "weights")
        b = tf.Variable(tf.zeros([n_neurons]), name = "biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return(tf.nn.relu(z))
        if activation == "tanh":
            return(tf.nn.tanh(z))
        if activation == "sigmoid":
            return(tf.nn.sigmoid(z))
        else:
            return(z)

def run_tf(n_hidden1, n_hidden2, n_hidden3, activation_fn_layer_1, activation_fn_layer_2, activation_fn_layer_3, dropout_rate, DATA_TEST=DATA_TEST):
    # Set up placeholder for inputs
    X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
    y = tf.placeholder(tf.int64, shape=(None), name = "y")

    keep_prob = tf.placeholder(tf.float32)

    # USE THIS TO TRAIN THE MODEL FIRST
    # X_train, X_test, y_train, y_test = train_test_split(X1, y1, train_size = 0.9, stratify = y1)
    # X_new = X_test
    # train = np.column_stack((y_train, X_train)) # put together y_train and X_train
    # dataset = Dataset(train)  # instantiate Dataset object

    mean_acc = []
    kf = StratifiedKFold(n_splits = 10)
    kf.get_n_splits(X1, y1)
    for train_index, test_index in kf.split(X1,y1):
        

        X_train, X_test = X1.iloc[train_index,:], X1.iloc[test_index,:]
        y_train, y_test = y1.iloc[train_index,:], y1.iloc[test_index,:]
        X_new = X_test 
        

        train = np.column_stack((y_train, X_train)) # put together y_train and X_train
        dataset = Dataset(train)  # instantiate Dataset object

       
    # # use real model!
    # X_new = DATA_TEST.as_matrix().astype(np.float64)
    # X_train = X1
    # y_train = y1
    # full_train = np.column_stack((y1, X1))
    # dataset = Dataset(full_train)

    # for i in range(1):
    # Design Dnn     
        # X_drop = tf.nn.dropout(X, keep_prob)
        with tf.name_scope("dnn"):
            hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation = activation_fn_layer_1)
            hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
            
            hidden2 = neuron_layer(hidden1_drop, n_hidden2, "hidden2", activation  = activation_fn_layer_2)
            hidden2_drop = tf.nn.dropout(hidden2,keep_prob)
            
            hidden3 = neuron_layer(hidden2_drop, n_hidden3, "hidden3", activation = activation_fn_layer_3)
            hidden3_drop = tf.nn.dropout(hidden3,keep_prob)
           
            # hidden4 = neuron_layer(hidden3_drop, 500, "hidden4", activation = "sigmoid")
            # hidden4_drop = tf.nn.dropout(hidden4,keep_prob)

            # hidden5 = neuron_layer(hidden3_drop, 500, "hidden4", activation = "relu")
            # hidden5_drop = tf.nn.dropout(hidden5,keep_prob)

            logits  = neuron_layer(hidden3_drop, n_outputs, "outputs")
            
        # Specify loss function
        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                 labels = y, logits = logits)
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

        #%%
        # Execution phase

        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                
                for iteration in range(len(X_train) // batch_size):
                    X_batch, y_batch = dataset.next_batch(batch_size)

                    sess.run(training_op, feed_dict = {X : X_batch, y: y_batch, 
                                                       keep_prob: dropout_rate})

            
                # acc_train = accuracy.eval(feed_dict = {X : X_batch, y: y_batch, 
                #                                        keep_prob: 1.0})
                # acc_test  = accuracy.eval(feed_dict = {X : X_test,  y: y_test,  
                #                                        keep_prob: 1.0})
            
                # print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
                
            save_path = saver.save(sess, LOG_PATH)


        # Predict

        with tf.Session() as sess:
            saver.restore(sess, LOG_PATH)
            Z = logits.eval(feed_dict = {X: X_new, keep_prob: 1.0})
            y_pred = np.argmax(Z, axis = 1)
          
        print(n_hidden1, activation_fn_layer_1, n_hidden2, activation_fn_layer_2, n_hidden3, activation_fn_layer_3, dropout_rate)
        print(confusion_matrix(y_test, y_pred))
        f1_result = f1_score(y_test, y_pred, average='micro')
        print(f1_result)
        mean_acc.append(f1_result)    

        # # keep track of different combinatoins that were attempted
        with open("different_attempts.csv", "a+") as myfile:
            writer = csv.writer(myfile)
            writer.writerow([n_hidden1, activation_fn_layer_1, n_hidden2, activation_fn_layer_2, n_hidden3, activation_fn_layer_3, dropout_rate,f1_result, learning_rate])
            # myfile.write()

        end = time. time()
        print((end-start))
    print("--------------------------")
    print("--------------------------")
    print("--------------------------")
    print(mean_acc)
    print("mean, sd", np.mean(np.array(mean_acc)), np.std(np.array(mean_acc)))
    
    with open("tf_counts.csv", "a+") as myfile:
            writer = csv.writer(myfile)
            writer.writerow([n_hidden1, activation_fn_layer_1, n_hidden2, activation_fn_layer_2, n_hidden3, activation_fn_layer_3, dropout_rate,f1_result, learning_rate, np.mean(np.array(mean_acc)), np.std(np.array(mean_acc))])
    # # # Write to csv
    # results = y_pred
    # id_data2 = np.arange(0, len(DATA_TEST))  
    # results = pd.DataFrame({'id': id_data2, "y": results})
    # results.to_csv(BASE_PATH + 'task3_dnn.csv', index = False)


for i in range(4):
    run_tf(n_hidden1=500, 
        n_hidden2=500, 
        n_hidden3=500, 
        activation_fn_layer_1="relu", 
        activation_fn_layer_2="tanh", 
        activation_fn_layer_3="sigmoid", 
        dropout_rate=0.75)

    













        # xentropy = tf.nn.weighted_cross_entropy_with_logits(targets = tf.cast(y, tf.float32), logits = logits, pos_weight=weights)
        # weight_per_label = tf.transpose( tf.matmul(tf.cast(y, tf.float32), tf.transpose(weight_per_class)) )
        # xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #          labels = y, logits = logits)
        # xentropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=tf.cast(y, tf.float32), pos_weight=weights)
        # xentropy = tf.mul(weight_per_label, tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y, name="xent_raw"))

        # xentropy = tf.nn.weighted_cross_entropy_with_logits(targets=tf.cast(y, tf.float32), logits=logits, pos_weight = tf.constant([0.8,0.5,0.6,1]))
    # weight_per_class = tf.constant([0.59, 0.09, 0.29, 0.03])
            # # weight_per_class = tf.constant([1.0, 6.0, 2.0, 17.0])
            # onehot_labels = tf.one_hot(y, depth=4)
            # weights = tf.reduce_sum(tf.multiply(onehot_labels, weight_per_class), axis=1) # shape (batch_size, num_classes)
            # xentropy = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, 
            #     weights=weights, reduction=tf.losses.Reduction.SUM)
            # loss = tf.reduce_mean(xentropy, name = "loss")