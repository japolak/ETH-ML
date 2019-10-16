import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.metrics import f1_score
import random
import time
import csv
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

tf.logging.set_verbosity(tf.logging.ERROR)  # don't show warnings and messages

start_time = time. time()
# Helper functions and classes

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
BASE_PATH =  "/home/leslie/Desktop/AML/Task5/"   
LOG_PATH = os.path.join(BASE_PATH, "Logs/mymodel_final.ckpt")
sys.path.append(os.path.realpath("/home/leslie/Desktop/AML/Task5/"))
# DATA_TRAINING = os.path.join(BASE_PATH, "X_train.csv")
# DATA_TEST = os.path.join(BASE_PATH, "X_test.csv")
# Y_TRAIN = = os.path.join(BASE_PATH, "y_train.csv")

WORK_FOLDER = '/home/leslie/Desktop/AML/Task5/'


# LEARNING_RATE = float(sys.argv[1])
# BETA2 = float(sys.argv[2])
# BETA1 = float(sys.argv[3])
# DROPOUT = float(sys.argv[4])
# ACT1 = sys.argv[5]
# ACT2 = sys.argv[6]
# ACT3 = sys.argv[7]

LEARNING_RATE = 0.0001
BETA2 = 0.9
BETA1 = 0.9
DROPOUT = 0.8
ACT1 = "relu"
ACT2 = "relu"
ACT3 = "relu"

print(LEARNING_RATE, BETA2, BETA1, DROPOUT, ACT1, ACT2, ACT3)

def load_person_average_data():
    # function to prepare the data & impute any missing values
    
    # load data
    train_eeg1 = pd.read_csv(WORK_FOLDER + "data/train_eeg1_features_v0.csv", dtype=np.float64, header=None)
    train_eeg2 = pd.read_csv(WORK_FOLDER + "data/train_eeg2_features_v0.csv", dtype=np.float64, header=None)
    train_emg = pd.read_csv(WORK_FOLDER + "data/train_emg_features_v0.csv", dtype=np.float64, header=None)
    train_eeg1_v2 = pd.read_csv(WORK_FOLDER + "data/train_eeg1_features_v1.csv", dtype=np.float64, header=None)
    train_eeg2_v2 = pd.read_csv(WORK_FOLDER + "data/train_eeg2_features_v1.csv", dtype=np.float64, header=None)
    train_emg_v2 = pd.read_csv(WORK_FOLDER + "data/train_emg_features_v1.csv", dtype=np.float64, header=None)
    # train_eeg1r = pd.read_csv(WORK_FOLDER + "data/train_eeg1_features_r.csv", dtype=np.float64, header=None)
    # train_eeg2r = pd.read_csv(WORK_FOLDER + "data/train_eeg2_features_r.csv", dtype=np.float64, header=None)
    # train_emgr = pd.read_csv(WORK_FOLDER + "data/train_emg_features_r.csv", dtype=np.float64, header=None)
    # quantiles_eeg1 = pd.DataFrame(data=np.quantile(train_eeg1, (0.1, 0.25, 0.75, 0.9), axis=1).T)
    # quantiles_eeg2 = pd.DataFrame(data=np.quantile(train_eeg2, (0.1, 0.25, 0.75, 0.9), axis=1).T)
    quantiles_emg = pd.DataFrame(data=np.quantile(np.log(train_emg+1), 0.75, axis=1))
    std_emg = pd.DataFrame(data=np.std(np.log(train_emg+1)))

    # standardized_train = pd.read_csv(WORK_FOLDER + "data/train_standardized.csv", header=None)
    # standardized_test = pd.read_csv(WORK_FOLDER + "data/test_standardized.csv", header=None)

    y_train_data = pd.read_csv(WORK_FOLDER + "data/train_labels.csv", dtype=np.float64)
    
    test_eeg1 = pd.read_csv(WORK_FOLDER + "data/test_eeg1_features_v0.csv", dtype=np.float64, header=None)
    test_eeg2 = pd.read_csv(WORK_FOLDER + "data/test_eeg2_features_v0.csv", dtype=np.float64, header=None)
    test_emg = pd.read_csv(WORK_FOLDER + "data/test_emg_features_v0.csv", dtype=np.float64, header=None)
    test_eeg1_v2 = pd.read_csv(WORK_FOLDER + "data/test_eeg1_features_v1.csv", dtype=np.float64, header=None)
    test_eeg2_v2 = pd.read_csv(WORK_FOLDER + "data/test_eeg2_features_v1.csv", dtype=np.float64, header=None)
    test_emg_v2 = pd.read_csv(WORK_FOLDER + "data/test_emg_features_v1.csv", dtype=np.float64, header=None)
    # test_eeg1r = pd.read_csv(WORK_FOLDER + "data/test_eeg1_features_r.csv", dtype=np.float64, header=None)
    # test_eeg2r = pd.read_csv(WORK_FOLDER + "data/test_eeg2_features_r.csv", dtype=np.float64, header=None)
    # test_emgr = pd.read_csv(WORK_FOLDER + "data/test_emg_features_r.csv", dtype=np.float64, header=None)
    # quantiles_eeg1_test = pd.DataFrame(data=np.quantile(test_eeg1, (0.1, 0.25, 0.75, 0.9), axis=1).T)
    # quantiles_eeg2_test = pd.DataFrame(data=np.quantile(test_eeg2, (0.1, 0.25, 0.75, 0.9), axis=1).T)
    # quantiles_emg_test = pd.DataFrame(data=np.quantile(test_emg, (0.1, 0.25, 0.75, 0.9), axis=1).T)
    
    quantiles_emg_test = pd.DataFrame(data=np.quantile(np.log(test_emg+1), 0.75, axis=1))
    std_emg_test = pd.DataFrame(data=np.std(np.log(test_emg+1)))

    # Remove ID column from x train
    # col_names_x = train_eeg1.columns.get_values()
    # col_names_x = col_names_x.tolist()
    # features = col_names_x[0:]
    # print(train_eeg1[features].iloc[0:5,:])
    
    # without r features
    # x_train_data = pd.concat([train_eeg1, train_eeg2, train_emg], ignore_index=True, sort=True, axis=1)
    # x_test_data = pd.concat([test_eeg1, test_eeg2, test_emg], ignore_index=True, sort=True, axis=1)

    # without r features, added std and 0.75
    x_train_data = pd.concat([train_eeg1, train_eeg2, train_emg, train_eeg1_v2, train_eeg2_v2, train_emg_v2, quantiles_emg, std_emg], ignore_index=True, axis=1)
    
    #remove mouse 3
    # x_train_data = x_train_data.iloc[:43200,:]


    x_test_data = pd.concat([test_eeg1, test_eeg2, test_emg, test_eeg1_v2, test_eeg2_v2, test_emg_v2, quantiles_emg_test, std_emg_test], ignore_index=True, axis=1)


    # without r features but with quantiles
    # x_train_data = np.array(pd.concat([train_eeg1, train_eeg2, train_emg, quantiles_eeg1, quantiles_eeg2, quantiles_emg], ignore_index=True, sort=True, axis=1))
    # x_test_data = np.array(pd.concat([test_eeg1, test_eeg2, test_emg, quantiles_eeg1_test, quantiles_eeg2_test, quantiles_emg_test], ignore_index=True, sort=True, axis=1))

    # with r features
    # x_train_data = pd.concat([train_eeg1, train_eeg2, train_emg, train_eeg1r, train_eeg2r, train_emgr, quantiles_emg, std_emg], ignore_index=True, sort=True, axis=1)
    # x_test_data = pd.concat([test_eeg1, test_eeg2, test_emg, test_eeg1r, test_eeg2r, test_emgr, quantiles_emg_test, std_emg_test], ignore_index=True, sort=True, axis=1)
        
    # # Remove id column from y train
    col_names_y = y_train_data.columns.get_values()
    col_names_y = col_names_y.tolist()
    response = col_names_y[1:]
    ids = col_names_y[0]
    id_data = y_train_data[ids]
    y_train_data = pd.DataFrame(data=np.array(y_train_data[response]) - 1)

    # remove mouse 3
    # y_train_data = y_train_data.iloc[:43200]

    # 34114, 27133
    # Subsample:
    print("data is read")
    end = time. time()
    print((end-start_time))
    
    class0_indices = np.array(np.where(y_train_data == 0)[0])
    class1_indices = np.array(np.where(y_train_data == 1)[0])
    class2_indices = np.array(np.where(y_train_data == 2)[0])

    class0_sample = list(np.random.choice(class0_indices, 3553))
    class1_sample = list(np.random.choice(class1_indices, 3553))
    class2_sample = list(np.random.choice(class2_indices, 3553))

    print("sampling done")
    end = time. time()
    print((end-start_time))

    indices = class0_sample + class1_sample + class2_sample

    new_x_train = x_train_data.iloc[indices,:]
    new_y_train = y_train_data.iloc[indices,:]

    print(new_x_train.shape, new_y_train.shape)

    # for i in range(len(indices)):
    #     print(i)
    #     index = indices[i]
    #     new_x_train = np.vstack((new_x_train, x_train_data[index,:]))
    #     new_y_train.append(y_train_data[index])
    
    print("data set prepared")
    end = time. time()
    print((end-start_time))

    x_train_data = pd.DataFrame(new_x_train)
    x_test_data = pd.DataFrame(data=x_test_data)
    y_train_data = pd.DataFrame(data=new_y_train)

    print(np.sum(np.array(y_train_data) == 0), np.sum(np.array(y_train_data) == 1), np.sum(np.array(y_train_data) == 2))

    df_train = pd.DataFrame(data=[])
    df_test = pd.DataFrame(data=[])
    j = 0
    for i in range(x_train_data.shape[1]):
        # if (np.std(np.array(x_train_data.iloc[:,i])) == np.std(np.array(x_train_data.iloc[:,i])) == True) & (np.std(np.array(x_train_data.iloc[:,i])) > 0.00001):
        if ((np.std(np.array(x_train_data.iloc[:,i])) > 0.0001) & (np.std(np.array(x_test_data.iloc[:,i])) > 0.0001)):
            df_train[j] = x_train_data.iloc[:,i]
            df_test[j] = x_test_data.iloc[:,i]
            j += 1

    x_train_data = df_train
    x_test_data = df_test

    print(x_train_data.shape)    
    
    # ## Scale data
    scaler = StandardScaler()  
    scaler.fit(x_train_data)  
    x_train_data = pd.DataFrame(data=scaler.transform(x_train_data))
    x_test_data = pd.DataFrame(data=scaler.transform(x_test_data))

    # return the data in a tuple 
    return(x_train_data, y_train_data, x_test_data, id_data)

X1, y1, DATA_TEST, id_data = load_person_average_data()




   # Import and prepare data
X1.as_matrix().astype(np.float32)
y1.as_matrix().astype(np.int)
DATA_TEST.as_matrix().astype(np.float64)


# Train data
# Define Dnn constants
n_inputs = DATA_TEST.shape[1]
n_outputs = len(np.unique(y1))
n_epochs = 50
batch_size = 500
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

def run_tf(model, n_hidden1, n_hidden2, n_hidden3, activation_fn_layer_1, activation_fn_layer_2, activation_fn_layer_3, dropout_rate, DATA_TEST=DATA_TEST):
    # Set up placeholder for inputs
    X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
    y = tf.placeholder(tf.int64, shape=(None), name = "y")

    keep_prob = tf.placeholder(tf.float32)

    # if doing internal CV, uncomment this section:
    # if model == "train":

    #     BMAC_all = []
    #     kf = StratifiedKFold(n_splits = 10)
    #     kf.get_n_splits(X1, y1)
    #     for train_index, test_index in kf.split(X1,y1):

    #         X_train, X_test = X1.iloc[train_index,:], X1.iloc[test_index,:]
    #         y_train, y_test = y1.iloc[train_index,:], y1.iloc[test_index,:]
    #         X_new = X_test 
        
    #         train = np.column_stack((y_train, X_train)) # put together y_train and X_train
    #         dataset = Dataset(train)  # instantiate Dataset object

       
    # # if using full model, uncomment this section:
    if model == "test":
        X_new = DATA_TEST.as_matrix().astype(np.float64)
        X_train = X1
        y_train = y1
        full_train = np.column_stack((y1, X1))
        dataset = Dataset(full_train)
        for i in range(1):

    # Design Dnn     
            with tf.name_scope("dnn"):
                hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation = activation_fn_layer_1)
                hidden1_drop = tf.nn.dropout(hidden1,keep_prob)
                
                hidden2 = neuron_layer(hidden1_drop, n_hidden2, "hidden2", activation  = activation_fn_layer_2)
                hidden2_drop = tf.nn.dropout(hidden2,keep_prob)
                
                hidden3 = neuron_layer(hidden2_drop, n_hidden3, "hidden3", activation = activation_fn_layer_3)
                hidden3_drop = tf.nn.dropout(hidden3,keep_prob)

                # hidden4 = neuron_layer(hidden3_drop, 500, "hidden3", activation = activation_fn_layer_3)
                # hidden3_drop = tf.nn.dropout(hidden4,keep_prob)
               
                # hidden4 = neuron_layer(hidden3_drop, n_hidden4, "hidden3", activation = "relu")
                # hidden4_drop = tf.nn.dropout(hidden4,keep_prob)

                logits  = neuron_layer(hidden3_drop, n_outputs, "outputs")

            
            # Specify loss function
            with tf.name_scope("loss"):
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                         labels = y, logits = logits)
                loss = tf.reduce_mean(xentropy, name = "loss")
                
            # Specify gradient descent optimizer
            with tf.name_scope("train"):
                optimizer = tf.train.AdamOptimizer(learning_rate, beta1=BETA1, beta2=BETA2)
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
                    
                Z = logits.eval(feed_dict = {X: X_new, keep_prob: 1.0})

                y_pred = np.argmax(Z, axis = 1)
                print(np.sum(np.array(y_pred)==0), np.sum(np.array(y_pred)==1), np.sum(np.array(y_pred)==2))
            
            if model == "train":
                print(n_hidden1, activation_fn_layer_1, n_hidden2, activation_fn_layer_2, n_hidden3, activation_fn_layer_3, dropout_rate)
                BMAC = balanced_accuracy_score(y_test, y_pred)
                
                # print("mean:", np.mean(np.array(BMAC_all), "std:", np.std(np.array(BMAC_all))))
                BMAC_all.append(BMAC)    
                print("BMACall:", BMAC_all)

                # keep track of different combinatoins that were attempted
                # with open("different_attempts.csv", "a+") as myfile:
                #     writer = csv.writer(myfile)
                #     writer.writerow([n_hidden1, activation_fn_layer_1, n_hidden2, activation_fn_layer_2, n_hidden3, activation_fn_layer_3, dropout_rate,f1_result, learning_rate])
                #     # myfile.write()

                end = time. time()
                print((end-start_time))

            else:
                results = np.array(y_pred) + 1
                id_data2 = np.arange(0, len(results))  
                results = pd.DataFrame({'Id': id_data2, "y": results})
                results.to_csv(BASE_PATH + 'task5_dnn.csv', index = False)
            

                                              

run_tf(model="train", 
    n_hidden1=500, 
    n_hidden2=500, 
    n_hidden3=500, 
    activation_fn_layer_1=ACT1, 
    activation_fn_layer_2=ACT2, 
    activation_fn_layer_3=ACT3, 
    dropout_rate=DROPOUT)

