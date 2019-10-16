tic
train_eeg1_features_v2 = csvread('train_eeg1_features_v2.csv');
train_eeg2_features_v2 = csvread('train_eeg2_features_v2.csv');
train_emg_features_v2 = csvread('train_emg_features_v2.csv');
test_eeg1_features_v2 = csvread('test_eeg1_features_v2.csv');
test_eeg2_features_v2 = csvread('test_eeg2_features_v2.csv');
test_emg_features_v2 = csvread('test_emg_features_v2.csv');
toc
%%
tic
train_eeg1_features_v0 = csvread('train_eeg1_features_v0.csv');
train_eeg2_features_v0 = csvread('train_eeg2_features_v0.csv');
train_emg_features_v0 = csvread('train_emg_features_v0.csv');
test_eeg1_features_v0 = csvread('test_eeg1_features_v0.csv');
test_eeg2_features_v0 = csvread('test_eeg2_features_v0.csv');
test_emg_features_v0 = csvread('test_emg_features_v0.csv');
toc
%%
train_labels = csvread('train_labels.csv',1,1);