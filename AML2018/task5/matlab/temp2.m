%% Pool features and channels
tic
trainFeatures=[train_eeg1_features_v0,train_eeg2_features_v0,train_emg_features_v0,...
               train_eeg1_features_v2,train_eeg2_features_v2,train_emg_features_v2];
trainLabels=train_labels;
testFeatures=[test_eeg1_features_v0,test_eeg2_features_v0,test_emg_features_v0,...
               test_eeg1_features_v2,test_eeg2_features_v2,test_emg_features_v2];
toc