clear all
%% Load Data
%  1.5 min
tic
train_eeg1 = csvread('train_eeg1.csv',1,1);
train_eeg2 = csvread('train_eeg2.csv',1,1);
train_emg = csvread('train_emg.csv',1,1);
train_labels = csvread('train_labels.csv',1,1);

test_eeg1 = csvread('test_eeg1.csv',1,1);
test_eeg2 = csvread('test_eeg2.csv',1,1);
test_emg = csvread('test_emg.csv',1,1);
toc

%% Split Dataset into individual Mice
tic
[m1_eeg1,m2_eeg1,m3_eeg1]=mice_split(train_eeg1,3);
[m1_eeg2,m2_eeg2,m3_eeg2]=mice_split(train_eeg2,3);
[m1_emg,m2_emg,m3_emg]=mice_split(train_emg,3);
[m1_labels,m2_labels,m3_labels]=mice_split(train_labels,3);

[m4_eeg1,m5_eeg1]=mice_split(test_eeg1,2);
[m4_eeg2,m5_eeg2]=mice_split(test_eeg2,2);
[m4_emg,m5_emg]=mice_split(test_emg,2);
toc

%% Plot epoch
epoch=1;
data=train_eeg1;
row=train_eeg1(epoch,:)';
plot(1:512,row,'b',[0 512],[0 0],'r','linewidth',1)

% See BasicSpectralAnalysis.m
 
%% Feature extraction
% 22 min
tic
data=train_eeg1;
train_eeg1_features=[ef_time(data),ef_freq(data)];
data=train_eeg2;
train_eeg2_features=[ef_time(data),ef_freq(data)]; 
data=train_emg;
train_emg_features=[ef_time(data),ef_freq(data)];

data=test_eeg1;
test_eeg1_features=[ef_time(data),ef_freq(data)];
data=test_eeg2;
test_eeg2_features=[ef_time(data),ef_freq(data)];
data=test_emg;
test_emg_features=[ef_time(data),ef_freq(data)];
toc

%% Save features

csvwrite('train_eeg1_features.csv',train_eeg1_features);
csvwrite('train_eeg2_features.csv',train_eeg2_features);
csvwrite('train_emg_features.csv',train_emg_features);

csvwrite('test_eeg1_features.csv',test_eeg1_features);
csvwrite('test_eeg2_features.csv',test_eeg2_features);
csvwrite('test_emg_features.csv',test_emg_features);


%% Split Features into individual Mice
tic
[m1_eeg1_features,m2_eeg1_features,m3_eeg1_features]=mice_split(train_eeg1_features,3);
[m1_eeg2_features,m2_eeg2_features,m3_eeg2_features]=mice_split(train_eeg2_features,3);
[m1_emg_features, m2_emg_features, m3_emg_features] =mice_split(train_emg_features,3);
[m1_labels,m2_labels,m3_labels]=mice_split(train_labels,3);

[m4_eeg1_features,m5_eeg1_features]=mice_split(test_eeg1_features,2);
[m4_eeg2_features,m5_eeg2_features]=mice_split(test_eeg2_features,2);
[m4_emg_features,m5_emg_features]=mice_split(test_emg_features,2);
toc


%% Pool features and channels
tic
trainFeatures=[train_eeg1_features,train_eeg2_features,train_emg_features];
trainLabels=train_labels;
testFeatures=[test_eeg1_features,test_eeg2_features,test_emg_features];
toc
%% Select model SVM
tic
rng(1)
template = templateSVM('KernelFunction','polynomial',...
                        'PolynomialOrder',2,...
                        'KernelScale','auto',...
                        'BoxConstraint',1,...
                        'Standardize',true);
model = fitcecoc(trainFeatures,trainLabels,...
                'Learners',template,...
                'Coding','onevsone',...
                'ClassNames',{'1','2','3'});  
toc
%% Select model trees
tic
rng(1)
template = templateEnsemble('AdaBoostM1',100,'tree')
model = fitensamble(trainFeatures,trainLabels,...
                    'Learners',template,...
                    'Type','classification')
toc

%% Cross-Validation  
% SVM
tic
kfoldmodel = crossval(model,'kfold',3);
classLabels = kfoldPredict(kfoldmodel);
loss = kfoldLoss(kfoldmodel)*100;
predictLabels = str2double(classLabels);
[confmatCV,grouporder] = confusionmat(trainLabels,predictLabels);
CVTable = PrecisionRecall(confmatCV);
toc
%%

loss
confmatCV
CVTable

%% Fit full model

predictLabels = predict(model,testFeatures);
id=0:(size(predictLabels,1)-1);
y=str2num(cell2mat(predictLabels));
out=[id' y];

% Save Predictions

csvwrite('task5.csv',out)

%% New features
tic
T = 512
AR_order = 4;
level = 4;
[train_eeg1_features_v2,indecies] = ExtractFeatures(train_eeg1,T,AR_order,level);
toc

%%



TrainData = csvread('X_train.csv',1,1);
TrainLabels = csvread('y_train.csv',1,1);
TestData = csvread('X_test.csv',1,1);

%% Preprocesing 1 ---- Replace zeros till window size
 samples = 8192
 [ TrainDataP ] = ReplaceMissing( TrainData , samples );
 [ TestDataP ] = ReplaceMissing( TestData , samples );
%%

epoch=1;
data=TrainDataP;
row=TrainDataP(epoch,:)';
plot(1:samples,row,'b',[0 8192],[0 0],'r','linewidth',1)

%%

data=TrainDataP;
train_features=[ef_time(data),ef_freq(data)];
data=TestDataP;
test_features=[ef_time(data),ef_freq(data)];

