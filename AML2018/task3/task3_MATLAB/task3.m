
%% Load Data ----
 TrainData = csvread('X_train.csv',1,1);
 TrainLabels = csvread('y_train.csv',1,1);
 TestData = csvread('X_test.csv',1,1);

%% Length of time data and window
samples = 8192;
%% AR order
order = 4;
%% MODWPT level
level = 4;
%% Preprocesing 1 ---- Replace zeros till window size

 [ TrainDataP ] = ReplaceMissing( TrainData , samples );
 [ TestDataP ] = ReplaceMissing( TestData , samples );

%% Preprocessing 2 ----- Split into trn and val sets

 ECG = struct('TrainData',TrainData,...
             'TestData',TestData,...
              'TrainLabels',TrainLabels);
              
 [ trnData,valData,trnLabels,valLabels] = RandomSplit( ECG );
 
%% Extract Features -----
 
window = round(size(TrainDataP,2)/8);

 [trnFeatures,featureindices] = ExtractFeatures(trnData,order,level);
 [valFeatures,featureindices] = ExtractFeatures(valData,order,level);
 %[TrainFeatures,featureindices] = ExtractFeatures(ECG.TrainData,window,order,level);
 %[TestFeatures,featureindices] = ExtractFeatures(ECG.TestData,window,order,level);

%% Save Features ----
% csvwrite('X_train_features.csv',TrainFeatures)
% csvwrite('X_test_features.csv',TestFeatures)


%% Define Model

rng(1)
template = templateSVM('KernelFunction','gaussian',...
                        'KernelScale','auto',...
                        'Standardize',true);

            
                    
                    
                    
%% trn,val Fit Model
features = [trnFeatures; valFeatures];
labels = [trnLabels; valLabels];
  
%features = TrainFeatures;
%labels = TrainLabels;

model = fitcecoc(features,labels,...
                'Learners',template,...
                'Coding','onevsone',...
                'ClassNames',{'0','1','2','3'});        

%% trn,val CrossValidation and Prediction

kfoldmodel = crossval(model,'KFold',5);
classLabels = kfoldPredict(kfoldmodel);

%% trn,val Model Evaluation
            
loss = kfoldLoss(kfoldmodel)*100;
classLabels = str2double(classLabels);
[confmatCV,grouporder] = confusionmat(labels,classLabels);

100-loss
confmatCV

%% Fit Full Model and Generate Prediction


model = fitcecoc(TrainFeatures,TrainLabels,...
                'Learners',template,...
                'Coding','onevsone',...
                'ClassNames',{'0','1','2','3'});
predictLabels = predict(model,TestFeatures);
a=0:(size(predictLabels,1)-1);
b=str2num(cell2mat(predictLabels));
c=[a' b];

% Save Predictions

csvwrite('task3.csv',c)










