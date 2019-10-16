function [ trnData,valData,trnLabels,valLabels] = RandomSplit( ECG )
%This function splits the data into train and test randomly
percent_train = 70;
split=round(size(ECG.TrainLabels,1)*percent_train/100);
index = randperm(size(ECG.TrainData,1));
trnData = ECG.TrainData(index(1:split),:);
trnLabels = ECG.TrainLabels(index(1:split),:);
valData = ECG.TrainData(index((split+1:end)),:);
valLabels = ECG.TrainLabels(index((split+1:end)),:);
end

