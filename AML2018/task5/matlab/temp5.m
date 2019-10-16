tic
predictLabels = predict(model,testFeatures);
id=0:(size(predictLabels,1)-1);
y=str2num(cell2mat(predictLabels));
out=[id' y];

% Save Predictions

csvwrite('task5_v0v2_SVM_P2_ovo.csv',out)
toc