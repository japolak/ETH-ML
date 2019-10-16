tic
rng(1)
template = templateSVM('KernelFunction','polynomial',...
                        'PolynomialOrder',2,...
                        'KernelScale','auto',...
                        'BoxConstraint',1,...
                        'Standardize',true);
toc             
tic
model = fitcecoc(trainFeatures,trainLabels,...
                'Learners',template,...
                'Coding','onevsone',...
                'ClassNames',{'1','2','3'});  
toc
tic
kfoldmodel = crossval(model,'kfold',3);
classLabels = kfoldPredict(kfoldmodel);
loss = kfoldLoss(kfoldmodel)*100;
predictLabels = str2double(classLabels);
[confmatCV,grouporder] = confusionmat(trainLabels,predictLabels);
CVTable = PrecisionRecall(confmatCV);
toc
loss
confmatCV
CVTable
