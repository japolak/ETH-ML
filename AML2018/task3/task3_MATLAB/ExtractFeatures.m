function [Features,featureindices] = ExtractFeatures(Data,order,level)
% This function is only in support of XpwWaveletMLExample. It may change or
% be removed in a future release.
Features = [];
%Data=valData;

for idx =1:size(Data,1)
    x = Data(idx,:);
    x = detrend(x,0);
    window = round(size(x,2)/8);
    arcoefs = EF_blockAR(x,order,window);
    se = EF_shannonEntropy(x,window,level);
    [cp,rh] = EF_leaders(x,window);
    wvar = modwtvar(modwt(x,'db2'),'db2');
    Features = [Features; arcoefs se cp rh wvar']; %#ok<AGROW>

end

featureindices = struct();

featureindices.ARfeatures = 1:32;
startidx = 33;
endidx = 33+(16*8)-1;
featureindices.SEfeatures = startidx:endidx;
startidx = endidx+1;
endidx = startidx+7;
featureindices.CP2features = startidx:endidx;
startidx = endidx+1;
endidx = startidx+7;
featureindices.HRfeatures = startidx:endidx;
startidx = endidx+1;
endidx = startidx+13;
featureindices.WVARfeatures = startidx:endidx;
end
