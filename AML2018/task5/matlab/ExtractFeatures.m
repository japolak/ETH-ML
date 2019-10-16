function [ Features,featureindices] = ...
    ExtractFeatures(tData,T,AR_order,level)
% This function is only in support of XpwWaveletMLExample. It may change or
% be removed in a future release.
    Features = [];

    for idx =1:size(tData,1)
        x1 = tData(idx,:);
        x1 = detrend(x1,0);
        arcoefs = blockAR(x1,AR_order,T);
        se = shannonEntropy(x1,T,level);
        [cp,rh] = leaders(x1,T);
        wvar = modwtvar(modwt(x1,'db2'),'db2');
        Features = [Features;arcoefs se cp rh wvar']; %#ok<AGROW>
        idx
    end

end
