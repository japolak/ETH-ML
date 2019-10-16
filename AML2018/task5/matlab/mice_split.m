function [mouse1,mouse2,mouse3] = mice_split(data,mice)
%MICE_SPLIT split data depending on the number of mice it's consisting of
    %%
    m=21600;
    mouse1= data(1:m,:);
    mouse2 = data(m+1:2*m,:);
    if mice==3
        mouse3 = data(2*m+1:3*m,:);
    end


end

