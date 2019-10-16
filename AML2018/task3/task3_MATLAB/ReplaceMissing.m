function [ DataP ] = ReplaceMissing( Data , samples )
%REPLACEMISSING Summary of this function goes here
%   Detailed explanation goes here

y=[]; x=[]; x=Data; y=x;
for i=1:size(x,1)
    if find(x(i,:),1,'last')<=samples
        temp = [];
        temp = repmat(x(i,1:find(x(i,:),1,'last'))',10,1)';
        y(i,:)=temp(:,1:size(x,2));
    end  
    
end
DataP=y(:,1:samples);


end

