function [ meanFrequency, medianFrequency ] = mm_frequency( data )
%Mean and Median frequency of a function
outmeanf=[];
outmedf=[];

for i=1:length(data)
    row=data(i,:)';
    meanf=meanfreq(row);
    medf=medfreq(row);
    outmeanf=[outmeanf;meanf];
    outmedf=[outmedf;medf];
end
meanFrequency=double(outmeanf);
medianFrequency=double(outmedf);  
end

