function numZeroCrossing = zero_crossings(data)
% Count the number of zero-crossings from the supplied matrix for each row
zcd = dsp.ZeroCrossingDetector;
out=[];
for i=1:length(data)
    cross=step(zcd,data(i,:)');
    out=[out; cross];
end
numZeroCrossing=double(out);
    
end