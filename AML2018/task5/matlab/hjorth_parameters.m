function [mobility,complexity] = hjorth_parameters(data)
% Hjorth parameters. Function computes the Hjorth mobility and complexity.
outmob=[];
outcom=[];
for i=1:length(data)
    xV=data(i,:)';
    n = length(xV);
    dxV = diff([0;xV]);
    ddxV = diff([0;dxV]);
    mx2 = mean(xV.^2);
    mdx2 = mean(dxV.^2);
    mddx2 = mean(ddxV.^2);

    mob = sqrt(mdx2 / mx2);
    com = sqrt(mddx2 / mdx2 - mdx2 / mx2);
    
    outmob=[outmob; mob];
    outcom=[outcom; com];
end
 mobility = double(outmob);
 complexity = double(outcom);
 
end