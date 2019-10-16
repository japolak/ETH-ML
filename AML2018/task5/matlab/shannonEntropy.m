function se = shannonEntropy(x,numbuffer,level)
    numwindows = round(numel(x)/numbuffer);
    y = buffer(x,numbuffer);
    se = zeros(2^level,size(y,2));
    for kk = 1:size(y,2)
        wpt = modwpt(y(:,kk),level);
        % Sum across time
        E = sum(wpt.^2,2);
        Pij = wpt.^2./E;
        % The following is eps(1)
        se(:,kk) = -sum(Pij.*log(Pij+eps),2);
    end

    se = reshape(se,2^level*numwindows,1);
    se = se';
end