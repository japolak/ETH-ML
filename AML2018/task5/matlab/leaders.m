function [cp,rh] = leaders(x,numbuffer)
    y = buffer(x,numbuffer);
    cp = zeros(1,size(y,2));
    rh = zeros(1,size(y,2));
    for kk = 1:size(y,2)
        [~,h,cptmp] = dwtleader(y(:,kk));
        cp(kk) = cptmp(2);
        rh(kk) = range(h);  
    end
end