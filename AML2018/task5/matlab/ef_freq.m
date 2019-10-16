function [features_freq] = ef_freq(data)
%Extract frequency-domain features

out=[];
ratio=0;                            % 0 - do NOT compute ratios; 1 - compute ratios



for i=1:length(data)                % Loop through epochs
 
    x=data(i,:)';                   % Set current epoch to x - time series signal vector
    n = length(x);                  %Number of samples
    fs = 128;                       %Sample frequency (samples per unit time or space)
    
    y = fft(x - mean(x),n);         % Fast Fourier Transform
    ampl = abs(y);                  % Amplitude of spectrum density
    ampl = ampl(1:1+n/2);           % Half-spectrum
    power = abs(y.^2);              % Raw power spectrum density        (NOTE: without dividing by n)
    power = power(1:1+n/2);         % Half-spectrum                     (NOTE: for conveniece working with 257, ie end boundary)
    [v,k] = max(power);             % Find maximum
    f = transpose((0:n/2)* fs/n);   % Frequency scale

    domFreq = f(k);                                     % Dominant frequency estimate
    avgFreq = sum(power.*f)/sum(power);                 % Average frequency
    coeff=polyfit(log(f(2:end)),log(power(2:end)),1);   % Linear fit of PSD in loglog graph
    factExp = coeff(1);                                 % Factoral Exponent - Slope of linear fit of PSD in loglog graph
    integ = cumtrapz(f,power);                          % Numeric integration
    SEF = interp1(integ, f, 0.95*integ(end), 'linear'); % Interpolation to find SEF = Spectral Edge Density - intensity below which 95% of the total power of the signal is located

    SMzc=sqrt(sum((f.^2).*power)/sum(power))/pi;        % Spectral Moment - zero crossings per sec
    SMex=sqrt(sum((f.^2).*power))/pi;                   % Spectral Moment - extrema per sec

    npower=power/sum(power);                            % Normalised power spectrum
    sEnt = -sum(npower.*log2(npower));                  % Spectral Entropy
    rEnt = -log2(sum(power.^2));                        % Renyi Entropy

    bandpower = sum(power(2:161));                      % Total power in .25 - 40 Hz
    wDelta = sum(power(2:17));                          % .25 - 4 Hz
    wTheta = sum(power(18:33));                         % 4 - 8 Hz
    wAlpha = sum(power(34:49));                         % 8 - 12 Hz
    wSigma = sum(power(50:65));                         % 12 - 16 Hz
    wBeta = sum(power(66:121));                         % 16 - 30 Hz
    wGamma = sum(power(122:257));                       % 30+ Hz

    if ratio==1 
        rDT=wDelta/wTheta;                              % Power ratios
        rDA=wDelta/wAlpha;
        rDS=wDelta/wSigma;
        rDB=wDelta/wBeta;
        rDG=wDelta/wGamma;
        rTA=wTheta/wAlpha;
        rTS=wTheta/wSigma;
        rTB=wTheta/wBeta;
        rTG=wTheta/wGamma;
        rAS=wAlpha/wSigma;
        rAB=wAlpha/wBeta;
        rAG=wAlpha/wGamma;
        rSB=wSigma/wBeta;
        rSG=wSigma/wGamma;
        rBG=wBeta/wGamma;
        
        out=[out; domFreq,avgFreq,factExp,SEF,bandpower,SMzc,SMex,sEnt,rEnt,wDelta,wTheta,wAlpha,wSigma,wBeta,wGamma,rDT,rDA,rDS,rDB,rDG,rTA,rTS,rTB,rTG,rAS,rAB,rAG,rSB,rSG,rBG];
    else
        out=[out; domFreq,avgFreq,factExp,SEF,bandpower,SMzc,SMex,sEnt,rEnt,wDelta,wTheta,wAlpha,wSigma,wBeta,wGamma];
    end

end

features_freq=double(out);


end

