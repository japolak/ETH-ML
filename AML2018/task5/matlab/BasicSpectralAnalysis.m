%% Basic Spectral Analysis

epoch=10;                   %Selected epoch
x=mouse1_eeg1(epoch,:)';    %Sampled data / time series signal
n = length(x);              %Number of samples
fs = 128;                   %Sample frequency (samples per unit time or space)
dt = 1/fs;                  %Time or space increment per sample
t = (0:n-1)/fs;             %Time or space range for data
y = fft(x);                 %Discrete Fourier transform of data (DFT)
ampl = abs(y);              %Amplitude of the DFT
power = (abs(y).^2)/n;      %Power of the DFT
fs/n;                       %Frequency increment
f = (0:n-1)*(fs/n);         %Frequency range
fn = fs/2;                  %Nyquist frequency (midpoint of frequency range)

periodogram(x,t,n,fs)
[pxx,fxx]=periodogram(x,t,n,fs);

%% Power Spectral Density plot

y0=y(1:n/2);                %Half-spectrum y values, since it's symmetrical
f0=f(1:n/2);                %Half-spectrum frequency rance
power0=power(1:n/2);        %Half-spectrum power

plot(f0,power0)
xlabel('Frequency')
ylabel('Power')
axis('tight'),grid('on'),title('Power Spectral Density')


%% Frequency Features

epoch=10;                       %Selected epoch
x=mouse1_eeg1(epoch,:)';        %Sampled data / time series signal
y = fft(x - mean(x),n);         % Fast Fourier Transform
ampl = abs(y);                  % Amplitude of spectrum density
ampl = ampl(1:1+n/2);           % Half-spectrum
power = abs(y.^2);              % Raw power spectrum density        (NOTE: without dividing by n)
power = power(1:1+n/2);         % Half-spectrum                     (NOTE: for conveniece working with 257, ie end boundary)
[v,k] = max(power);             % Find maximum
f = transpose((0:n/2)* fs/n);   % Frequency scale

dominantFreq = f(k)                                 % Dominant frequency estimate
averageFreq = sum(power.*f)/sum(power)              % Average frequency
coeff=polyfit(log(f(2:end)),log(power(2:end)),1);   % Linear fit of PSD in loglog graph
factoralExponent = coeff(1)                         % Slope of linear fit of PSD in loglog graph
integ = cumtrapz(f,power);                          % Numeric integration
SEF = interp1(integ, f, 0.95*integ(end), 'linear')  % Interpolation to find SEF = Spectral Edge Density - intensity below which 95% of the total power of the signal is located

bandpower = sum(power(2:161))                       % Total power in .25 - 40 Hz
wDelta = sum(power(2:17))                           % .25 - 4 Hz
wTheta = sum(power(18:33))                          % 4 - 8 Hz
wAlpha = sum(power(34:49))                          % 8 - 12 Hz
wSigma = sum(power(50:65))                          % 12 - 16 Hz
wBeta = sum(power(66:121))                          % 16 - 30 Hz
wGamma = sum(power(122:257))                        % 30+ Hz

SMzc=sqrt(sum((f.^2).*power)/sum(power))/pi         % Spectral Moment - zero crossings per sec
SMex=sqrt(sum((f.^2).*power))/pi                    % Spectral Moment - extrema per sec

npower=power/sum(power);                            % Normalised power spectrum
spectralEnt = -sum(npower.*log2(npower))            % Spectral Entropy
renyiEnt = -log2(sum(power.^2))                     % Renyi Entropy

ratioDT=wDelta/wTheta;                              % Power ratios
ratioDA=wDelta/wAlpha;
ratioDS=wDelta/wSigma;
ratioDB=wDelta/wBeta;
ratioDG=wDelta/wGamma;
ratioTA=wTheta/wAlpha;
ratioTS=wTheta/wSigma;
ratioTB=wTheta/wBeta;
ratioTG=wTheta/wGamma;
ratioAS=wAlpha/wSigma;
ratioAB=wAlpha/wBeta;
ratioAG=wAlpha/wGamma;
ratioSB=wSigma/wBeta;
ratioSG=wSigma/wGamma;
ratioBG=wBeta/wGamma;

% plot(f, power)





