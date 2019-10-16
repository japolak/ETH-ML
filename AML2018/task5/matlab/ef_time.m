function [features_time] = ef_time(data)
%Extract time-domain features
f_mean=mean(data,2);                            % Mean
f_med=median(data,2);                           % Median
f_var=var(data,0,2);                            % Variance
f_max=max(data,[],2);                           % Maximum
f_kurt=kurtosis(data,1,2);                      % Kurtosis
f_skew=skewness(data,1,2);                      % Skewness
f_zc=zero_crossings(data);                      % Zero-Crosings
[f_mob,f_comp] = hjorth_parameters(data);       % Hjorth parameteres - mobility and complexity
[f_meanfreq,f_medfreq]=mm_frequency(data);      % Mean and median frequency

features_time=[f_mean , f_med, f_var, f_max, f_kurt, f_skew, f_zc, f_mob, f_comp, f_meanfreq, f_medfreq];
end

