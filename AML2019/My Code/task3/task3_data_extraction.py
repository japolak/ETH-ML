import os
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statistics import mean, median, variance
from scipy.stats import skew
import biosppy as bsp
from ecgdetectors import Detectors
from extract_features import *

# import the data
path = '/Users/leandereberhard/Desktop/ETH/AML/task3'
os.chdir(path)

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')

test_id = X_test.iloc[:, 0]
train_id = X_train.iloc[:, 0]

X_train = X_train.drop('id', axis=1)
y_train = y_train.drop('id', axis=1)
X_test = X_test.drop('id', axis=1)



# # plot the data
# test_row.plot()
# pyplot.show()
#
# # detect the R peaks
#
# # initialize detector with the sample rate of the ECG
# detector = Detectors(300)
# # get r peaks using Pan-Tompkins, other algorithms possible for this
# r_peaks = detector.pan_tompkins_detector(test_row)
# # two average detector
# r_peaks = detector.two_average_detector(test_row)
# # stationary wavelet detector
# r_peaks = detector.swt_detector(test_row)
#
# # r peak detection using biosppy package
# # first process the raw ecg signal
# processed_row = bsp.signals.ecg.ecg(signal=test_row, sampling_rate=100, show=True)
# # plot processed row
# pd.DataFrame(processed_row['filtered']).plot()
# pyplot.show()


def var(lst):
    if len(lst) > 1:
        return variance(lst)
    else:
        return 0
def mn(lst):
    if len(lst) >= 1:
        return mean(lst)
    else:
        return 0
def md(lst):
    if len(lst) >= 1:
        return median(lst)
    else:
        return 0

def list_outliers(data_frame):
    outliers_ind = []

    for index, row in data_frame.iterrows():
        try:
            # process the row as usual
            row = row.dropna()
            processed_signal = bsp.signals.ecg.ecg(signal=row, sampling_rate=300, show=False)
            # extract the peaks
            r_peaks = bsp.signals.ecg.christov_segmenter(processed_signal['filtered'], sampling_rate=300)
            # correct the peaks
            r_peaks = bsp.signals.ecg.correct_rpeaks(signal=processed_signal['filtered'], rpeaks=r_peaks['rpeaks'],
                                                     sampling_rate=300,
                                                     tol=0.05)
            # probably shitty data if there is only one r peak
            if len(r_peaks[0]) <= 1:
                outliers_ind.append(index)
        except ValueError:
            print("Failed at index " + str(index))

    return outliers_ind

# plot a row; specify the row
def plot_row(row_index):
    # process data
    test_row = X_train.iloc[row_index, :].dropna()
    processed_signal = bsp.signals.ecg.ecg(signal=test_row, sampling_rate=300, show=False)
    # R peaks
    r_peaks = bsp.signals.ecg.christov_segmenter(processed_signal['filtered'], sampling_rate=300)
    r_peaks = bsp.signals.ecg.correct_rpeaks(signal=processed_signal['filtered'], rpeaks=r_peaks['rpeaks'], sampling_rate=300, tol=0.05)
    # QRS peaks
    extracted_peaks = extract_qs_peaks(processed_signal, r_peaks, tolerance=30)
    qrs_extracted = extract_qrs_duration(processed_signal, extracted_peaks, tolerance=30)
    # plot
    pd.DataFrame(processed_signal['filtered']).plot()
    for peak in r_peaks['rpeaks']: pyplot.axvline(x=peak, color='red')
    #for peak in extracted_peaks['s_peaks']: pyplot.axvline(x=peak, color='green')
    #for peak in extracted_peaks['q_peaks']: pyplot.axvline(x=peak, color='orange')
    #for bound in qrs_extracted['lower_bounds']: pyplot.axvline(x=bound, color='black')
    #for bound in qrs_extracted['upper_bounds']: pyplot.axvline(x=bound, color='black')
    pyplot.axhline(y=mean(processed_signal['filtered']))
    pyplot.show()



# try extracting the data from a test row
look_at = 45
test_row = X_train.iloc[look_at, :].dropna()
y_train.iloc[look_at]

start_ind = 10
test_row = X_train.iloc[start_ind, :].dropna()

# test the functions
# process signal using biosppy
processed_signal = bsp.signals.ecg.ecg(signal=test_row, sampling_rate=300, show=False)
# extract the peaks
r_peaks = bsp.signals.ecg.christov_segmenter(processed_signal['filtered'], sampling_rate=300)
# correct the peaks
r_peaks = bsp.signals.ecg.correct_rpeaks(signal=processed_signal['filtered'], rpeaks=r_peaks['rpeaks'], sampling_rate=300, tol=0.05)

# extract Q,S peaks and QRS duration
extracted_peaks = extract_qs_peaks(processed_signal, r_peaks, tolerance=30)
qrs_extracted = extract_qrs_duration(processed_signal, extracted_peaks, tolerance=30)

# try to extract some features from this row
filtered = processed_signal['filtered']
average_level = mean(filtered)

# list all the outliers (rows where only a single r peak was detected)
bad_indices = list_outliers(X_train)

# figure out where the loop got stuck
next(np.where(extracted_features == x)[0][0] for x in extracted_features['average_level'] if x == 0)

start_ind = 1760
X_train_sub = X_train.iloc[start_ind:len(train_id), ]

# check where to make the cut
plot_row(start_ind)

plot_row(10)

# extract the features for each row
# initialize data frame
columns = ['average_level', 'mean_r_amp', 'median_r_amp', 'var_r_amp', 'mean_q_amp', 'median_q_amp', 'var_q_amp',
           'mean_s_amp', 'median_s_amp', 'var_s_amp', 'mean_qrs_dur', 'median_qrs_dur', 'var_qrs_dur']

extracted_features = pd.DataFrame(index=train_id, columns=columns)
extracted_features = extracted_features.fillna(0)

# Format – row_index: where_to_start_series
nasty_train = {83: 1000, 126: 1000, 129: 1140, 235: 1900, 260: 1600, 330: 1500, 333: 500, 429: 200, 528: 1000, 530: 600,
               595: 1300, 715: 1900, 726: 250, 809: 200, 957: 400, 958: 1700, 1001: 1700, 1044: 1000, 1101: 1800,
               1114: 1700, 1138: 1700, 1271: 2500, 1306: 900, 1307: 1000, 1431: 1900, 1474: 500, 1477: 400, 1759: 1300,
               1760: 500, 1796: 700, 1838: 700, 1923: 1000, 1952: 500, 1956: 1400, 2168: 1500, 2292: 2700, 2328: 700,
               2771: 1200, 2773: 900, 2866: 1100, 3024: 500, 3150: 1200, 3310: 1500, 3388: 800, 3424: 200,
               3679: 1600, 3782: 800, 3784: 300, 3845: 1300, 3871: 300, 3905: 1200, 4076: 800, 4104: 700, 4163: 500,
               4186: 1100, 4253: 1200, 4400: 200, 4480: 1300, 4569: 1100, 4590: 800, 4663: 800, 4851: 200, 5029: 800,
               5036: 1500, 5041: 500, 5073: 1500}

nasty_test = {154: 600, 163: 300, 254: 1200, 290: 2000, 340: 2100, 424: 700, 437: 900, 515: 300, 532: 1500,
              629: 2200, 634: 200, 845: 1100, 1056: 500, 1107: 1100, 1114: 650, 1117: 1400, 1225: 2900, 1236: 1300,
              1355: 300, 1398: 700, 1401: 1100, 1418: 1200, 1580: 1100, 1655: 200, 1664: 1200, 1671: 600,
              1770: 2700, 2029: 1400, 2032: 400, 2127: 1500, 2230: 1300, 2327: 1200, 2331: 800, 2333: 1200,
              2351: 600, 2412: 1000, 2513: 7000, 2550: 1100, 2846: 1000, 2948: 700, 2982: 2300, 3149: 1500,
              3168: 1500, 3244: 300, 3345: 300}

for index, row in X_train_sub.iterrows():
    row = row.dropna()

    # catch exceptions
    if index in nasty_data.keys():
        row = row[nasty_data[index]:len(row)]

    try:
        # process raw signal
        processed_signal = bsp.signals.ecg.ecg(signal=row, sampling_rate=300, show=False)
        # extract the peaks
        r_peaks = bsp.signals.ecg.christov_segmenter(processed_signal['filtered'], sampling_rate=300)
        # correct the peaks
        r_peaks = bsp.signals.ecg.correct_rpeaks(signal=processed_signal['filtered'], rpeaks=r_peaks['rpeaks'],
                                                 sampling_rate=300,
                                                 tol=0.05)

        # extract Q,S peaks and QRS duration
        extracted_peaks = extract_qs_peaks(processed_signal, r_peaks, tolerance=30)
        qrs_extracted = extract_qrs_duration(processed_signal, extracted_peaks, tolerance=30)

        # try to extract some features from this row
        filtered = processed_signal['filtered']
        average_level = mean(filtered)
        extracted_features.loc[index, 'average_level'] = average_level

        # r amplitude
        r_amplitudes = [filtered[i] - average_level for i in r_peaks['rpeaks']]
        extracted_features.loc[index, 'mean_r_amp'] = mn(r_amplitudes)
        extracted_features.loc[index, 'median_r_amp'] = md(r_amplitudes)
        extracted_features.loc[index, 'var_r_amp'] = var(r_amplitudes)

        # q amplitude
        q_amplitudes = [filtered[i] - average_level for i in extracted_peaks['q_peaks']]
        extracted_features.loc[index, 'mean_q_amp'] = mn(q_amplitudes)
        extracted_features.loc[index, 'median_q_amp'] = md(q_amplitudes)
        extracted_features.loc[index, 'var_q_amp'] = var(q_amplitudes)

        # s amplitude
        s_amplitudes = [filtered[i] - average_level for i in extracted_peaks['s_peaks']]
        extracted_features.loc[index, 'mean_s_amp'] = mn(s_amplitudes)
        extracted_features.loc[index, 'median_s_amp'] = md(s_amplitudes)
        extracted_features.loc[index, 'var_s_amp'] = var(s_amplitudes)

        # qrs duration
        qrs_durations = qrs_extracted['qrs_durations']
        extracted_features.loc[index, 'mean_qrs_dur'] = mn(qrs_durations)
        extracted_features.loc[index, 'median_qrs_dur'] = md(qrs_durations)
        extracted_features.loc[index, 'var_qrs_dur'] = var(qrs_durations)
    except ValueError:
        print("Error: Check row " + str(index))


####################
# write extracted features to csv file
extracted_features.to_csv('extracted_qrs_features.csv')


# given the r peaks, can now extract heartbeats
heartbeats = bsp.signals.ecg.extract_heartbeats(signal=processed_signal['filtered'], rpeaks=r_peaks['rpeaks'],
                                                sampling_rate=300)

# plot the peaks on the raw ECG
pd.DataFrame(test_row).plot()
for peak in r_peaks:
    pyplot.axvline(x=peak, color='red')
pyplot.show()

pyplot.axvline()

# plot an individual template
index = 0

pd.DataFrame(heartbeats['templates'][index]).plot()
pyplot.axvline(x=heartbeats['rpeaks'][index] - sum([len(template)
                                                    for template in heartbeats['templates'][0:index]]))
pyplot.show()

len(heartbeats['templates'][0:2])

####################
# predict only the majority class

test_id
pred = [2 for i in range(len(test_id))]

out = pd.DataFrame({'id': test_id, 'y': pred})
out.to_csv('predict_all_2.csv', index=False)
