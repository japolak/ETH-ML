import os
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statistics import mean, median, variance
from scipy.stats import skew
import biosppy as bsp
from extract_features import *

##########################
# SET THESE VARIABLES
# might also need to change the id column name depending on data used
##########################
path = '/Users/leandereberhard/Desktop/ETH/AML/task3'
X_train_file_name = 'X_train_clean.csv'
y_train_file_name = 'y_train.csv'
X_test_file_name = 'X_test_clean.csv'
train_output_file_name = 'features_train.csv'
test_output_file_name = 'features_test.csv'

train_id_col_name = 'Unnamed: 0'
y_id_col_name = 'id'
test_id_col_name = 'Unnamed: 0'
##########################
##########################


# import the data
os.chdir(path)

X_train = pd.read_csv(X_train_file_name)
y_train = pd.read_csv(y_train_file_name)
X_test = pd.read_csv(X_test_file_name)

test_id = X_test.iloc[:, 0]
train_id = X_train.iloc[:, 0]

# drop index row
X_train = X_train.drop(train_id_col_name, axis=1)
y_train = y_train.drop(y_id_col_name, axis=1)
X_test = X_test.drop(test_id_col_name, axis=1)


# invert rows
ind_to_invert = pd.read_csv('indices_inversion.csv')['x']

# for ind in ind_to_invert:
#     X_train.iloc[ind, :] = [-1 * x for x in X_train.iloc[ind, :]]
#

#######################
#######################
# plotting functions / outlier detector
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
def plot_row(data, row_index):
    # process data
    test_row = data.iloc[row_index, :].dropna()
    processed_signal = bsp.signals.ecg.ecg(signal=test_row, sampling_rate=300, show=False)
    r_peaks = bsp.signals.ecg.christov_segmenter(processed_signal['filtered'], sampling_rate=300)
    r_peaks = bsp.signals.ecg.correct_rpeaks(signal=processed_signal['filtered'], rpeaks=r_peaks['rpeaks'], sampling_rate=300,
                                             tol=0.05)
    average_level = mean(processed_signal['filtered'])
    extracted_peaks = extract_qs_peaks(processed_signal, r_peaks, tolerance=30)
    qrs_extracted = extract_qrs_duration(processed_signal, extracted_peaks, tolerance=30)
    pd.DataFrame(processed_signal['filtered']).plot()

    # plot
    for peak in r_peaks['rpeaks']:
        pyplot.axvline(x=peak, color='red')
    for peak in extracted_peaks['s_peaks']:
        pyplot.axvline(x=peak, color='green')
    for peak in extracted_peaks['q_peaks']:
        pyplot.axvline(x=peak, color='orange')
    for bound in qrs_extracted['lower_bounds']:
        pyplot.axvline(x=bound, color='black')
    for bound in qrs_extracted['upper_bounds']:
        pyplot.axvline(x=bound, color='black')
    pyplot.axhline(y=average_level)
    pyplot.show()

def plot_row_hb(data, plot_index):
    row = data.iloc[plot_index,].dropna()
    processed_signal = bsp.signals.ecg.ecg(signal=row, sampling_rate=300, show=False)
    heartbeats = processed_signal['templates']

    mean_heartbeat = np.mean(heartbeats, axis=0)
    median_level = median(mean_heartbeat)

    qs_extracted_hb = extract_qs_peaks_hb(mean_heartbeat)
    r_peak = np.argmax(mean_heartbeat)
    s_peak = qs_extracted_hb['s_peak']
    q_peak = qs_extracted_hb['q_peak']

    qrs_extracted_hb = extract_qrs_duration_hb(mean_heartbeat, qs_extracted_hb)
    lower_bound_hb = qrs_extracted_hb['lower_bound']
    upper_bound_hb = qrs_extracted_hb['upper_bound']

    t_extracted_hb = extract_t_peak_hb(mean_heartbeat, qrs_extracted_hb)['t_peak']

    pd.DataFrame(mean_heartbeat).plot()
    pyplot.axvline(x=r_peak, color='red')
    pyplot.axvline(x=s_peak, color='green')
    pyplot.axvline(x=q_peak, color='orange')
    pyplot.axvline(x=lower_bound_hb, color='black')
    pyplot.axvline(x=upper_bound_hb, color='black')
    pyplot.axvline(x=t_extracted_hb, color='gray')
    pyplot.axhline(y=median_level)
    legend = pyplot.legend()
    legend.get_texts()[0].set_text(y_train.iloc[plot_index, 1])
    pyplot.show()
#######################
#######################


#######################
# extract features
# specify data from which to generate features (X_train or X_test) and file name
# e.g. generate_features(X_train, "test_file")
#######################
def generate_features(data, file_name):
    # initialize data frame
    columns = ['median_qrs_dur', 'var_qrs_dur', 'min_qrs_dur', 'max_qrs_dur', 'skew_qrs_dur', 'median_q_amp',
               'var_q_amp', 'min_q_amp', 'max_q_amp', 'skew_q_amp', 't_amp_average_hb']

    extracted_features = pd.DataFrame(index=train_id, columns=columns)
    extracted_features = extracted_features.fillna(0)

    # rows with errors will be stored here
    bad_rows = []

    for index, row in data.iterrows():
        row = row.dropna()

        # cut off peaks
        top_cut = np.quantile(row, .95)
        bottom_cut = np.quantile(row, .05)
        row = [min(max(i, bottom_cut), top_cut) for i in row]

        try:
            # process raw signal
            processed_signal = bsp.signals.ecg.ecg(signal=row, sampling_rate=300, show=False)
            # extract the peaks
            r_peaks = bsp.signals.ecg.christov_segmenter(processed_signal['filtered'], sampling_rate=300)
            # correct the peaks
            r_peaks = bsp.signals.ecg.correct_rpeaks(signal=processed_signal['filtered'], rpeaks=r_peaks['rpeaks'],
                                                     sampling_rate=300, tol=0.05)

            # extract QRS, Q and T information
            extracted_peaks = extract_qs_peaks(processed_signal, r_peaks, tolerance=30)
            qrs_extracted = extract_qrs_duration(processed_signal, extracted_peaks, tolerance=30)

            filtered = processed_signal['filtered']
            median_level = median(filtered)

            # heartbeat information
            heartbeats = processed_signal['templates']
            mean_heartbeat = np.mean(heartbeats, axis=0)
            qs_extracted_hb = extract_qs_peaks_hb(mean_heartbeat)
            r_peak = np.argmax(mean_heartbeat)
            q_peak = qs_extracted_hb['q_peak']
            q_peak = qs_extracted_hb['q_peak']
            qrs_extracted_hb = extract_qrs_duration_hb(mean_heartbeat, qs_extracted_hb)
            lower_bound_hb = qrs_extracted_hb['lower_bound']
            upper_bound_hb = qrs_extracted_hb['upper_bound']

            # write extracted features to data frame
            # QRS features
            qrs_durations = qrs_extracted['qrs_durations']
            extracted_features.loc[index, 'median_qrs_dur'] = median(qrs_durations)
            extracted_features.loc[index, 'var_qrs_dur'] = variance(qrs_durations)
            extracted_features.loc[index, 'min_qrs_dur'] = min(qrs_durations)
            extracted_features.loc[index, 'max_qrs_dur'] = max(qrs_durations)
            extracted_features.loc[index, 'skew_qrs_dur'] = skew(qrs_durations)

            # Q peak features
            q_amplitudes = median_level + [filtered[i] for i in extracted_peaks['q_peaks']]
            extracted_features.loc[index, 'median_q_amp'] = median(q_amplitudes)
            extracted_features.loc[index, 'var_q_amp'] = variance(q_amplitudes)
            extracted_features.loc[index, 'min_q_amp'] = min(q_amplitudes)
            extracted_features.loc[index, 'max_q_amp'] = max(q_amplitudes)
            extracted_features.loc[index, 'skew_q_amp'] = skew(q_amplitudes)

            # T peak feature
            t_peak = extract_t_peak_hb(mean_heartbeat, qrs_extracted_hb)['t_peak']
            extracted_features.loc[index, 't_amp_average_hb'] = median_level + mean_heartbeat[t_peak]

            print("Done " + str(index))

        except ValueError:
            print("Error: Check row " + str(index))
            bad_rows.append(index)

    # write extracted features to csv file
    extracted_features.to_csv(file_name)

    print(str(len(bad_rows)) + " bad rows, check indices:")
    print(bad_rows)


generate_features(X_train, train_output_file_name)
generate_features(X_test, test_output_file_name)
