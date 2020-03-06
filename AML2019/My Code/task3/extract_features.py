import numpy as np
from statistics import mean, median

#############################
# input filtered signal from output of biosppy.signals.ecg.ecg() and extracted r_peaks from biosppy.signals.ecg.correct_rpeaks
# tolerance sets the range of indices to search for the peak
# returns dictionary containing lists of indices of 's_peaks' and 'q_peaks'
#############################
def extract_qs_peaks(processed_signal, r_peaks, tolerance=20):
    filtered = processed_signal['filtered']
    s_peaks = [0 for i in range(len(r_peaks[0]))]
    q_peaks = [0 for i in range(len(r_peaks[0]))]

    for i in range(len(r_peaks[0])):
        # current r peak
        current_r_peak = r_peaks[0][i]

        # save detected s peak
        upper_index = min(current_r_peak + tolerance, len(filtered))
        s_peak = current_r_peak + np.argmin(filtered[current_r_peak:upper_index])
        s_peaks[i] = s_peak

        # save detected q peak
        lower_index = max(current_r_peak - tolerance, 0)
        q_peak = current_r_peak - tolerance + np.argmin(filtered[lower_index:current_r_peak])
        q_peaks[i] = q_peak

    return {'s_peaks': s_peaks, 'q_peaks': q_peaks}


#############################
# given q and s peaks, find the QRS duration
# input q and s peaks from output of extract_qs_peaks and processed signal from biosppy.signals.ecg.ecg()
# tolerance sets the range of indices to search for the point at which the signal returns to its mean
# returns a dictionary of 'qrs_duration' in seconds, and indices 'lower_bound' and 'upper_bound'
#############################
def extract_qrs_duration(processed_signal, qs_data, tolerance=20, frequency=300):
    s_peaks = qs_data['s_peaks']
    q_peaks = qs_data['q_peaks']
    filtered = processed_signal['filtered']

    # baseline level of the signal
    average_level = median(filtered)

    # initialization of output lists
    qrs_durations = [0 for i in range(len(s_peaks))]
    lower_bounds = [0 for i in range(len(s_peaks))]
    upper_bounds = [0 for i in range(len(s_peaks))]

    for i in range(len(s_peaks)):
        # lower range to search over
        lower_index = max(0, q_peaks[i] - tolerance)
        search_lower = np.flip(filtered[lower_index:q_peaks[i] + 1])
        # get the first index in the reversed list exceeding the average level
        lower_difference = next((np.where(search_lower == x)[0][0] for x in search_lower if x >= average_level),
                                tolerance)
        lower_bound = q_peaks[i] - lower_difference

        # upper range to search over
        upper_index = min(s_peaks[i] + tolerance, len(filtered))
        search_upper = filtered[s_peaks[i]:upper_index]
        # get the first index in the list exceeding the average level
        upper_difference = next((np.where(search_upper == x)[0][0] for x in search_upper if x >= average_level),
                                tolerance)
        upper_bound = s_peaks[i] + upper_difference

        duration = upper_bound - lower_bound

        qrs_durations[i] = duration / frequency
        lower_bounds[i] = lower_bound
        upper_bounds[i] = upper_bound

    return {'qrs_durations': qrs_durations, 'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds}





#############################
# input heartbeat from biosppy.signals.ecg.correct_rpeaks
# tolerance sets the range of indices to search for the peak
# returns dictionary containing 's_peak' and 'q_peak'
#############################
def extract_qs_peaks_hb(mean_heartbeat, tolerance=30):
    r_peak = np.argmax(mean_heartbeat)

    # save detected s peak
    upper_index = min(r_peak + tolerance, len(mean_heartbeat))
    s_peak = r_peak + np.argmin(mean_heartbeat[r_peak:upper_index])

    # save detected q peak
    lower_index = max(r_peak - tolerance, 0)
    q_peak = r_peak - tolerance + np.argmin(mean_heartbeat[lower_index:r_peak])

    return {'s_peak': s_peak, 'q_peak': q_peak}


#############################
# given q and s peak, find the QRS duration for the mean heartbeat
# input q and s peak from output of extract_qs_peaks_hb
# tolerance sets the range of indices to search for the point at which the signal returns to its mean
# returns a dictionary of 'qrs_duration' in seconds, and indices of 'lower_bound' and 'upper_bound'
#############################
def extract_qrs_duration_hb(mean_heartbeat, qs_data, tolerance=30, frequency=300):
    s_peak = qs_data['s_peak']
    q_peak = qs_data['q_peak']

    # baseline level of the signal
    median_level = median(mean_heartbeat)

    # lower range to search over
    lower_index = max(0, q_peak - tolerance)
    search_lower = np.flip(mean_heartbeat[lower_index:q_peak + 1])
    # get the first index in the reversed list exceeding the average level
    lower_difference = next((np.where(search_lower == x)[0][0] for x in search_lower if x >= median_level),
                            tolerance)
    lower_bound = q_peak - lower_difference

    # upper range to search over
    upper_index = min(s_peak + tolerance, len(mean_heartbeat))
    search_upper = mean_heartbeat[s_peak:upper_index]
    # get the first index in the list exceeding the average level
    upper_difference = next((np.where(search_upper == x)[0][0] for x in search_upper if x >= median_level),
                            tolerance)
    upper_bound = s_peak + upper_difference

    duration = (upper_bound - lower_bound) / frequency

    return {'qrs_duration': duration, 'lower_bound': lower_bound, 'upper_bound': upper_bound}


#############################
# find T peak
# input upper_bound extract_qrs_duration_hb
# tolerance sets the range of indices to search for the point at which the signal returns to its mean
# returns a dictionary of 'qrs_duration' in seconds, and indices of 'lower_bound' and 'upper_bound'
#############################
def extract_t_peak_hb(mean_heartbeat, qrs_data, tolerance=30):
    upper_bound = qrs_data['upper_bound']

    # save detected s peak
    t_peak = upper_bound + np.argmax(mean_heartbeat[upper_bound:len(mean_heartbeat)])

    return {'t_peak': t_peak}
