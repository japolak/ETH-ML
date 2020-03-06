import biosppy
import numpy as np
import pandas as pd
from extract_features import *
import statistics
from biosppy.signals import ecg
import matplotlib.pyplot as plt



def var(lst):
    if len(lst) > 1:
        return statistics.variance(lst)
    else:
        return 0


def mn(lst):
    if len(lst) >= 1:
        return statistics.mean(lst)
    else:
        return 0


def md(lst):
    if len(lst) >= 1:
        return statistics.median(lst)
    else:
        return 0


def list_outliers(data_frame):
    outliers_ind = []

    for index, row in data_frame.iterrows():
        try:
            # process the row as usual
            row = row.dropna()
            processed_signal = biosppy.signals.ecg.ecg(
                signal=row, sampling_rate=300, show=False)
            # extract the peaks
            r_peaks = biosppy.signals.ecg.christov_segmenter(
                processed_signal['filtered'], sampling_rate=300)
            # correct the peaks
            r_peaks = biosppy.signals.ecg.correct_rpeaks(signal=processed_signal['filtered'], rpeaks=r_peaks['rpeaks'],
                                                     sampling_rate=300,
                                                     tol=0.05)
            # probably shitty data if there is only one r peak
            if len(r_peaks[0]) <= 1:
                outliers_ind.append(index)
        except ValueError:
            print("Failed at index " + str(index))

    return outliers_ind

# plot a row; specify the row


def plot_row(row_index, data, show_qrs = False):
    import matplotlib.pyplot as plt
    # process data
    test_row = data.iloc[row_index, :].dropna()
    processed_signal = biosppy.signals.ecg.ecg(
        signal=test_row, sampling_rate=300, show=False)
    # R peaks
    r_peaks = biosppy.signals.ecg.christov_segmenter(
        processed_signal['filtered'], sampling_rate=300)
    r_peaks = biosppy.signals.ecg.correct_rpeaks(
        signal=processed_signal['filtered'], rpeaks=r_peaks['rpeaks'], sampling_rate=300, tol=0.05)
    # QRS peaks
    extracted_peaks = extract_qs_peaks(processed_signal, r_peaks, tolerance=30)
    qrs_extracted = extract_qrs_duration(
        processed_signal, extracted_peaks, tolerance=30)
    # plot
    pd.DataFrame(processed_signal['filtered']).plot()
    for peak in r_peaks['rpeaks']:
        plt.axvline(x=peak, color='red')
    if show_qrs: 
        for peak in extracted_peaks['s_peaks']: plt.axvline(x=peak, color='green')
        for peak in extracted_peaks['q_peaks']: plt.axvline(x=peak, color='orange')
        for bound in qrs_extracted['lower_bounds']: plt.axvline(x=bound, color='black')
        for bound in qrs_extracted['upper_bounds']: plt.axvline(x=bound, color='black')
    plt.axhline(y=mean(processed_signal['filtered']))
    plt.show()


def denoiseDataManual(data, dictionary):
    m = data.shape[1]
    for index, start in dictionary.items():
        empty = np.full(m, np.nan)
        trim = data.iloc[index, :].dropna()[start:].values
        empty[0:len(trim)] = trim
        data.iloc[index, :] = empty
    print('Data denoised!')
    return data


def trimDataManual(dataOriginal, fskip = 180, lskip = 180):
    data = dataOriginal.copy()
    m = data.shape[1]
    for index, row in data.iterrow():
        empty = np.full(m, np.nan)
        values = row.dropna().values
        n = len(values)
        trim = values[fskip:(n-lskip)]
        empty[(fskip):(n-lskip)] = trim
        data.iloc[index, :] = empty
    return data


def inspectInvertionsOLD(OGdata,labels=None, splits=7, thr=1, multip=2, verbose=1):
    data = OGdata.copy()
    inv_data = []
    inv_label = []
    for index, row in data.iterrows():
        fskip = 360
        lskip = 180
        signal = row.dropna().values
        n = signal.shape[0]
        window = int(int((n-fskip-lskip)/180)/splits)
        if window == 0: 
            window = 1 ; fskip = 0; lskip = 0
        vote = np.zeros(splits)
        for i in range(splits):
            mask = signal[(fskip + window*i*180):(fskip + window*(i+1)*180)]
            bottom_peak = (max(mask) - median(mask))*multip < (median(mask) - min(mask))
            if bottom_peak:
                vote[i] = 1
        if splits - vote.sum() <= thr:
            inv_data.append(index)
            if verbose: print('Index', index, 'flip, ', int(vote.sum()),'/', splits, 'votes')
            if labels is not None: inv_label.append(int(labels.iloc[index,:].values.ravel()))
    print('\n Togeter',len(inv_data),'observations would be inverted! \n')
    if labels is not None: 
        print(' Class distribution of those inverions',np.bincount(np.array(inv_label)))
        return inv_data, inv_label
    else: return inv_data

def invertData(OGdata, invert_list=None):
    data = OGdata.copy()
    for ind in invert_list:
        row = data.iloc[ind, :].dropna().values
        data.iloc[ind, 0:len(row)] = row*(-1)
    return data

def invertDataLoop(OGdata, invert_list=None):
    data = OGdata.copy()
    for index,row in data.iterrows():
        if index in invert_list:
            raw = row.dropna().values
            data.iloc[index, 0:len(raw)] = raw*(-1)
    return data


def invertDataFast(OGdata, invert_list=None):
    data = OGdata.copy()
    data.iloc[invert_list,:] = data.iloc[invert_list,:].values*(-1)
    return data

def reject(peaks, m=2, b=100):
    return peaks[abs(peaks - np.median(peaks)) < (m * np.quantile(peaks, 0.75)+b)]

def threshold(peaks, m=2, b=100):
    return (m*np.quantile(peaks, 0.75)+np.median(peaks)+b)

def makeplot(signal, fIndp=0, bIndp=0, smin=0, smax=0, m=2, b=100):
    import matplotlib.pyplot as plt
    plt.plot(range(0,len(signal)),signal)
    if not fIndp==bIndp==smin==smax==0:
        plt.axvline(x=fIndp, color='green')
        plt.axvline(x=bIndp, color='red')
        plt.axhline(y=-threshold(smin, m, b), color='grey')
        plt.axhline(y=threshold(smax, m, b), color='grey')
    plt.show()

def denoiseData(OGdata, run=1, m=2, b=1, plot_subset=None):
    if plot_subset is not None: 
        temp = OGdata.copy()
        data = temp.iloc[plot_subset,:]
    if plot_subset is None:
        data = OGdata.copy()

    width_data = data.shape[1]
    length_data = data.shape[0]
    for index, row in data.iterrows():
        signal = row.dropna().values
        #for index in range(length_data):
        #signal = data.iloc[index,:].dropna().values
        if len(signal) % 180 == 0: signal = signal[0:(len(signal)-1)]
        l = len(signal)
        if run == 1:
            if l > 6000:
                p = 0.25
            else:
                p = 0.2
        if run == 2:
            if l > 5000:
                p = 0.1
            else: 
                p = 0.05
        hb = int(l/180)
        #print('length:',l,'heartbeats:',hb)
        s = signal - median(signal)
        smin = np.zeros(hb+1)
        smax = np.zeros(hb+1)
        for j,i in enumerate(range(hb+1,0,-1)):
            #print(i, j)
            if i == hb+1: 
                mask = s[0:(l-(i-1)*180)]
                #print(0, (l-(i-1)*180))
            else: 
                mask = s[(l-i*180): (l-(i-1)*180)]
                #print((l-i*180), (l-(i-1)*180))
            smin[j] = -min(mask)
            smax[j] = max(mask)
        smini = np.invert(np.in1d(smin, reject(smin,m,b)))
        smaxi = np.invert(np.in1d(smax, reject(smax,m,b)))
        lim = int(hb*p)+1

        front = [0,0]
        for a, check in enumerate([smini.astype(int)[0:lim], smaxi.astype(int)[0:lim]]):
            if check.sum()>0: 
                front[a] = abs((np.argmax(check[::-1])+1)-(lim+1))
        fInd = np.array(front).max()

        back = [0, 0]
        for b, check in enumerate([smini.astype(int)[::-1][0:lim], smaxi.astype(int)[::-1][0:lim]]):
            if check.sum() > 0:
                back[b] = abs((np.argmax(check[::-1])+1)-(lim+1))
        bInd = np.array(back).max()

        if fInd>0: 
            fIndp = (l-(hb+1-fInd)*180)
        else: 
            fIndp = 0
        if bInd>0: 
            bIndp = (l - bInd*180)
        else: 
            bIndp = l
        clean = signal[fIndp:bIndp]
        empty = np.full(width_data, np.nan)
        empty[0:len(clean)] = clean
        if plot_subset is not None: 
            makeplot(signal, fIndp, bIndp, smin, smax, m, b)
        if plot_subset is None:
            data.iloc[index, :] = empty
        print('Index:', index, ' from:', fIndp, 'to:', bIndp,'signal len:', len(signal), 'clean len:', len(clean))
    if plot_subset is None:
        return data


def makemaeplot(peaks, m=5):
    plt.plot(range(0, len(peaks)), peaks)
    plt.axhline(y=thresholdmae(peaks, m, 0.75), color='grey')
    plt.axhline(y=thresholdmae(peaks, m, 0.25), color='grey')
    plt.show()

def rejectmae(peaks, m=5, q=0.75):
    return peaks[abs(peaks) < m * np.quantile(peaks, q)]

def thresholdmae(peaks, m=5, q=0.75):
    return (m*np.quantile(peaks, q)+np.median(peaks))

def removeOut(temp, m=5, make_plot=False):
    mae = abs(temp).sum(axis=1)
    err = mae - np.median(mae)
    if make_plot:
        makemaeplot(err, m)
    errmin = np.in1d(err, rejectmae(err,m,0.25)).astype(int)
    errmax = np.invert(np.in1d(err, rejectmae(err,m,0.75))).astype(int)
    mask = np.invert((errmin+errmax).astype(bool))
    tempf = temp[mask, :]
    if make_plot:
        for i in range(tempf.shape[0]):
            plt.plot(tempf[i, :], color='purple')
        plt.show()
    return tempf

def peak_magnitude(mask, multip=1.3):
    return (max(mask) - np.median(mask)) * multip < (np.median(mask) - min(mask))

def peak_order(mask):
    return np.argmin(mask) < np.argmax(mask)

def inspectInvertions(OGdata, labels = None, verbose = True, plot = False, subset = None, thr=0.9):
    
    if subset is not None:
        temp = OGdata.copy()
        data = temp.iloc[subset, :]
    if subset is None:
        data = OGdata.copy()

    data = OGdata.copy()
    inv_data = []
    inv_label = []
    for index, row in data.iterrows():
        raw_signal = row.dropna().values
        #index = 116
        #raw_signal = data.iloc[index, :].dropna().values
        signal = ecg.ecg(signal=raw_signal, sampling_rate=300, show=False)
        # Filtered ECG signal (N,)
        #Sfilt = signal['filtered']
        # R-peak location indecies (n,)
        #Srpeak = signal['rpeaks']
        # Template heartbeat  (n,180)
        Stemp = signal['templates']
        # Heart rate Instantaneous (n-1,)
        # Sheartr = signal['heart_rate']
        

        hb = len(Stemp)
        Stempf = removeOut(Stemp, m=4)
        hbf = Stempf.shape[0]
        votes = np.zeros(hbf)
        for i in range(hbf):
            mask = Stempf[i, :]
            if peak_magnitude(mask,1.5):  
                votes[i] = 1
            if (peak_magnitude(mask,0.8) or peak_order(mask)) and (min(mask)<-400):
                votes[i] = 1
            if plot: plt.plot(mask, color='purple')
        if plot: plt.show()

        if (votes.sum()/hbf) >= thr:
            inv_data.append(index)
            if verbose: print('Index', index, 'inverted, prob:', round(votes.sum()/hbf, 3))
            if labels is not None: inv_label.append(int(labels.iloc[index,:].values.ravel()))
    print('\n Togeter',len(inv_data),'observations would be inverted! \n')
    if labels is not None: 
        print(' Class distribution of those inverions',np.bincount(np.array(inv_label)))
        return inv_data, inv_label
    if labels is None:
        return inv_data

