import numpy as np
import pyentrp.entropy as entropy
import pywt
from scipy.signal import butter, decimate, lfilter
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA


"""
Codes for butter_bandpass and butter_bandpass_filter are from
https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
"""


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def do_bandpass_filter(eeg_epochs, lowcut=0.5, highcut=40, fs=256,
                       order=4):
    bpf_eeg_epochs = []
    for eeg_epoch in eeg_epochs:
        bpf_eeg_epoch = butter_bandpass_filter(eeg_epoch, lowcut, highcut, fs,
                                               order)
        bpf_eeg_epochs.append(bpf_eeg_epoch)
    return bpf_eeg_epochs


def butter_lowpass(highcut, fs, order=4):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def butter_lowpass_filter(data, highcut, fs, order=4):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def do_lowpass_filter(eeg_epochs, highcut=128, fs=256, order=4):
    lpf_eeg_epochs = []
    for eeg_epoch in eeg_epochs:
        lpf_eeg_epoch = butter_lowpass_filter(eeg_epoch, highcut, fs,
                                              order)
        lpf_eeg_epochs.append(lpf_eeg_epoch)
    return lpf_eeg_epochs


def normalize(values):
    meanv = np.mean(values)
    stdv = np.std(values)
    norm_values = [(value - meanv) / stdv for value in values]
    return norm_values


def renyi_entropy(time_series):
    order = 5
    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations
    sorted_idx = entropy._embed(x, order).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    alpha = 2
    re = np.log2((np.power(p, alpha)).sum()) / (1 - alpha)
    return re


def awica(eeg_epoch):
    clean_eeg_epoch = eeg_epoch
    # extract wavelet components (WCs) using DWT
    coeffs = pywt.wavedec(eeg_epoch, 'db4', level=3)
    wcs = []
    n = len(eeg_epoch)
    wcs.append(pywt.upcoef('a', coeffs[0], 'db4', level=3, take=n))
    for i in range(1, 4):
        wcs.append(pywt.upcoef('d', coeffs[i], 'db4', level=4 - i, take=n))

    # select artifactual WCs
    kurtosis_values = normalize([kurtosis(wc) for wc in wcs])
    entropy_values = normalize([renyi_entropy(wc) for wc in wcs])
    crit_wc_index = [i for i in range(len(wcs)) if
                     abs(kurtosis_values[i]) > 1.5 or
                     abs(entropy_values[i] > 1.5)]

    # extract wavelet independent components (WICs) from critical WCs
    if len(crit_wc_index) > 0:
        crit_wcs = [wcs[i] for i in crit_wc_index]
        crit_wcs = np.transpose(crit_wcs)
        transformer = FastICA(tol=0.0001, max_iter=200)
        crit_wics = transformer.fit_transform(crit_wcs)
        crit_wics = np.transpose(crit_wics)
        n_trials = n // 128
        crit_wic_index = []
        # select and reject artifactual WICs
        for j, wic in enumerate(crit_wics):
            kurtosis_values = normalize([kurtosis(wic[i * 128: i * 128 + 127])
                                        for i in range(n_trials)])
            entropy_values = normalize([renyi_entropy(wic[i * 128: i * 128 +
                                       127]) for i in range(n_trials)])
            n_large_kurtosis = sum(abs(value) > 1.5 for value in
                                   kurtosis_values)
            n_large_entropy = sum(abs(value) > 1.5 for value in entropy_values)
            found_artifact = False
            if (n_large_kurtosis > 0.20 * n_trials or n_large_entropy > 0.20 *
               n_trials):
                crit_wics[j] = np.zeros_like(wic)
                found_artifact = True
        # reconstruction
        if found_artifact:
            crit_wics = np.transpose(crit_wics)
            crit_wcs = transformer.inverse_transform(crit_wics)
            crit_wcs = np.transpose(crit_wcs)
            j = 0
            for i in crit_wc_index:
                wcs[i] = crit_wcs[j]
                j += 1
            clean_eeg_epoch = np.sum(wcs, axis=0)
    return clean_eeg_epoch


def do_awica(eeg_epochs):
    clean_eeg_epochs = []
    for eeg_epoch in eeg_epochs:
        clean_eeg_epoch = awica(eeg_epoch)
        clean_eeg_epochs.append(clean_eeg_epoch)
    return clean_eeg_epochs


def do_decimate(eeg_epochs, dec_factor=4):
    dec_eeg_epochs = []
    for eeg_epoch in eeg_epochs:
        dec_eeg_epoch = decimate(eeg_epoch, dec_factor)
        dec_eeg_epochs.append(dec_eeg_epoch)
    return dec_eeg_epochs


def process(eeg_epochs, pp_set='pp4'):
    if pp_set == 'pp1':
        ops_set = [do_bandpass_filter]
        kwargs = {}
    elif pp_set == 'pp2':
        ops_set = [do_bandpass_filter, do_awica]
        kwargs = {}
    elif pp_set == 'pp3':
        ops_set = [do_lowpass_filter]
        kwargs = {}
    elif pp_set == 'pp4':
        ops_set = [do_decimate]
        kwargs = {'dec_factor': 4}
    elif pp_set == 'pp5':
        ops_set = [do_decimate]
        kwargs = {'dec_factor': 2}
    for ops in ops_set:
        eeg_epochs = ops(eeg_epochs, **kwargs)
    return eeg_epochs
