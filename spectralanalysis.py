import numpy as np
from scipy.integrate import simps
from scipy.signal import welch


def calc_psd(eeg_epoch, fs=256):
    f, psd = welch(eeg_epoch, fs, window='hamming')
    return f, psd


def calc_bandpower(psd, freqs, freq_low, freq_high):
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= freq_low, freqs <= freq_high)
    bp = simps(psd[idx_band], dx=freq_res)
    return bp


def calc_edge_frequency(psd, freqs, target):
    total_power = calc_bandpower(psd, freqs, freqs[0], freqs[-1])
    power_ratio = 0.0
    findex = 0
    while power_ratio < target and findex < len(freqs) - 2:
        findex += 1
        power_ratio += (calc_bandpower(psd, freqs, freqs[findex - 1],
                        freqs[findex]) / total_power)
    return freqs[findex]


def calc_spectral_entropy(psd):
    psd_norm = np.divide(psd, psd.sum())
    se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    se /= np.log2(psd_norm.size)
    return se
