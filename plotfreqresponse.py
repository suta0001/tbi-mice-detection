import matplotlib.pyplot as plt
import numpy as np
import os
import preprocess as pp
import pyedflib
from scipy.fftpack import fft


# set up file location path
edf_path = 'data/edf/'

# load EEG signals
fft_signals = []
N = 0
for edf_filename in os.listdir(edf_path):
#for edf_filename in ['TBI101_BL5.edf', 'TBI102_BL5.edf', 'TBI103_BL5.edf']:
    with pyedflib.EdfReader(edf_path + edf_filename) as edf_file:
        eeg_signal = edf_file.readSignal(0)
        # print(edf_filename + ': ' + ' ' +
        #       'mean = ' + str(statistics.mean(eeg_signal)) + ' ' +
        #       'stdev = ' + str(statistics.stdev(eeg_signal)) + ' ' +
        #       'min = ' + str(np.amin(eeg_signal)) + ' ' +
        #       'max = ' + str(np.amax(eeg_signal)))
        N = len(eeg_signal)
        fft_signals.append(fft(pp.butter_bandpass_filter(eeg_signal, 1, 5, 256)))

# plot all EEG signals for a given range
T = 1.0 / 256.0
tstart = 0
tend = len(fft_signals[0])
x = np.linspace(0, 1.0 / (2.0 * T), N // 2)
axref = plt.subplot(len(fft_signals), 1, 1)
plt.plot(x, 2.0 / N * np.abs(fft_signals[0][0: N//2]))
for i in range(1, len(fft_signals)):
    plt.subplot(len(fft_signals), 1, i + 1, sharex=axref)
    plt.plot(x, 2.0 / N * np.abs(fft_signals[i][0: N//2]))
plt.show()
