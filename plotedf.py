import matplotlib.pyplot as plt
import numpy as np
import os
import preprocess as pp
import pyedflib
import statistics


# set up file location path
edf_path = 'data/edf/'

# load EEG signals
eeg_signals = []
for edf_filename in ['TBI106_BL5.edf']:  # os.listdir(edf_path):
    with pyedflib.EdfReader(edf_path + edf_filename) as edf_file:
        eeg_signal = edf_file.readSignal(0)
        # print(edf_filename + ': ' + ' ' +
        #       'mean = ' + str(statistics.mean(eeg_signal)) + ' ' +
        #       'stdev = ' + str(statistics.stdev(eeg_signal)) + ' ' +
        #       'min = ' + str(np.amin(eeg_signal)) + ' ' +
        #       'max = ' + str(np.amax(eeg_signal)))
    eeg_signals.append(eeg_signal)
eeg_signals.append(pp.awica(eeg_signals[0]))
# plot all EEG signals for a given range
tstart = 0
tend = len(eeg_signals[0])
t = list(range(tstart, tend))
axref = plt.subplot(len(eeg_signals), 1, 1)
plt.plot(t, eeg_signals[0][tstart: tend])
plt.ylabel('EEG signal (uV)')
for i in range(1, len(eeg_signals)):
    plt.subplot(len(eeg_signals), 1, i + 1, sharex=axref)
    plt.plot(t, eeg_signals[i][tstart: tend])
    plt.ylabel('EEG signal (uV)')
plt.xlabel('time')
plt.show()
