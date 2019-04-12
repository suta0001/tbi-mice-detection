import numpy as np


def zero_crossing_rate(eeg_epoch):
    eeg_epoch = np.asarray(eeg_epoch)
    zero_crosses = np.nonzero(np.diff(eeg_epoch > 0))[0]
    return zero_crosses.size
