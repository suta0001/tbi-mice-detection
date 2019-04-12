import numpy as np


def zero_crossing_rate(eeg_epoch):
    eeg_epoch = np.asarray(eeg_epoch)
    zero_crosses = np.nonzero(np.diff(eeg_epoch > 0))[0]
    return zero_crosses.size


def calc_hjorth_params(eeg_epoch):
    eeg_epoch = np.asarray(eeg_epoch)
    first_deriv = np.diff(eeg_epoch)
    second_deriv = np.diff(eeg_epoch, 2)

    var_zero = np.mean(eeg_epoch ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity
