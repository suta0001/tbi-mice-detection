import csv
import extractfeatures as ef
import numpy as np
import os
from scipy.stats import kurtosis
from scipy.stats import skew
import sourcedata as sd
import statistics
import timedanalysis as ta
import sys


# parameter to be varied
eeg_epoch_width_in_s = int(sys.argv[1])
eeg_source = 'pp2'

# set up file location paths
epochs_path = 'data/epochs/'
timed_path = 'data/timed/'

epochs_files = [file for file in os.listdir(epochs_path) if
                '_{0}_ew{1}'.format(eeg_source, eeg_epoch_width_in_s) in file]
for epochs_file in epochs_files:
    species = epochs_file.split('_' + eeg_source)[0]
    print('Processing ' + species)
    # read EEG epochs
    epochs_filename = epochs_path + epochs_file
    eeg_epochs = []
    with open(epochs_filename, mode='r', newline='') as csvfile:
        filereader = csv.reader(csvfile)
        for row in filereader:
            eeg_epoch = [float(i) for i in row]
            # extract features
            feature = []
            feature.append(statistics.mean(eeg_epoch))
            feature.append(statistics.variance(eeg_epoch))
            feature.append(skew(eeg_epoch))
            feature.append(kurtosis(eeg_epoch))
            feature.append(ta.zero_crossing_rate(eeg_epoch))
            feature.extend(ta.calc_hjorth_params(eeg_epoch))
            feature.append(np.percentile(eeg_epoch, 75))
            eeg_epochs.append(feature)
    # write features
    template = '{0}_{1}_ew{2}.csv'
    common_labels = [eeg_source, str(eeg_epoch_width_in_s)]
    output_filename = template.format(species, *common_labels)
    sd.write_data(timed_path + output_filename, eeg_epochs)
