import csv
import extractfeatures as ef
import math
import numpy as np
import os
from scipy.stats import kurtosis
from scipy.stats import skew
import sourcedata as sd
import spectralanalysis as sa
import statistics
import sys

# parameter to be varied
eeg_epoch_width_in_s = int(sys.argv[2])
eeg_source = sys.argv[1]
num_classes = int(sys.argv[3])

# set up file location paths
epochs_path = 'data/epochs_{0}c/'.format(str(num_classes))
spectral_path = 'data/spectral_{0}c/'.format(str(num_classes))

# define frequency bands
eeg_bands = {
    'delta': (1.0, 3.5),
    'theta': (4.0, 7.5),
    'alpha': (8.0, 12.0),
    'sigma': (13.0, 16.0),
    'beta': (16.5, 25.0),
    'gamma': (30.0, 35.0)
}
bands = ['delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma']

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
            data = [float(i) for i in row]
            # extract features
            feature = []
            f, psd = sa.calc_psd(data, 256)
            total_power = sa.calc_bandpower(psd, f, f[0], f[-1])
            power = [20.0 * math.log10(sa.calc_bandpower(psd, f,
                     eeg_bands[band][0], eeg_bands[band][1]) /
                     1000.0) for band in bands]
            rel_power = [power[i] / total_power for i in range(len(power))]
            feature.extend(power)
            feature.extend(rel_power)
            feature.append(rel_power[0] / rel_power[1])
            feature.append(rel_power[0] / rel_power[2])
            feature.append(rel_power[0] / rel_power[3])
            feature.append(rel_power[0] / rel_power[4])
            feature.append(rel_power[0] / rel_power[5])
            feature.append(rel_power[1] / rel_power[2])
            feature.append(rel_power[1] / rel_power[3])
            feature.append(rel_power[1] / rel_power[4])
            feature.append(rel_power[1] / rel_power[5])
            feature.append(rel_power[2] / rel_power[3])
            feature.append(rel_power[2] / rel_power[4])
            feature.append(rel_power[2] / rel_power[5])
            feature.append(rel_power[3] / rel_power[4])
            feature.append(rel_power[3] / rel_power[5])
            feature.append(rel_power[4] / rel_power[5])
            feature.append((rel_power[0] + rel_power[1]) /
                           (rel_power[2] + rel_power[3] + rel_power[4] +
                           rel_power[5]))
            feature.append(sa.calc_edge_frequency(psd, f, 0.95))
            feature.append(sa.calc_edge_frequency(psd, f, 0.50))
            feature.append(f[np.argmax(psd)])
            feature.append(sa.calc_spectral_entropy(psd))
            feature.append(20.0 * math.log10(statistics.mean(psd) / 1000.0))
            feature.append(20.0 * math.log10(statistics.variance(psd) / 1.0e6))
            feature.append(skew(psd))
            feature.append(kurtosis(psd))
            eeg_epochs.append(feature)
    # write features
    template = '{0}_{1}_ew{2}.csv'
    common_labels = [eeg_source, str(eeg_epoch_width_in_s)]
    output_filename = template.format(species, *common_labels)
    sd.write_data(spectral_path + output_filename, eeg_epochs)
