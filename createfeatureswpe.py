import csv
import extractfeatures as ef
import os
import pyentrp.entropy as entropy
import pywt
import sourcedata as sd
import sys

# parameter to be varied
eeg_epoch_width_in_s = int(sys.argv[1])
eeg_source = 'pp2'

# set up file location paths
epochs_path = 'data/epochs/'
wpe_path = 'data/wpe/'

epochs_files = [file for file in os.listdir(epochs_path) if
                '_{0}_ew{1}'.format(eeg_source, eeg_epoch_width_in_s) in file]
for epochs_file in epochs_files:
    # species = epochs_file.split('_eeg')[0]
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
            # extract wavelet components
            coeffs = pywt.wavedec(eeg_epoch, 'db4', level=6)
            wcs = []
            n = len(eeg_epoch)
            wcs.append(pywt.upcoef('a', coeffs[0], 'db4', level=6, take=n))
            for i in range(1, 7):
                wcs.append(pywt.upcoef('d', coeffs[i], 'db4', level=7 - i,
                                       take=n))
            # calculate permutation entropy for each wavelet component
            for wc in wcs:
                for order in [3, 5, 7]:
                    for delay in [1, 5, 10]:
                        feature.append(entropy.permutation_entropy(wc,
                                       order=order, delay=delay,
                                       normalize=True))
            eeg_epochs.append(feature)
    # write features
    template = '{0}_{1}_ew{2}.csv'
    common_labels = [eeg_source, str(eeg_epoch_width_in_s)]
    output_filename = template.format(species, *common_labels)
    sd.write_data(wpe_path + output_filename, eeg_epochs)
