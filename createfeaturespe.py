import csv
import extractfeatures as ef
import os
import pyentrp.entropy as entropy
import sourcedata as sd
import sys

# parameter to be varied
eeg_epoch_width_in_s = int(sys.argv[2])
pe_orders = list(range(3, 8))
pe_delays = list(range(1, 11))
eeg_source = sys.argv[1]
num_classes = int(sys.argv[3])

# set up file location paths
epochs_path = 'data/epochs_{0}c/'.format(str(num_classes))
pe_path = 'data/pe_{0}c/'.format(str(num_classes))

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
            for order in pe_orders:
                for delay in pe_delays:
                    feature.append(entropy.permutation_entropy(eeg_epoch,
                                   order, delay, normalize=True))
            eeg_epochs.append(feature)
    # write features
    template = '{0}_{1}_ew{2}_{3}t{4}_{5}t{6}.csv'
    common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(pe_orders[0]),
                     str(pe_orders[-1]), str(pe_delays[0]), str(pe_delays[-1])]
    output_filename = template.format(species, *common_labels)
    sd.write_data(pe_path + output_filename, eeg_epochs)
