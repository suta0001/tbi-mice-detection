import csv
import os
import preprocess as pp
import sourcedata as sd
import sys

# parameter to be varied
eeg_epoch_width_in_s = int(sys.argv[1])
num_classes = int(sys.argv[2])
overlap = sys.argv[3].lower() == 'true'

# set up file location paths
if overlap:
    epochs_path = 'data/epochs_{0}c/'.format(str(num_classes))
else:
    epochs_path = 'data/epochs_novl_{0}c/'.format(str(num_classes))
epochs_files = [file for file in os.listdir(epochs_path) if
                '_pp1_ew{0}'.format(eeg_epoch_width_in_s) in file]
for epochs_file in epochs_files:
    species = epochs_file.split('_pp1')[0]
    print('Processing ' + species)
    # read EEG epochs
    epochs_filename = epochs_path + epochs_file
    eeg_epochs = []
    sd.read_data(epochs_filename, eeg_epochs)
    # do AWICA on EEG epochs
    eeg_epochs = pp.do_awica(eeg_epochs)
    # write filtered EEG epochs
    template = '{0}_pp2_ew{1}.csv'
    common_labels = [str(eeg_epoch_width_in_s)]
    output_filename = template.format(species, *common_labels)
    sd.write_data(epochs_path + output_filename, eeg_epochs)
