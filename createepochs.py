import csv
import os
import sourcedata as sd
import sys

# parameter to be varied
eeg_epoch_width_in_s = int(sys.argv[1])
num_classes = int(sys.argv[2])

# set up file location paths
edf_path = 'data/edf/'
stage_path = 'data/sleep_staging/'
epochs_path = 'data/epochs_{0}c/'.format(str(num_classes))

# create epochs from all EDF files
edf_files = [file for file in os.listdir(edf_path)]
for edf_file in edf_files:
    species = edf_file.split('.')[0]
    print('Processing ' + species)
    edf_filename = edf_path + edf_file
    stage_filename = stage_path + species + '_Stages.csv'
    eeg_epochs, stage_epochs = sd.create_epochs(eeg_epoch_width_in_s,
                                                edf_filename,
                                                stage_filename)
    template = '{0}_ew{1}.csv'
    common_labels = [str(eeg_epoch_width_in_s)]
    output_filename = template.format(species + '_eeg', *common_labels)
    sd.write_data(epochs_path + output_filename, eeg_epochs)
    output_filename = template.format(species + '_labels', *common_labels)
    sd.write_data(epochs_path + output_filename, stage_epochs)
