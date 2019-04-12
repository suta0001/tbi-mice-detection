import csv
import numpy as np
import os
from scipy.signal import spectrogram
import sys


# parameter to be varied
eeg_epoch_width_in_s = int(sys.argv[1])
eeg_source = 'pp2'

# set up file location paths
epochs_path = 'data/epochs/'
sxx_path = 'data/spectrogram/'

epochs_files = [file for file in os.listdir(epochs_path) if
                '_{0}_ew{1}'.format(eeg_source, eeg_epoch_width_in_s) in file]
for epochs_file in epochs_files:
    species = epochs_file.split('_' + eeg_source)[0]
    print('Processing ' + species)
    # set up input and output files
    epochs_filename = epochs_path + epochs_file
    sourcefile = open(epochs_filename, mode='r', newline='')
    filereader = csv.reader(sourcefile)
    template = '{0}_{1}_ew{2}.csv'
    common_labels = [eeg_source, str(eeg_epoch_width_in_s)]
    output_filename = sxx_path + template.format(species, *common_labels)
    targetfile = open(output_filename, mode='w', newline='')
    filewriter = csv.writer(targetfile)
    # read epoch and create spectrogram
    for row in filereader:
        data = np.asarray(row, dtype=float)
        f, t, Sxx = spectrogram(data, 256.0)
        Sxx = Sxx[0:41, :].flatten()
        filewriter.writerow(Sxx)
    # close files
    sourcefile.close()
    targetfile.close()
