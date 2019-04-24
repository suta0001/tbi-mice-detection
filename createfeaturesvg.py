import csv
import extractfeatures as ef
import networkx as nx
import os
import sourcedata as sd
import sys
import visibilitygraphs as vg


# parameter to be varied
eeg_epoch_width_in_s = int(sys.argv[2])
max_degree = 11
eeg_source = sys.argv[1]
num_classes = int(sys.argv[3])

# set up file location paths
epochs_path = 'data/epochs_{0}c/'.format(str(num_classes))
vg_path = 'data/vg_{0}c/'.format(str(num_classes))

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
            nvg = ef.create_visibility_graphs(data, 'normal')
            hvg = ef.create_visibility_graphs(data, 'horizontal')
            dvg = nx.difference(nvg, hvg)
            feature.append(vg.calc_mean_degree(dvg))
            feature.append(vg.calc_mean_degree(hvg))
            feature.extend(vg.calc_degree_probabilities(dvg, max_degree))
            eeg_epochs.append(feature)
    # write features
    template = '{0}_{1}_ew{2}_{3}.csv'
    common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(max_degree)]
    output_filename = template.format(species, *common_labels)
    sd.write_data(vg_path + output_filename, eeg_epochs)
