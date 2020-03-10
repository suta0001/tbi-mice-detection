import csv
import h5py
import numpy as np
import os
import pyedflib
from sklearn.utils.validation import column_or_1d


def build_dataset(epochs_path, num_classes, epoch_width_in_s, pp_step, featgen,
                  species_set):
    """Build labeled dataset for training or validation from source epochs.

    Args:
        epochs_path: directory path where epoch files are stored
        num_classes: number of classes
        epoch_width_in_s: epoch duration in s
        pp_step: applied preprocessing step(s)
        featgen: applied feature generator
        species_set: set of species used for the dataset

    Returns:
        A list of data epochs and a list of label epochs
    """

    data_epochs = []
    labels = []
    for species in species_set:
        filename = os.path.join(epochs_path,
                                '{}_BL5_ew{}.h5'.format(species,
                                                        epoch_width_in_s))
        if featgen is not None:
            groups = read_groups_from_hdf5(filename,
                                           '{}_{}'.format(pp_step, featgen))
        else:
            groups = read_groups_from_hdf5(filename,
                                           '{}'.format(pp_step))
        for group in groups:
            if featgen is not None:
                fgroup = '{}_{}/{}'.format(pp_step, featgen, group)
            else:
                fgroup = '{}/{}'.format(pp_step, group)
            temp_epochs = read_data_from_hdf5(filename, fgroup)
            data_epochs.extend(temp_epochs)
            labels += len(temp_epochs) * [get_class_label(num_classes,
                                                          species, group)]
    # convert datasets to numpy arrays
    data_epochs = np.array(data_epochs)
    labels = column_or_1d(np.array(labels, dtype=int))
    return data_epochs, labels


def calc_average_features(data_epochs, labels, num_classes):
    data_epochs = np.array(data_epochs)
    labels = np.array(labels, dtype=int)
    avg_features = np.zeros((num_classes, len(data_epochs[0])))
    for i in range(num_classes):
        temp_epochs = data_epochs[labels == i]
        avg_features[i] = np.mean(temp_epochs, axis=0)
    return avg_features


def create_epochs(time_window_in_s=4, edf_filename=None,
                  stage_filename=None, num_classes=4, overlap=True):
    """Create EEG and stage epochs from an EDF file and its associated
       sleep staging file.

    This function creates EEG and stage epochs from an EDF file
    containing 24-hour EEG recording sampled at 256 Hz and its associated
    sleep staging file.  A sleep stage is assigned for every 4-second
    time window.  This limits valid EEG epoch durations to integer multiples
    of 4 seconds.  There are 3 possible sets of stages to generate: 2 stages,
    4 stages, and 6 stages.  The epochs with duration larger than 4 s can be
    generated as overlapping epochs with a sliding step of 4 s.

    Args:
        time_window_in_s: epoch duration in s
        edf_filename: path to EDF file
        stage_filename: path to associated sleep staging file
        num_classes: number of stages
        overlap: set to True to generate overlapping epochs

    Returns:
        A list of EEG epochs and a list of stage epochs
    """

    # check for time window validity -- only multiple of 4 s is allowed
    if time_window_in_s % 4 != 0:
        print('Time window must be a multiple of 4 seconds.')
        return None, None

    # read EEG signal from EDF file
    edf_file = pyedflib.EdfReader(edf_filename)
    eeg_signal = edf_file.readSignal(0)

    # read sleep stages from CSV file
    # stages are assigned as follows:
    # 2 stages: 0 = Sham, 1 = TBI
    # 4 stages: 0 = Sham W, 1 = Sham NR and R
    #           2 = TBI W, 3 = TBI NR and R
    # 6 stages: 0 = Sham W, 1 = Sham NR, 2 = Sham R
    #           3 = TBI W, 4 = TBI NR, 5 = TBI R
    stage_file = open(stage_filename)
    for i in range(3):
        line = stage_file.readline()
    if 'SHAM' in line:
        offset = 0
    else:
        offset = num_classes // 2
    stages = [line.split(',')[2] for line in stage_file.readlines()[22:21623]]
    for i in range(len(stages)):
        if stages[i] == 'W' or num_classes == 2:
            stages[i] = 0 + offset
        elif stages[i] == 'NR':
            stages[i] = 1 + offset
        elif stages[i] == 'R':
            if num_classes == 4:
                stages[i] = 1 + offset
            elif num_classes == 6:
                stages[i] = 2 + offset
        else:
            stages[i] = -1

    # build EEG and stage epochs
    num_samples = time_window_in_s // 4 * 1024
    num_stages = time_window_in_s // 4
    stage_epochs = []
    if overlap:  # window step is 4 seconds
        num_epochs = ((len(eeg_signal) - num_samples) // 1024) + 1
        eeg_epochs = [eeg_signal[i * 1024:i * 1024 + num_samples]
                      for i in range(num_epochs)]
        for i in range(num_epochs):
            stages_temp = stages[i:i + num_stages]
            # assign stage only if all stages in the epochs are the same
            if stages_temp.count(0 + offset) == num_stages:
                stage_epochs.append([0 + offset])
            elif stages_temp.count(1 + offset) == num_stages:
                stage_epochs.append([1 + offset])
            elif stages_temp.count(2 + offset) == num_stages:
                stage_epochs.append([2 + offset])
            else:
                stage_epochs.append([-1])
    else:  # window step is epoch width
        num_epochs = len(eeg_signal) // num_samples
        eeg_epochs = [eeg_signal[i * num_samples:(i + 1) * num_samples]
                      for i in range(num_epochs)]
        for i in range(num_epochs):
            stages_temp = stages[i * num_stages:(i + 1) * num_stages]
            # assign stage only if all stages in the epochs are the same
            if stages_temp.count(0 + offset) == num_stages:
                stage_epochs.append([0 + offset])
            elif stages_temp.count(1 + offset) == num_stages:
                stage_epochs.append([1 + offset])
            elif stages_temp.count(2 + offset) == num_stages:
                stage_epochs.append([2 + offset])
            else:
                stage_epochs.append([-1])

    # drop epochs with stage == -1 before returning them
    eeg_epochs = [eeg_epochs[i] for i in range(num_epochs)
                  if stage_epochs[i][0] != -1]
    stage_epochs = [stage_epochs[i] for i in range(num_epochs)
                    if stage_epochs[i][0] != -1]

    # close files
    edf_file._close()
    stage_file.close()

    return eeg_epochs, stage_epochs


def get_class_label(num_classes, species, stage):
    if 'Sham' in species:
        offset = 0
    else:
        offset = num_classes // 2
    if num_classes == 4:
        if stage == 'wake':
            label = offset + 0
        elif stage == 'sleep':
            label = offset + 1
    elif num_classes == 6:
        if stage == 'wake':
            label = offset + 0
        elif stage == 'nrem':
            label = offset + 1
        elif stage == 'rem':
            label = offset + 2
    else:
        label = offset
    return label


def read_data(filename):
    dataset = []
    with open(filename, mode='r', newline='') as csvfile:
        filereader = csv.reader(csvfile)
        for row in filereader:
            data = [float(i) for i in row]
            dataset.append(data)
    return dataset


def read_data_from_hdf5(filename, group):
    with h5py.File(filename, 'r') as f:
        dataset = f[group][:]
    return dataset


def read_groups_from_hdf5(filename, group=None):
    with h5py.File(filename, 'r') as f:
        g = f[group] if group else f
        groups = list(g.keys())
    return groups


def write_data(filename, dataset):
    with open(filename, mode='w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        for data in dataset:
            filewriter.writerow(data)


def write_attrs_to_hdf5(filename, group=None, **kwargs):
    with h5py.File(filename, 'a') as f:
        g = f[group] if group else f
        for key, value in kwargs.items():
            g.attrs.create(key, value)


def write_data_to_hdf5(filename, group, dataset):
    dataset = np.asarray(dataset)
    with h5py.File(filename, 'a') as f:
        if group in f.keys():
            del f[group]
        f.create_dataset(group, data=dataset, compression='gzip')
