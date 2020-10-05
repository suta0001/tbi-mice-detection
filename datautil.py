import csv
import h5py
import numpy as np
import os
import random
import pyedflib
from sklearn.utils.validation import column_or_1d
import statistics


def build_dataset(epochs_path, num_classes, epoch_width_in_s, pp_step, featgen,
                  species_set, num_samples=0):
    """Build labeled dataset for training or validation from source epochs.

    Args:
        epochs_path: directory path where epoch files are stored
        num_classes: number of classes
        epoch_width_in_s: epoch duration in s
        pp_step: applied preprocessing step(s)
        featgen: applied feature generator
        species_set: set of species used for the dataset
        num_samples: number of samples to build (0 means all samples)

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
    if num_samples != 0 and num_samples <= len(labels):
        idx = random.sample(list(range(len(labels))), num_samples)
        data_epochs = data_epochs[idx, :]
        labels = labels[idx]
    return data_epochs, labels


def calc_average_features(data_epochs, labels, num_classes):
    data_epochs = np.array(data_epochs)
    labels = np.array(labels, dtype=int)
    avg_features = np.zeros((num_classes, len(data_epochs[0])))
    for i in range(num_classes):
        temp_epochs = data_epochs[labels == i]
        avg_features[i] = np.mean(temp_epochs, axis=0)
    return avg_features


def calc_baseline_spectral_powers(epochs_path, num_classes, epoch_width_in_s,
                                  pp_step, species_set):
    """Calculate baseline spectral powers for each class.
       Supported for 4-class classification only.

    Args:
        epochs_path: directory path where epoch files are stored
        num_classes: number of classes
        epoch_width_in_s: epoch duration in s
        pp_step: applied preprocessing step(s)
        species_set: set of species used for the dataset

    Returns:
        A list of average subband powers from first 5 epochs of each species
        for all classes (row = class, column = average subband powers).
        Rows = [Sham Wake, Sham Sleep, TBI Wake, TBI Sleep]
        Columns = [delta, theta, alpha, sigma, beta, gamma]
    """
    baselines_sw = np.zeros(6)
    baselines_ss = np.zeros(6)
    baselines_tw = np.zeros(6)
    baselines_ts = np.zeros(6)
    num_species_sw = 0
    num_species_ss = 0
    num_species_tw = 0
    num_species_ts = 0
    for species in species_set:
        filename = os.path.join(epochs_path,
                                '{}_BL5_ew{}.h5'.format(species,
                                                        epoch_width_in_s))
        groups = read_groups_from_hdf5(filename,
                                       '{}_{}'.format(pp_step, 'spectral'))
        for group in groups:
            fgroup = '{}_spectral/{}'.format(pp_step, group)
            temp_epochs = np.array(read_data_from_hdf5(filename, fgroup))
            # average power in each subband for first 5 epochs
            averages = np.mean(temp_epochs[0:5], axis=0)
            if 'Sham' in species and group == 'wake':
                baselines_sw = np.add(baselines_sw, averages[0:6])
                num_species_sw += 1
            elif 'Sham' in species and group == 'sleep':
                baselines_ss = np.add(baselines_ss, averages[0:6])
                num_species_ss += 1
            elif 'TBI' in species and group == 'wake':
                baselines_tw = np.add(baselines_tw, averages[0:6])
                num_species_tw += 1
            elif 'TBI' in species and group == 'sleep':
                baselines_ts = np.add(baselines_ts, averages[0:6])
                num_species_ts += 1
    baselines = [np.divide(baselines_sw, num_species_sw),
                 np.divide(baselines_ss, num_species_ss),
                 np.divide(baselines_tw, num_species_tw),
                 np.divide(baselines_ts, num_species_ts)]
    return baselines


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


def decibel_normalize(featgen, baselines, epochs, labels):
    """Apply decibel normalization to subband power.

    Args:
        featgen: applied feature generator (normsp or pe_normsp only)
        baselines: subband power baselines
        epochs: feature epochs to normalize
        labels: feature labels

    Returns:
        Normalized feature epochs
    """
    norm_epochs = np.empty_like(epochs)
    for i in range(len(labels)):
        cur_baselines = baselines[labels[i]]
        if featgen == 'spectral':
            db_norm_powers = np.log10(np.divide(epochs[i][0:6], cur_baselines))
            norm_epochs[i] = np.array(db_norm_powers.tolist() + [epochs[i][6]])
        elif featgen == 'pe_spectral':
            db_norm_powers = np.log10(np.divide(epochs[i][9:15],
                                      cur_baselines))
            norm_epochs[i] = np.array(epochs[i][0:9].tolist() +
                                      db_norm_powers.tolist() +
                                      [epochs[i][15]])
        elif featgen == 'wpe_spectral':
            db_norm_powers = np.log10(np.divide(epochs[i][63:69],
                                      cur_baselines))
            norm_epochs[i] = np.array(epochs[i][0:63].tolist() +
                                      db_norm_powers.tolist() +
                                      [epochs[i][69]])
    return norm_epochs


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


def write_metrics(metrics_path, model, num_classes, eeg_epoch_width_in_s,
                  overlap, num_samples, pp_step, featgen, target_names,
                  reports):
    """
    Write classification metrics to files. target_names must be used to
    generate reports.

    Args:
        metrics_path: directory to write the metrics into
        model: classifier model
        num_classes: number of classification target classes
        eeg_epoch_width_in_s: EEG epoch width in seconds
        overlap: True for overlapping EEG epochs
        num_samples: number of samples used in training (0 = all samples)
        pp_step: preprocessing step
        featgen: feature extraction method
        target_names: names of target classes
        reports: list of sklearn.metrics.classification_report

    Returns:
        None
    """
    # define file name format
    if overlap:
        outfile = '{}rs_{}c_ew{}_{}_{}_{}_metrics.csv'
        moutfile = '{}rs_novl_{}c_ew{}_{}_{}_{}_avg_metrics.csv'
        soutfile = '{}rs_novl_{}c_ew{}_{}_{}_{}_std_metrics.csv'
    else:
        outfile = '{}rs_novl_{}c_ew{}_{}_{}_{}_metrics.csv'
        moutfile = '{}rs_novl_{}c_ew{}_{}_{}_{}_avg_metrics.csv'
        soutfile = '{}rs_novl_{}c_ew{}_{}_{}_{}_std_metrics.csv'
    outfile = outfile.format(model, num_classes, eeg_epoch_width_in_s,
                             pp_step, featgen, num_samples)
    moutfile = moutfile.format(model, num_classes, eeg_epoch_width_in_s,
                               pp_step, featgen, num_samples)
    soutfile = soutfile.format(model, num_classes, eeg_epoch_width_in_s,
                               pp_step, featgen, num_samples)

    # define output files
    outfile = os.path.join(metrics_path, outfile)
    moutfile = os.path.join(metrics_path, moutfile)
    soutfile = os.path.join(metrics_path, soutfile)

    metrics = ['precision', 'recall', 'f1-score', 'support']
    outputs = []
    # fold data
    # form array of header labels and add to outputs
    header_labels = ['fold', 'accuracy']
    for label in target_names:
        for metric in metrics:
            header_labels.append('{}_{}'.format(label, metric))
    outputs.append(header_labels)

    # form array of metric values and add to outputs
    for i in range(len(reports)):
        metric_values = [i, reports[i]['accuracy']]
        for label in target_names:
            for metric in metrics:
                metric_values.append(reports[i][label][metric])
        outputs.append(metric_values)
    write_data(outfile, outputs)

    # summary data
    # form array of header labels and add to outputs
    moutputs = []
    soutputs = []
    header_labels = ['model', 'num_classes', 'epoch_width', 'overlap',
                     'num_samples', 'preprocess', 'feat', 'accuracy']
    for label in target_names:
        for metric in metrics:
            header_labels.append('{}_{}'.format(label, metric))
    moutputs.append(header_labels)
    soutputs.append(header_labels)
    # form array of metric values and add to outputs
    if num_samples == 0:
        num_samples = 'all'
    accuracies = [reports[i]['accuracy'] for i in range(len(reports))]
    mmetric_values = [model, num_classes, eeg_epoch_width_in_s,
                      overlap, num_samples, pp_step, featgen,
                      statistics.mean(accuracies)]
    smetric_values = [model, num_classes, eeg_epoch_width_in_s,
                      overlap, num_samples, pp_step, featgen,
                      statistics.stdev(accuracies)]
    for label in target_names:
        for metric in metrics:
            values = [reports[i][label][metric] for i in range(len(reports))]
            mmetric_values.append(statistics.mean(values))
            smetric_values.append(statistics.stdev(values))
    moutputs.append(mmetric_values)
    soutputs.append(smetric_values)
    write_data(moutfile, moutputs)
    write_data(soutfile, soutputs)
