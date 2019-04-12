import csv
import pyedflib


def create_epochs(time_window_in_s=4, edf_filename=None,
                  stage_filename=None):
    # check for time window validity -- only multiple of 4 s is allowed
    if time_window_in_s % 4 != 0:
        print('Time window must be a multiple of 4 seconds.')
        return None, None

    # read EEG signal from EDF file
    edf_file = pyedflib.EdfReader(edf_filename)
    eeg_signal = edf_file.readSignal(0)

    # read sleep stages from CSV file
    # for Sham* subjects: W -> 0, NR -> 1, R -> 1
    # for TBI* subjects: W -> 2, NR -> 3, R -> 3
    stage_file = open(stage_filename)
    for i in range(3):
        line = stage_file.readline()
    if 'SHAM' in line:
        offset = 0
    else:
        offset = 2
    stages = [line.split(',')[2] for line in stage_file.readlines()[22:21623]]
    for i in range(len(stages)):
        if stages[i] == 'W':
            stages[i] = 0 + offset
        elif stages[i] == 'NR' or stages[i] == 'R':
            stages[i] = 1 + offset
        else:
            stages[i] = -1

    # build EEG and label epochs
    num_samples = time_window_in_s // 4 * 1024
    num_epochs = ((len(eeg_signal) - num_samples) // 1024) + 1
    num_stages = time_window_in_s // 4
    eeg_epochs = [eeg_signal[i * 1024:i * 1024 + num_samples]
                  for i in range(num_epochs)]
    stage_epochs = []
    for i in range(num_epochs):
        stages_temp = stages[i:i + num_stages]
        if stages_temp.count(0 + offset) == num_stages:
            stage_epochs.append([0 + offset])
        elif stages_temp.count(1 + offset) == num_stages:
            stage_epochs.append([1 + offset])
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


def read_data(filename, dataset):
    with open(filename, mode='r', newline='') as csvfile:
        filereader = csv.reader(csvfile)
        for row in filereader:
            data = [float(i) for i in row]
            dataset.append(data)
    return


def write_data(filename, dataset):
    with open(filename, mode='w', newline='') as csvfile:
        filewriter = csv.writer(csvfile)
        for data in dataset:
            filewriter.writerow(data)
    return
