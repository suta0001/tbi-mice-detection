import csv
import sourcedata as sd
import sys

# parameters to be varied
eeg_epoch_width_in_s = int(sys.argv[1])
eeg_source = 'pp2'

# set up dataset folds for cross-validation
epochs_path = 'data/epochs/'
sxx_path = 'data/spectrogram/'
cv_path = 'data/cv/'
dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
sxx_template = '{0}_{1}_ew{2}.csv'
sxx_common_labels = [eeg_source, str(eeg_epoch_width_in_s)]
epoch_template = '{0}_ew{1}.csv'
epoch_common_labels = [str(eeg_epoch_width_in_s)]
cv_template = '{0}_{1}_ew{2}_f{3}_sxx.csv'

for fold, dataset_fold in enumerate(dataset_folds):
    cv_common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(fold)]
    # process training set
    train_epochs = []
    train_labels = []
    print('Processing fold ' + str(fold) + '...')
    for i in range(0, 7):
        sxx_epochs = []
        stage_epochs = []
        print('Processing training set: ' + dataset_fold[i] + '...')
        input_filename = sxx_template.format(dataset_fold[i] + '_BL5',
                                             *sxx_common_labels)
        sd.read_data(sxx_path + input_filename, sxx_epochs)
        eeg_epochs = [sxx_epochs[i] for i in range(len(sxx_epochs))]
        input_filename = epoch_template.format(dataset_fold[i] + '_BL5_labels',
                                               *epoch_common_labels)
        sd.read_data(epochs_path + input_filename, stage_epochs)
        train_epochs.extend(eeg_epochs)
        train_labels.extend(stage_epochs)
    # process testing set
    test_epochs = []
    test_labels = []
    for i in range(7, 11):
        sxx_epochs = []
        stage_epochs = []
        print('Processing testing set: ' + dataset_fold[i] + '...')
        input_filename = sxx_template.format(dataset_fold[i] + '_BL5',
                                             *sxx_common_labels)
        sd.read_data(sxx_path + input_filename, sxx_epochs)
        eeg_epochs = [sxx_epochs[i] for i in range(len(sxx_epochs))]
        input_filename = epoch_template.format(dataset_fold[i] + '_BL5_labels',
                                               *epoch_common_labels)
        sd.read_data(epochs_path + input_filename, stage_epochs)
        test_epochs.extend(eeg_epochs)
        test_labels.extend(stage_epochs)

    # write training and testing features and labels to files
    print('Writing to files...')
    output_filename = cv_template.format('train_data', *cv_common_labels)
    sd.write_data(cv_path + output_filename, train_epochs)
    output_filename = cv_template.format('train_labels', *cv_common_labels)
    sd.write_data(cv_path + output_filename, train_labels)
    output_filename = cv_template.format('test_data', *cv_common_labels)
    sd.write_data(cv_path + output_filename, test_epochs)
    output_filename = cv_template.format('test_labels', *cv_common_labels)
    sd.write_data(cv_path + output_filename, test_labels)
