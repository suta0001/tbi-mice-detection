import csv
import extractfeatures as ef
import sourcedata as sd
import sys

# parameters to be varied
eeg_epoch_width_in_s = int(sys.argv[1])
pe_orders = list(range(3, 8))
pe_delays = list(range(1, 11))
max_degree = 11
eeg_source = 'pp2'

# set up dataset folds for cross-validation
epochs_path = 'data/epochs/'
pe_path = 'data/pe/'
vg_path = 'data/vg/'
spectral_path = 'data/spectral/'
timed_path = 'data/timed/'
cv_path = 'data/cv/'
dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
pe_template = '{0}_{1}_ew{2}_{3}t{4}_{5}t{6}.csv'
pe_common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(pe_orders[0]),
                    str(pe_orders[-1]), str(pe_delays[0]), str(pe_delays[-1])]
vg_template = '{0}_{1}_ew{2}_{3}.csv'
vg_common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(max_degree)]
spectral_template = '{0}_{1}_ew{2}.csv'
spectral_common_labels = [eeg_source, str(eeg_epoch_width_in_s)]
timed_template = '{0}_{1}_ew{2}.csv'
timed_common_labels = [eeg_source, str(eeg_epoch_width_in_s)]
epoch_template = '{0}_ew{1}.csv'
epoch_common_labels = [str(eeg_epoch_width_in_s)]
cv_template = '{0}_{1}_ew{2}_f{3}_{4}t{5}_{6}t{7}_{8}_sp_ti.csv'

for fold, dataset_fold in enumerate(dataset_folds):
    cv_common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(fold),
                        str(pe_orders[0]), str(pe_orders[-1]),
                        str(pe_delays[0]), str(pe_delays[-1]),
                        str(max_degree)]
    # process training set
    train_epochs = []
    train_labels = []
    print('Processing fold ' + str(fold) + '...')
    for i in range(0, 7):
        pe_epochs = []
        vg_epochs = []
        spectral_epochs = []
        timed_epochs = []
        stage_epochs = []
        print('Processing training set: ' + dataset_fold[i] + '...')
        input_filename = pe_template.format(dataset_fold[i] + '_BL5',
                                            *pe_common_labels)
        sd.read_data(pe_path + input_filename, pe_epochs)
        input_filename = vg_template.format(dataset_fold[i] + '_BL5',
                                            *vg_common_labels)
        sd.read_data(vg_path + input_filename, vg_epochs)
        input_filename = spectral_template.format(dataset_fold[i] + '_BL5',
                                                  *spectral_common_labels)
        sd.read_data(spectral_path + input_filename, spectral_epochs)
        input_filename = timed_template.format(dataset_fold[i] + '_BL5',
                                               *timed_common_labels)
        sd.read_data(timed_path + input_filename, timed_epochs)
        eeg_epochs = [pe_epochs[i] + vg_epochs[i] + spectral_epochs[i] +
                      timed_epochs[i] for i in range(len(pe_epochs))]
        input_filename = epoch_template.format(dataset_fold[i] + '_BL5_labels',
                                               *epoch_common_labels)
        sd.read_data(epochs_path + input_filename, stage_epochs)
        train_epochs.extend(eeg_epochs)
        train_labels.extend(stage_epochs)
    # process testing set
    test_epochs = []
    test_labels = []
    for i in range(7, 11):
        pe_epochs = []
        vg_epochs = []
        spectral_epochs = []
        timed_epochs = []
        stage_epochs = []
        print('Processing testing set: ' + dataset_fold[i] + '...')
        input_filename = pe_template.format(dataset_fold[i] + '_BL5',
                                            *pe_common_labels)
        sd.read_data(pe_path + input_filename, pe_epochs)
        input_filename = vg_template.format(dataset_fold[i] + '_BL5',
                                            *vg_common_labels)
        sd.read_data(vg_path + input_filename, vg_epochs)
        input_filename = spectral_template.format(dataset_fold[i] + '_BL5',
                                                  *spectral_common_labels)
        sd.read_data(spectral_path + input_filename, spectral_epochs)
        input_filename = timed_template.format(dataset_fold[i] + '_BL5',
                                               *timed_common_labels)
        sd.read_data(timed_path + input_filename, timed_epochs)
        eeg_epochs = [pe_epochs[i] + vg_epochs[i] + spectral_epochs[i] +
                      timed_epochs[i] for i in range(len(pe_epochs))]
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
