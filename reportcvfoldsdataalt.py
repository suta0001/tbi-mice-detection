import csv
import extractfeatures as ef
import sourcedata as sd

# parameters to be varied
eeg_epoch_width_in_s = 4

# set up dataset folds for cross-validation
epochs_path = 'data/epochs/'
dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
epoch_template = '{0}_ew{1}.csv'
epoch_common_labels = [str(eeg_epoch_width_in_s)]
cv_template = 'label_data_ew{0}.csv'
cv_common_labels = [str(eeg_epoch_width_in_s)]

label_counts = []
for fold, dataset_fold in enumerate(dataset_folds):

    # process training set
    train_labels = []
    print('Processing fold ' + str(fold) + '...')
    for i in range(0, 7):
        stage_epochs = []
        print('Processing training set: ' + dataset_fold[i] + '...')
        input_filename = epoch_template.format(dataset_fold[i] +
                                               '_BL5_labels',
                                               *epoch_common_labels)
        sd.read_data(epochs_path + input_filename, stage_epochs)
        train_labels.extend(stage_epochs)
    # process testing set
    test_labels = []
    for i in range(7, 11):
        stage_epochs = []
        print('Processing testing set: ' + dataset_fold[i] + '...')
        input_filename = epoch_template.format(dataset_fold[i] +
                                               '_BL5_labels',
                                               *epoch_common_labels)
        sd.read_data(epochs_path + input_filename, stage_epochs)
        test_labels.extend(stage_epochs)

    # calculate label counts
    train_labels = [item for sublist in train_labels for item in sublist]
    test_labels = [item for sublist in test_labels for item in sublist]
    train_label_counts = []
    test_label_counts = []
    for i in range(4):
        train_label_counts.append(train_labels.count(i))
        test_label_counts.append(test_labels.count(i))
    label_counts.append([fold] + train_label_counts + test_label_counts)

    # write label counts to file
    print('Writing to file...')
    output_filename = cv_template.format(*cv_common_labels)
    sd.write_data(output_filename, label_counts)
