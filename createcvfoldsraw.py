import csv
import extractfeatures as ef
import sourcedata as sd
import sys

# parameters to be varied
eeg_epoch_width_in_s = int(sys.argv[2])
eeg_source = sys.argv[1]
num_classes = int(sys.argv[3])

# set up dataset folds for cross-validation
epochs_path = 'data/epochs_{0}c/'.format(str(num_classes))
cv_path = 'data/cv_raw_{0}c/'.format(str(num_classes))
dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
epoch_template = '{0}_ew{1}.csv'
epoch_common_labels = [str(eeg_epoch_width_in_s)]
cv_template = '{0}_{1}_ew{2}_f{3}_{4}.csv'

for fold, dataset_fold in enumerate(dataset_folds):
    cv_common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(fold)]
    print('Processing fold ' + str(fold) + '...')
    for i in range(len(dataset_fold)):
        if i < 7:
            data_purpose = 'train'
        else:
            data_purpose = 'test'
        print('Processing ' + data_purpose + ' set: ' + dataset_fold[i] +
              '...')
        data_filename = (epochs_path +
                         epoch_template.format(dataset_fold[i] + '_BL5_' +
                                               eeg_source,
                                               *epoch_common_labels))
        labels_filename = (epochs_path +
                           epoch_template.format(dataset_fold[i] +
                                                 '_BL5_labels',
                                                 *epoch_common_labels))
        datafile = open(data_filename, mode='r', newline='')
        labelsfile = open(labels_filename, mode='r', newline='')
        datareader = csv.reader(datafile)
        labelsreader = csv.reader(labelsfile)
        # reset file id at the beginning of training or test set
        if i == 0 or i == 7:
            id = 0
        for data, label in zip(datareader, labelsreader):
            out_filename = cv_path + cv_template.format(data_purpose,
                                                        *cv_common_labels, id)
            outfile = open(out_filename, mode='w', newline='')
            filewriter = csv.writer(outfile)
            filewriter.writerow(data + label)
            outfile.close()
            id += 1
        labelsfile.close()
        datafile.close()
