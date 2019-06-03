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
cv_path = 'data/cv_{0}c/'.format(str(num_classes))
dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
epoch_template = '{0}_ew{1}.csv'
epoch_common_labels = [str(eeg_epoch_width_in_s)]
cv_template = '{0}_{1}_ew{2}_f{3}.csv'

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
        out_filename = cv_path + cv_template.format(data_purpose + '_data',
                                                    *cv_common_labels)
        outfile = open(out_filename, mode='w', newline='')
        filewriter = csv.writer(outfile)
        out_filename = cv_path + cv_template.format(data_purpose + '_labels',
                                                    *cv_common_labels)
        loutfile = open(out_filename, mode='w', newline='')
        labelswriter = csv.writer(loutfile)
        for data, label in zip(datareader, labelsreader):
            filewriter.writerow(data)
            labelswriter.writerow(label)
        loutfile.close()
        outfile.close()
        labelsfile.close()
        datafile.close()
