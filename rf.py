import csv
import extractfeatures as ef
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.validation import column_or_1d
import sourcedata as sd
import statistics
import sys


# parameters to be varied
eeg_epoch_width_in_s = int(sys.argv[2])
feature_type = sys.argv[3]
pe_orders = list(range(3, 8))
pe_delays = list(range(1, 11))
max_degree = 11
eeg_source = sys.argv[1]
num_classes = int(sys.argv[4])
if num_classes == 2:
    target_names = ['Sham', 'TBI']
elif num_classes == 4:
    target_names = ['SW', 'SS', 'TW', 'TS']
elif num_classes == 6:
    target_names = ['SW', 'SN', 'SR', 'TW', 'TN', 'TR']

# read dataset per fold
accuracies = []
reports = []
template = None
common_labels = None

for fold in range(10):
    common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(fold)]
    cv_path = 'data/cv_{0}c/'.format(str(num_classes))
    if feature_type == 'pe':
        template = '{0}_{1}_ew{2}_f{3}_{4}t{5}_{6}t{7}.csv'
        common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(fold),
                         str(pe_orders[0]), str(pe_orders[-1]),
                         str(pe_delays[0]), str(pe_delays[-1])]
    elif feature_type == 'vg':
        template = '{0}_{1}_ew{2}_f{3}_{4}.csv'
        common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(fold),
                         str(max_degree)]
    elif feature_type == 'sp':
        template = '{0}_{1}_ew{2}_f{3}_sp.csv'
    elif feature_type == 'ti':
        template = '{0}_{1}_ew{2}_f{3}_ti.csv'
    elif feature_type == 'sxx':
        template = '{0}_{1}_ew{2}_f{3}_sxx.csv'
    elif feature_type == 'wpe':
        template = '{0}_{1}_ew{2}_f{3}_wpe.csv'
    elif feature_type == 'all':
        template = '{0}_{1}_ew{2}_f{3}_{4}t{5}_{6}t{7}_{8}_sp_ti.csv'
        common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(fold),
                         str(pe_orders[0]), str(pe_orders[-1]),
                         str(pe_delays[0]), str(pe_delays[-1]),
                         str(max_degree)]
    train_epochs = []
    train_labels = []
    test_epochs = []
    test_labels = []
    input_filename = template.format('train_data', *common_labels)
    sd.read_data(cv_path + input_filename, train_epochs)
    input_filename = template.format('train_labels', *common_labels)
    sd.read_data(cv_path + input_filename, train_labels)
    input_filename = template.format('test_data', *common_labels)
    sd.read_data(cv_path + input_filename, test_epochs)
    input_filename = template.format('test_labels', *common_labels)
    sd.read_data(cv_path + input_filename, test_labels)

    # convert datasets to numpy arrays
    train_epochs = np.array(train_epochs)
    train_labels = column_or_1d(np.array(train_labels, dtype=int))
    test_epochs = np.array(test_epochs)
    test_labels = column_or_1d(np.array(test_labels, dtype=int))

    # define classifier
    clf = RandomForestClassifier(n_estimators=100)

    # train classifier
    clf.fit(train_epochs, train_labels)

    # make prediction
    predict_labels = clf.predict(test_epochs)

    # calculate accuracy score
    accuracy = accuracy_score(test_labels, predict_labels)
    accuracies.append(accuracy)
    print('Fold = ' + str(fold) + ' Accuracy = ' + str(accuracy))

    # calculate confusion matrix
    print(confusion_matrix(test_labels, predict_labels))

    # print report
    report = classification_report(test_labels, predict_labels,
                                   target_names=target_names,
                                   output_dict=True)
    reports.append(report)
    print(classification_report(test_labels, predict_labels,
                                target_names=target_names))

# print out results summary
print('Mean  accuracy = ' + str(statistics.mean(accuracies)))
print('Stdev accuracy = ' + str(statistics.stdev(accuracies)))
for name in target_names:
    precisions = [reports[i][name]['precision'] for i in range(len(reports))]
    print('Mean  prec ' + name + '  = ' + str(statistics.mean(precisions)))
    print('Stdev prec ' + name + '  = ' + str(statistics.stdev(precisions)))
for name in target_names:
    recalls = [reports[i][name]['recall'] for i in range(len(reports))]
    print('Mean  recll ' + name + ' = ' + str(statistics.mean(recalls)))
    print('Stdev recll ' + name + ' = ' + str(statistics.stdev(recalls)))
