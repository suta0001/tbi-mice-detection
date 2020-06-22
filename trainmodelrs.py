import apputil as au
import datautil as du
import models
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import statistics


"""Train and cross-validate machine learning model"""
# set command-line arguments parser
parser = au.set_common_arg_parser('Train and cross-validate model')
parser.add_argument('pp_step', default=None,
                    choices=['eeg', 'pp1', 'pp2', 'pp3', 'pp4'],
                    help='applied preprocessing step')
parser.add_argument('featgen', default=None,
                    choices=['pe', 'vg', 'spectral', 'timed', 'wpe', 'siamese',
                             'siamesers', 'siamdist', 'siamrsdist',
                             'pe_spectral'],
                    help='applied feature generator')
parser.add_argument('model', default=None,
                    choices=['ffnn3hl', 'knn', 'rf', 'xgb'],
                    help='machine-learning model')
args = parser.parse_args()

# set up file location paths
if args.no_overlap:
    epochs_path = os.path.join('data', 'epochs_novl_{0}c'.format(
        str(args.num_classes)))
else:
    epochs_path = os.path.join('data', 'epochs_{0}c'.format(
        str(args.num_classes)))
if not os.path.isdir(epochs_path):
    os.makedirs(epochs_path)

# train and validate using cross-validation folds
accuracies = []
reports = []
template = None
common_labels = None
dataset_folds = [line.rstrip().split(',') for line in open('cv_folds2.txt')]
epochs, labels = du.build_dataset(epochs_path,
                                  args.num_classes,
                                  args.eeg_epoch_width_in_s,
                                  args.pp_step,
                                  args.featgen,
                                  dataset_folds[0])
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
fold = 0
for train_index, test_index in sss.split(epochs, labels):
    train_epochs, train_labels = epochs[train_index], labels[train_index]
    test_epochs, test_labels = epochs[test_index], labels[test_index]

    # define classifier
    clf = models.get_ml_model(args.model)

    # train classifier
    if args.model == 'ffnn3hl':
        clf.fit(train_epochs, train_labels, 16, epochs=50, verbose=0)
    else:
        clf.fit(train_epochs, train_labels)

    # make prediction
    predict_labels = clf.predict(test_epochs)
    if args.model == 'ffnn3hl':
        predict_labels = predict_labels.argmax(axis=1)
    # calculate accuracy score
    accuracy = accuracy_score(test_labels, predict_labels)
    accuracies.append(accuracy)
    print('Fold = ' + str(fold) + ' Accuracy = ' + str(accuracy))

    # calculate confusion matrix
    print(confusion_matrix(test_labels, predict_labels))

    # define class labels
    if args.num_classes == 2:
        target_names = ['Sham', 'TBI']
    elif args.num_classes == 4:
        target_names = ['SW', 'SS', 'TW', 'TS']
    elif args.num_classes == 6:
        target_names = ['SW', 'SN', 'SR', 'TW', 'TN', 'TR']

    # print report
    report = classification_report(test_labels, predict_labels,
                                   target_names=target_names,
                                   output_dict=True)
    reports.append(report)
    print(classification_report(test_labels, predict_labels,
                                target_names=target_names))
    fold += 1

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

# write to file
if args.no_overlap:
    outfile = 'metrics/{}rs_novl_{}c_ew{}_{}_{}_metrics.csv'
else:
    outfile = 'metrics/{}rs_{}c_ew{}_{}_{}_metrics.csv'
outfile = outfile.format(args.model, args.num_classes,
                         args.eeg_epoch_width_in_s, args.pp_step, args.featgen)
metrics = ['precision', 'recall', 'f1-score', 'support']
outputs = []
# form array of header labels and add to outputs
header_labels = ['fold', 'accuracy']
for label in target_names:
    for metric in metrics:
        header_labels.append('{}_{}'.format(label, metric))
outputs.append(header_labels)
# form array of metric values and add to outputs
for i in range(len(dataset_folds)):
    metric_values = [i, reports[i]['accuracy']]
    for label in target_names:
        for metric in metrics:
            metric_values.append(reports[i][label][metric])
    outputs.append(metric_values)
du.write_data(outfile, outputs)