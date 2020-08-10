import apputil as au
import datautil as du
import models
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import statistics


"""Train and cross-validate machine learning model"""
# set command-line arguments parser
parser = au.set_common_arg_parser('Train and cross-validate model')
parser.add_argument('pp_step', default=None,
                    choices=['eeg', 'pp1', 'pp2', 'pp3', 'pp4', 'pp5'],
                    help='applied preprocessing step')
parser.add_argument('featgen', default=None,
                    choices=['pe', 'vg', 'spectral', 'timed', 'wpe', 'siamese',
                             'siamesers', 'siamdist', 'siamrsdist',
                             'pe_spectral', 'wpe_spectral'],
                    help='applied feature generator')
parser.add_argument('model', default=None,
                    choices=['ffnn3hl', 'knn', 'rf', 'xgb'],
                    help='machine-learning model')
args = parser.parse_args()

# set up file location paths
epochs_novl_path = os.path.join('data', 'epochs_novl_{0}c'.format(
    str(args.num_classes)))
epochs_path = os.path.join('data', 'epochs_{0}c'.format(
    str(args.num_classes)))

# train and validate using cross-validation folds
accuracies = []
reports = []
template = None
common_labels = None
dataset_folds = [line.rstrip().split(',') for line in open('cv_folds3.txt')]
for fold, dataset_fold in enumerate(dataset_folds):
    train_epochs, train_labels = du.build_dataset(epochs_path,
                                                  args.num_classes,
                                                  args.eeg_epoch_width_in_s,
                                                  args.pp_step,
                                                  args.featgen,
                                                  dataset_fold[0:9])
    test_epochs, test_labels = du.build_dataset(epochs_novl_path,
                                                args.num_classes,
                                                args.eeg_epoch_width_in_s,
                                                args.pp_step,
                                                args.featgen,
                                                dataset_fold[9:])
    # if using normalized spectral feature, transform spectral powers into
    # normalized decibel values
    if 'spectral' in args.featgen:
        baselines = du.calc_baseline_spectral_powers(epochs_path,
                                                     args.num_classes,
                                                     args.eeg_epoch_width_in_s,
                                                     args.pp_step,
                                                     dataset_fold[0:9])
        train_epochs = du.decibel_normalize(args.featgen, baselines, train_epochs,
                                            train_labels)
        test_epochs = du.decibel_normalize(args.featgen, baselines, test_epochs,
                                           test_labels)

    # normalize across features
    normalizer = StandardScaler()
    normalizer.fit(train_epochs)
    train_epochs = normalizer.transform(train_epochs)
    test_epochs = normalizer.transform(test_epochs)

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
outfile = 'metrics/{}mx_{}c_ew{}_{}_{}_metrics.csv'
moutfile = 'metrics/{}mx_{}c_ew{}_{}_{}_avg_metrics.csv'
soutfile = 'metrics/{}mx_{}c_ew{}_{}_{}_std_metrics.csv'
outfile = outfile.format(args.model, args.num_classes,
                         args.eeg_epoch_width_in_s, args.pp_step, args.featgen)
moutfile = moutfile.format(args.model, args.num_classes,
                           args.eeg_epoch_width_in_s, args.pp_step,
                           args.featgen)
soutfile = soutfile.format(args.model, args.num_classes,
                           args.eeg_epoch_width_in_s, args.pp_step,
                           args.featgen)
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
mmetric_values = [args.model, args.num_classes, args.eeg_epoch_width_in_s,
                  not args.no_overlap, 'all', args.pp_step, args.featgen,
                  statistics.mean(accuracies)]
smetric_values = [args.model, args.num_classes, args.eeg_epoch_width_in_s,
                  not args.no_overlap, 'all', args.pp_step, args.featgen,
                  statistics.stdev(accuracies)]
for label in target_names:
    for metric in metrics:
        values = [reports[i][label][metric] for i in range(len(reports))]
        mmetric_values.append(statistics.mean(values))
        smetric_values.append(statistics.stdev(values))
moutputs.append(mmetric_values)
soutputs.append(smetric_values)
du.write_data(moutfile, moutputs)
du.write_data(soutfile, soutputs)
