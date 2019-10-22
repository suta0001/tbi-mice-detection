import apputil as au
import datautil as du
import os


"""Generate average features from Siamese features"""
# set command-line arguments parser
parser = au.set_common_arg_parser('Generate average features\
                                  from Siamese features.')
parser.add_argument('source', default=None,
                    choices=['eeg', 'pp1', 'pp2', 'pp3', 'pp4'],
                    help='source of epochs to be processed')
parser.add_argument('featgen', default=None,
                    choices=['siamese', 'siamesers'],
                    help='feature generator')
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

# calculate average features from Siamese features of all species
dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
data_epochs, labels = du.build_dataset(epochs_path,
                                       args.num_classes,
                                       args.eeg_epoch_width_in_s,
                                       args.source,
                                       args.featgen,
                                       dataset_folds[0])
avg_features = du.calc_average_features(data_epochs, labels, args.num_classes)
du.write_data('avg_{}_features.csv'.format(args.featgen), avg_features)
