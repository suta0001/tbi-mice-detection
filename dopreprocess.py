import apputil as au
import concurrent.futures
import datautil as du
import os
import preprocess as pp


"""Preprocess EEG Epochs"""
# set command-line arguments parser
parser = au.set_common_arg_parser('Apply preprocessing steps.')
parser.add_argument('pp_set', default='pp4',
                    choices=['pp1', 'pp2', 'pp3', 'pp4'],
                    help='set of preprocessing steps')
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


def process_file(path, file):
    filename = os.path.join(path, file)
    groups = du.read_groups_from_hdf5(filename, 'eeg')
    for group in groups:
        fgroup = 'eeg/{}'.format(group)
        tgroup = '{}/{}'.format(args.pp_set, group)
        print('Processing {}: group = {} to group = {}'.format(filename,
              fgroup, tgroup))
        epochs = du.read_data_from_hdf5(filename, fgroup)
        epochs = pp.process(epochs, args.pp_set)
        du.write_data_to_hdf5(filename, tgroup, epochs)


# preprocess epochs from all files
files = [file for file in os.listdir(epochs_path) if
         '_ew{}.h5'.format(args.eeg_epoch_width_in_s) in file]
with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_cpus) as executor:
    for file in files:
        executor.submit(process_file, epochs_path, file)
