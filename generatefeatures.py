import apputil as au
import concurrent.futures
import datautil as du
import featureextraction as fe
import os


"""Generate features"""
# set command-line arguments parser
parser = au.set_common_arg_parser('Generate features.')
parser.add_argument('source', default=None,
                    choices=['eeg', 'pp1', 'pp2', 'pp3', 'pp4'],
                    help='source of epochs to be processed')
parser.add_argument('featgen', default=None,
                    choices=['pe', 'vg', 'spectral', 'timed', 'wpe', 'siamese',
                             'siamesers', 'siamdist', 'siamrsdist',
                             'pe_spectral', 'wpe_spectral'],
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


def process_file(path, file):
    filename = os.path.join(path, file)
    groups = du.read_groups_from_hdf5(filename, args.source)
    for group in groups:
        fgroup = '{}/{}'.format(args.source, group)
        tgroup = '{}_{}/{}'.format(args.source, args.featgen, group)
        print('Processing {}: group = {} to group = {}'.format(filename,
              fgroup, tgroup))
        epochs = du.read_data_from_hdf5(filename, fgroup)
        epochs = fe.process(epochs, args.featgen)
        du.write_data_to_hdf5(filename, tgroup, epochs)


# preprocess epochs from all files
files = [file for file in os.listdir(epochs_path) if
         '_ew{}.h5'.format(args.eeg_epoch_width_in_s) in file]
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_cpus) as executor:
        for file in files:
            executor.submit(process_file, epochs_path, file)
