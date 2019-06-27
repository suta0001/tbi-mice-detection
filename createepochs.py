import apputil as au
import concurrent.futures
import csv
import os
import datautil as du
import sys


"""Create EEG Epochs from EDF Files"""
# set command-line arguments parser
parser = au.set_common_arg_parser('Create epochs from EDF files.')
args = parser.parse_args()

# set up file location paths
edf_path = os.path.join('data', 'edf')
stage_path = os.path.join('data', 'sleep_staging')
if args.no_overlap:
    epochs_path = os.path.join('data', 'epochs_novl_{0}c'.format(
        str(args.num_classes)))
else:
    epochs_path = os.path.join('data', 'epochs_{0}c'.format(
        str(args.num_classes)))
if not os.path.isdir(epochs_path):
    os.makedirs(epochs_path)


def process_edf(edf_filename, stage_filename, eeg_epoch_width_in_s,
                num_classes, overlap, epochs_filename):
    eeg_epochs, stage_epochs = du.create_epochs(eeg_epoch_width_in_s,
                                                edf_filename, stage_filename,
                                                num_classes, overlap)
    du.write_hdf5_root_attrs(epochs_filename, eeg_epoch_width_in_s,
                             num_classes, overlap)
    du.write_data_to_hdf5(epochs_filename, 'eeg', eeg_epochs)
    du.write_data_to_hdf5(epochs_filename, 'stage', stage_epochs)

# create epochs from all EDF files
edf_files = [file for file in os.listdir(edf_path)]
with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_cpus) as executor:
    for edf_file in edf_files:
        species = edf_file.split('.')[0]
        print('Processing ' + species)
        edf_filename = os.path.join(edf_path, edf_file)
        stage_filename = os.path.join(stage_path, species + '_Stages.csv')
        template = '{0}_ew{1}.h5'
        common_labels = [str(args.eeg_epoch_width_in_s)]
        epochs_filename = os.path.join(epochs_path, template.format(
            species, *common_labels))
        executor.submit(process_edf, edf_filename, stage_filename,
                        args.eeg_epoch_width_in_s, args.num_classes,
                        not args.no_overlap, epochs_filename)
