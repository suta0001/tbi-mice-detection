import argparse
import concurrent.futures
import csv
import os
import sourcedata as sd
import sys


"""Create EEG Epochs from EDF Files"""
# set parameters based on command-line arguments
parser = argparse.ArgumentParser(
    description='Create epochs from EDF files.')
parser.add_argument('eeg_epoch_width_in_s', default=32, type=int,
                    choices=[4, 8, 16, 32, 64, 128],
                    help='EEG epoch width in seconds')
parser.add_argument('num_classes', default=4, type=int, choices=[2, 4, 6],
                    help='number of classes')
parser.add_argument('--no_overlap', action='store_true',
                    help='create non-overlapping epochs')
parser.add_argument('--num_cpus', default=1, type=int,
                    help='number of CPUs for parallel processes')
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
                num_classes, overlap, eeg_epochs_filename,
                stage_epochs_filename):
    eeg_epochs, stage_epochs = sd.create_epochs(eeg_epoch_width_in_s,
                                                edf_filename, stage_filename,
                                                num_classes, overlap)
    sd.write_data(eeg_epochs_filename, eeg_epochs)
    sd.write_data(stage_epochs_filename, stage_epochs)

# create epochs from all EDF files
edf_files = [file for file in os.listdir(edf_path)]
with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_cpus) as executor:
    for edf_file in edf_files:
        species = edf_file.split('.')[0]
        print('Processing ' + species)
        edf_filename = os.path.join(edf_path, edf_file)
        stage_filename = os.path.join(stage_path, species + '_Stages.csv')
        template = '{0}_ew{1}.csv'
        common_labels = [str(args.eeg_epoch_width_in_s)]
        eeg_epochs_filename = os.path.join(epochs_path, template.format(
            species + '_eeg', *common_labels))
        stage_epochs_filename = os.path.join(epochs_path, template.format(
            species + '_labels', *common_labels))
        executor.submit(process_edf, edf_filename, stage_filename,
                        args.eeg_epoch_width_in_s, args.num_classes,
                        not args.no_overlap, eeg_epochs_filename,
                        stage_epochs_filename)
