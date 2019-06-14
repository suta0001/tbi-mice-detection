import argparse


def set_common_arg_parser(description):
    """Return a common argument parser"""
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
    return parser
