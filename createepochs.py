import apputil as au
import concurrent.futures
import os
import datautil as du


"""Create EEG Epochs from EDF Files"""
# set command-line arguments parser
parser = au.set_common_arg_parser('Create epochs from EDF files.')
args = parser.parse_args()

# set up file location paths
edf_path = os.path.join('data', 'edf')
stage_path = os.path.join('data', 'sleep_staging')
if args.no_overlap:
    epochs_path = os.path.join('data', 'epochs_novl_{}c'.format(
        str(args.num_classes)))
else:
    epochs_path = os.path.join('data', 'epochs_{}c'.format(
        str(args.num_classes)))
if not os.path.isdir(epochs_path):
    os.makedirs(epochs_path)


def write_epochs(epochs_filename, species, num_classes,
                 eeg_epochs, stage_epochs):
    if 'Sham' in species:
        offset = 0
    else:
        offset = num_classes // 2

    if num_classes == 4:
        wake_epochs = [eeg_epochs[i] for i in range(len(stage_epochs))
                       if stage_epochs[i][0] == (offset + 0)]
        sleep_epochs = [eeg_epochs[i] for i in range(len(stage_epochs))
                        if stage_epochs[i][0] == (offset + 1)]
        du.write_data_to_hdf5(epochs_filename, 'eeg/wake', wake_epochs)
        du.write_data_to_hdf5(epochs_filename, 'eeg/sleep', sleep_epochs)
    elif num_classes == 6:
        wake_epochs = [eeg_epochs[i] for i in range(len(stage_epochs))
                       if stage_epochs[i][0] == (offset + 0)]
        nrem_epochs = [eeg_epochs[i] for i in range(len(stage_epochs))
                       if stage_epochs[i][0] == (offset + 1)]
        rem_epochs = [eeg_epochs[i] for i in range(len(stage_epochs))
                      if stage_epochs[i][0] == (offset + 2)]
        du.write_data_to_hdf5(epochs_filename, 'eeg/wake', wake_epochs)
        du.write_data_to_hdf5(epochs_filename, 'eeg/nrem', nrem_epochs)
        du.write_data_to_hdf5(epochs_filename, 'eeg/rem', rem_epochs)
    else:
        du.write_data_to_hdf5(epochs_filename, 'eeg/data', eeg_epochs)


def process_edf(edf_path, edf_file, eeg_epoch_width_in_s,
                num_classes, overlap, epochs_path):
    species = edf_file.split('.')[0]
    print('Processing ' + species)
    edf_filename = os.path.join(edf_path, edf_file)
    stage_filename = os.path.join(stage_path, species + '_Stages.csv')
    template = '{}_ew{}.h5'
    common_labels = [str(args.eeg_epoch_width_in_s)]
    epochs_filename = os.path.join(epochs_path, template.format(
        species, *common_labels))
    eeg_epochs, stage_epochs = du.create_epochs(eeg_epoch_width_in_s,
                                                edf_filename, stage_filename,
                                                num_classes, overlap)
    du.write_attrs_to_hdf5(epochs_filename,
                           eeg_epoch_width_in_s=eeg_epoch_width_in_s,
                           num_classes=num_classes,
                           overlap=overlap)
    write_epochs(epochs_filename, species, num_classes, eeg_epochs,
                 stage_epochs)


# create epochs from all EDF files
edf_files = [file for file in os.listdir(edf_path)]
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_cpus) as executor:
        for edf_file in edf_files:
            executor.submit(process_edf, edf_path, edf_file,
                            args.eeg_epoch_width_in_s, args.num_classes,
                            not args.no_overlap, epochs_path)
