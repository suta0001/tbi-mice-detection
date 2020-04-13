import datautil as du
from featureextraction import generate_embeddings
import h5py
import numpy as np
import os
import pandas as pd
from scipy.signal import decimate
from sklearn.metrics import classification_report
import yaml


"""Evaluate performance of trained Siamese embedding generator"""
# general setup parameters
# currently only supports the values below
eeg_epoch_width_in_s = 32
num_classes = 4
target_names = ['SW', 'SS', 'TW', 'TS']
decimate_factor = 4
batch_size = 1024 * 4 * decimate_factor // eeg_epoch_width_in_s
data_path = 'data/epochs_{}c'.format(str(num_classes))
models_path = 'models'
# config_files = ['basesiam_4c_ew32_25.yaml',
#                 'siam1_4c_ew32_25.yaml',
#                 'siam2_4c_ew32_25.yaml',
#                 'siam3_4c_ew32_25.yaml',
#                 'siam4_4c_ew32_25.yaml',
#                 'siam5_4c_ew32_25.yaml',
#                 'siam6_4c_ew32_25.yaml',
#                 'siam7_4c_ew32_25.yaml',
#                 'siam8_4c_ew32_25.yaml',
#                 'siam9_4c_ew32_25.yaml',
#                 'siam10_4c_ew32_25.yaml',
#                 'siam11_4c_ew32_25.yaml',
#                 'siam12_4c_ew32_25.yaml',
#                 'siam13_4c_ew32_25.yaml']
config_files = ['siam12_4c_ew32_25.yaml']
train_outfile = 'train_4c_ew32_metrics.csv'
test_outfile = 'test_4c_ew32_metrics.csv'


def predict_labels_based_on_distance(epochs, avg_embeddings):
    labels = []
    for epoch in epochs:
        distances = []
        for avg_embedding in avg_embeddings:
            distances.append(np.sqrt(np.sum(np.square(epoch - avg_embedding))))
        labels.append(np.argmin(distances))
    return labels


def eval_performance(models_path, config_file):
    # read EEG epochs and labels used in training Siamese network
    # set up pandas dataframe from pairdata HDF5 file
    model_file = os.path.join(models_path, config_file)
    with open(model_file) as cfile:
        config_params = yaml.safe_load(cfile)
    num_samples = config_params['num_train_samples']
    if config_params['overlap']:
        pairdata_file = 'pairdata_BL5_ew{}_{}_{}_{}.h5'
    else:
        pairdata_file = 'pairdata_novl_BL5_ew{}_{}_{}_{}.h5'
    pairdata_file = pairdata_file.format(eeg_epoch_width_in_s,
                                         num_classes,
                                         batch_size,
                                         num_samples)
    df = pd.read_hdf(pairdata_file, 'pair_index', mode='r')
    df = df[0:num_samples]
    # get unique list of EEG epochs to read from files
    df0 = df[['species0', 'stage0', 'index0']]
    df0.columns = ['species', 'stage', 'index']
    df1 = df[['species1', 'stage1', 'index1']]
    df1.columns = ['species', 'stage', 'index']
    df = pd.concat([df0, df1], axis=0, ignore_index=True)
    del df0, df1
    df = df.drop_duplicates()
    # generate list of training EEG epochs and classes (labels)
    eeg_epochs = []
    labels = []
    file_template = '{}_BL5_' + 'ew{}.h5'.format(eeg_epoch_width_in_s)
    for index, row in df.iterrows():
        species = row['species']
        stage = row['stage']
        idx = row['index']
        data_file = os.path.join(data_path,
                                 file_template.format(species))
        with h5py.File(data_file, 'r') as dfile:
            eeg_epochs.append(dfile['eeg'][stage][idx])
        labels.append(du.get_class_label(num_classes, species, stage))

    # generate embeddings using trained models
    eeg_epochs = generate_embeddings(decimate(eeg_epochs, decimate_factor),
                                     model_file)

    # calculate average embeddings for all classes
    avg_embeddings = du.calc_average_features(eeg_epochs, labels, num_classes)
    du.write_data('{}_avg_emb.csv'.format(config_file), avg_embeddings)

    # read all EEG epochs and labels for both training and testing sets
    dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
    train_epochs, train_labels = du.build_dataset(data_path,
                                                  num_classes,
                                                  eeg_epoch_width_in_s,
                                                  'eeg',
                                                  None,
                                                  dataset_folds[0][0:7])
    test_epochs, test_labels = du.build_dataset(data_path,
                                                num_classes,
                                                eeg_epoch_width_in_s,
                                                'eeg',
                                                None,
                                                dataset_folds[0][7:11])

    # generate embeddings for all EEG epochs using trained models
    train_epochs = generate_embeddings(decimate(train_epochs, decimate_factor),
                                       model_file)
    test_epochs = generate_embeddings(decimate(test_epochs, decimate_factor),
                                      model_file)

    # set predicted class based on minimum distance to average embeddings
    predict_train_labels = predict_labels_based_on_distance(train_epochs,
                                                            avg_embeddings)
    predict_test_labels = predict_labels_based_on_distance(test_epochs,
                                                           avg_embeddings)

    # evaluate performance metrics
    train_report = classification_report(train_labels, predict_train_labels,
                                         target_names=target_names,
                                         output_dict=True)
    test_report = classification_report(test_labels, predict_test_labels,
                                        target_names=target_names,
                                        output_dict=True)

    return train_report, test_report


def write_reports_to_csv(models, reports, outfile):
    metrics = ['precision', 'recall', 'f1-score', 'support']
    outputs = []
    # form array of header labels and add to outputs
    header_labels = ['model', 'accuracy']
    for label in target_names:
        for metric in metrics:
            header_labels.append('{}_{}'.format(label, metric))
    outputs.append(header_labels)
    # form array of metric values and add to outputs
    for model, report in zip(models, reports):
        metric_values = [model, report['accuracy']]
        for label in target_names:
            for metric in metrics:
                metric_values.append(report[label][metric])
        outputs.append(metric_values)
    du.write_data(outfile, outputs)


train_reports = []
test_reports = []
for config_file in config_files:
    train_report, test_report = eval_performance(models_path, config_file)
    train_reports.append(train_report)
    test_reports.append(test_report)
write_reports_to_csv(config_files, train_reports, train_outfile)
write_reports_to_csv(config_files, test_reports, test_outfile)
