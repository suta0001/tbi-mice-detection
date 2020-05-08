import datautil as du
import datagenerator as dg
import os
import tensorflow as tf
from sklearn.metrics import classification_report
import yaml


"""Evaluate performance of trained Deep CNN1D
   (Random Sampling version)"""
# general setup parameters
# currently only supports the values below
target_names = ['SW', 'SS', 'TW', 'TS']
decimate_factor = 4
models_path = 'models'
model_files = ['model.h5']
outfile = 'cnn1drs_4c_metrics.csv'


def eval_performance(models_path, model_file):
    # set up common things
    num_classes = 4
    eeg_epoch_width_in_s = 64
    data_path = 'data/epochs_novl_{}c'.format(str(num_classes))
    
    # load model
    filepath = os.path.join(models_path, model_file)
    pmodel = tf.keras.models.load_model(filepath)

    # define the test set
    file_template = '{}_BL5_' + 'ew{}.h5'.format(str(eeg_epoch_width_in_s))
    dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
    species_set = dataset_folds[0][0]
    batch_size = 1024 * 4 * decimate_factor // eeg_epoch_width_in_s
    test_gen = dg.DataGenerator(data_path, file_template, species_set,
                                'test', batch_size, num_classes,
                                shuffle=False,
                                decimate=decimate_factor,
                                test_percent=100,
                                overlap=False)

    # get true and predicted labels
    labels = test_gen.get_labels()
    predict_labels = pmodel.predict_generator(test_gen,
                                              max_queue_size=1)
    predict_labels = predict_labels.argmax(axis=1)

    # evaluate performance metrics
    report = classification_report(labels, predict_labels,
                                   target_names=target_names,
                                   output_dict=True)
    return report


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


reports = []
for model_file in model_files:
    report = eval_performance(models_path, model_file)
    reports.append(report)
write_reports_to_csv(model_files, reports, outfile)
