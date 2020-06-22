import datautil as du
import datagenerator as dg
import os
import tensorflow as tf
from sklearn.metrics import classification_report
import yaml


"""Evaluate performance of trained Deep CNN1D"""
# general setup parameters
# currently only supports the values below
target_names = ['SW', 'SS', 'TW', 'TS']
decimate_factor = 4
models_path = 'models'
config_files = ['cnn1d0_4c_ew64_50.yaml',
                'cnn1d1_4c_ew64_50.yaml',
                'cnn1d2_4c_ew64_50.yaml',
                'cnn1d3_4c_ew64_50.yaml',
                'cnn1d4_4c_ew64_50.yaml',
                'cnn1d5_4c_ew64_50.yaml',
                'cnn1d6_4c_ew64_50.yaml',
                'cnn1d7_4c_ew64_50.yaml',
                'cnn1d8_4c_ew64_50.yaml',
                'cnn1d9_4c_ew64_50.yaml']
outfile = 'cnn1d_4c_metrics.csv'


def eval_performance(models_path, config_file, fold):
    # set up common things
    model_file = os.path.join(models_path, config_file)
    with open(model_file) as cfile:
        config_params = yaml.safe_load(cfile)
    num_classes = config_params['num_classes']
    eeg_epoch_width_in_s = config_params['epoch_width']
    if config_params['overlap']:
        data_path = 'data/epochs_{}c'.format(str(num_classes))
    else:
        data_path = 'data/epochs_novl_{}c'.format(str(num_classes))

    # load model
    filepath = os.path.join(models_path,
                            '{}_{}_best.h5'.format(config_file[:-5], fold))
    pmodel = tf.keras.models.load_model(filepath)

    # define the test set
    file_template = '{}_BL5_' + 'ew{}.h5'.format(str(eeg_epoch_width_in_s))
    dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
    species_set = dataset_folds[fold][7:]
    batch_size = 1024 * 4 * decimate_factor // eeg_epoch_width_in_s
    test_gen = dg.DataGenerator(data_path, file_template, species_set,
                                'test', batch_size, num_classes,
                                shuffle=False,
                                decimate=decimate_factor,
                                test_percent=100,
                                overlap=config_params['overlap'])

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


def write_reports_to_csv(models, folds, reports, outfile):
    metrics = ['precision', 'recall', 'f1-score', 'support']
    outputs = []
    # form array of header labels and add to outputs
    header_labels = ['model', 'fold', 'accuracy']
    for label in target_names:
        for metric in metrics:
            header_labels.append('{}_{}'.format(label, metric))
    outputs.append(header_labels)
    # form array of metric values and add to outputs
    for i in range(len(models)):
        metric_values = [models[i], folds[i], reports[i]['accuracy']]
        for label in target_names:
            for metric in metrics:
                metric_values.append(reports[i][label][metric])
        outputs.append(metric_values)
    du.write_data(outfile, outputs)


reports = []
folds = []
models = []
for config_file in config_files:
    files = [file for file in os.listdir(models_path) if
             config_file[:-5] in file and '_best' in file]
    for file in files:
        fold = int(file.split('_')[4])
        models.append(config_file)
        folds.append(fold)
        report = eval_performance(models_path, config_file, fold)
        reports.append(report)
write_reports_to_csv(models, folds, reports, outfile)
