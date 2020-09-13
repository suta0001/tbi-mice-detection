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
target_names = ['Sham', 'TBI']
models_path = 'models'
# config_files = ['cnn1drs0_4c_ew16_50.yaml',
#                 'cnn1drs0_4c_ew32_50.yaml',
#                 'cnn1drs0_4c_ew64_50.yaml',
#                 'cnn1drs1_4c_ew64_50.yaml',
#                 'cnn1drs2_4c_ew64_50.yaml',
#                 'cnn1drs3_4c_ew64_50.yaml',
#                 'cnn1drs4_4c_ew64_50.yaml',
#                 'cnn1drs5_4c_ew64_50.yaml',
#                 'cnn1drs6_4c_ew64_50.yaml',
#                 'cnn1drs7_4c_ew64_50.yaml',
#                 'cnn1drs8_4c_ew64_50.yaml',
#                 'cnn1drs9_4c_ew64_50.yaml',
#                 'cnn1drs10_4c_ew64_50.yaml',
#                 'cnn1drs11_4c_ew64_50.yaml',
#                 'cnn1drs12_4c_ew64_50.yaml',
#                 'cnn1drs13_4c_ew64_50.yaml',
#                 'cnn1drs14_4c_ew64_50.yaml']
config_files = ['cnn1drs16_4c_ew64_50.yaml']
outfile = 'metrics/cnn1drs_4c_2c_metrics.csv'
woutfile = 'metrics/cnn1drs_4c_2cw_metrics.csv'
soutfile = 'metrics/cnn1drs_4c_2cs_metrics.csv'


def process_into_2c(labels):
    labels_2c = []
    for label in labels:
        if label == 0 or label == 1:
            labels_2c.append(0)
        elif label == 2 or label == 3:
            labels_2c.append(1)
    return labels_2c


def select_labels(true_labels, predict_labels, stage='wake'):
    sel_true_labels = []
    sel_predict_labels = []
    for i in range(len(true_labels)):
        if stage == 'wake' and true_labels[i] % 2 == 0:
            sel_true_labels.append(true_labels[i])
            sel_predict_labels.append(predict_labels[i])
        elif stage == 'sleep' and true_labels[i] % 2 == 1:
            sel_true_labels.append(true_labels[i])
            sel_predict_labels.append(predict_labels[i])
    return sel_true_labels, sel_predict_labels


def eval_performance(models_path, config_file):
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
                            '{}_0_best.h5'.format(config_file[:-5]))
    pmodel = tf.keras.models.load_model(filepath)

    # define the test set
    file_template = '{}_BL5_' + 'ew{}.h5'.format(str(eeg_epoch_width_in_s))
    dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
    species_set = dataset_folds[0]
    decimate_factor = config_params['decimate']
    batch_size = 1024 * 4 * decimate_factor // eeg_epoch_width_in_s
    test_gen = dg.DataGenerator(data_path, file_template, species_set,
                                'test', batch_size, num_classes,
                                shuffle=False,
                                decimate=decimate_factor,
                                test_percent=config_params['test_percent'],
                                overlap=config_params['overlap'],
                                num_samples=config_params['num_samples'])

    # get true and predicted labels
    labels = test_gen.get_labels()
    predict_labels = pmodel.predict_generator(test_gen,
                                              max_queue_size=1)
    predict_labels = predict_labels.argmax(axis=1)
    wlabels, predict_wlabels = select_labels(labels, predict_labels, 'wake')
    slabels, predict_slabels = select_labels(labels, predict_labels, 'sleep')
    labels = process_into_2c(labels)
    predict_labels = process_into_2c(predict_labels)
    wlabels = process_into_2c(wlabels)
    predict_wlabels = process_into_2c(predict_wlabels)
    slabels = process_into_2c(slabels)
    predict_slabels = process_into_2c(predict_slabels)

    # evaluate performance metrics
    report = classification_report(labels, predict_labels,
                                   target_names=target_names,
                                   output_dict=True)
    wake_report = classification_report(wlabels, predict_wlabels,
                                        target_names=target_names,
                                        output_dict=True)
    sleep_report = classification_report(slabels, predict_slabels,
                                         target_names=target_names,
                                         output_dict=True)
    return report, wake_report, sleep_report


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
wreports = []
sreports = []
for config_file in config_files:
    report, wreport, sreport = eval_performance(models_path, config_file)
    reports.append(report)
    wreports.append(wreport)
    sreports.append(sreport)
write_reports_to_csv(config_files, reports, outfile)
write_reports_to_csv(config_files, wreports, woutfile)
write_reports_to_csv(config_files, sreports, soutfile)
