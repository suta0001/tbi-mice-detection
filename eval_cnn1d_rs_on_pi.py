import datautil as du
import datagenerator as dg
from models import get_baseline_convolutional_encoder, build_siamese_net
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import yaml


"""Evaluate performance of trained Deep CNN1D
   (Random Sampling version)"""
# general setup parameters
# currently only supports the values below
target_names = ['SW', 'SS', 'TW', 'TS']
decimate_factor = 4
models_path = 'models'
model_files = ['siamrs14_4c_ew64_25_0_best.h5']
outfile = 'pi_4c_metrics.csv'


def predict_labels_based_on_distance(epochs):
    # avg_embeddings are for siamrs14_4c_ew64_25 only
    avg_embeddings = [[0.42478859424591064,5.238905906677246,-3.848388910293579,-0.743630588054657],
                      [0.3430764377117157,5.655649185180664,8.637845039367676,0.6439431309700012],
                      [-0.29962027072906494,-5.041609764099121,-8.075459480285645,-0.6281353235244751],
                      [-0.45240318775177,-5.406970977783203,4.55150842666626,0.8279956579208374]]
    labels = []
    for epoch in epochs:
        distances = []
        for avg_embedding in avg_embeddings:
            distances.append(np.sqrt(np.sum(np.square(epoch - avg_embedding))))
        labels.append(np.argmin(distances))
    return labels


def eval_performance(models_path, model_file):
    # set up common things
    num_classes = 4
    eeg_epoch_width_in_s = 64
    num_tsteps = eeg_epoch_width_in_s / 4 * 1024 / decimate_factor
    data_path = 'data/epochs_{}c'.format(str(num_classes))

    # load model
    filepath = os.path.join(models_path, model_file)
    # uncomment below if using tensorflow 1.14 and earlier
    # model params are for siamrs14_4c_ew64_25 only
    with tf.device('/cpu:0'):
        pmodel = get_baseline_convolutional_encoder(128, 4,
                                                    dropout=0.10,
                                                    input_shape=(num_tsteps, 1))
        pmodel = build_siamese_net(pmodel, (num_tsteps, 1),
                                   distance_metric='uni_euc_cont_loss')
        pmodel.load_weights(filepath)
        pmodel = pmodel.layers[2]
    # end comment for tensorflow 1.14
    # uncomment below if using latest tensorflow
    # pmodel = tf.keras.models.load_model(filepath)
    # end comment for latest tensorflow

    # define the test set
    file_template = '{}_BL5_' + 'ew{}.h5'.format(str(eeg_epoch_width_in_s))
    dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
    species_set = ['Sham102']#, 'TBI102']
    batch_size = 1024 * 4 * decimate_factor // eeg_epoch_width_in_s
    test_gen = dg.DataGenerator(data_path, file_template, species_set,
                                'test', batch_size, num_classes,
                                shuffle=False,
                                decimate=decimate_factor,
                                test_percent=100,
                                overlap=False)

    # get true and predicted labels
    labels = test_gen.get_labels()
    embeddings = pmodel.predict_generator(test_gen,
                                          max_queue_size=1)
    predict_labels = predict_labels_based_on_distance(embeddings)

    # evaluate performance metrics
    report = classification_report(labels, predict_labels,
                                   target_names=target_names,
                                   output_dict=True)
    acc = accuracy_score(labels, predict_labels)
    print("acc = {}".format(acc))

    print("classification_report:")
    print(report)

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
#write_reports_to_csv(model_files, reports, outfile)
