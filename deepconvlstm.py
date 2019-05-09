import csv
import math
import numpy as np
import os
import random
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import sourcedata as sd
import statistics
import sys
import tensorflow as tf
from time import time
import yaml


def get_num_samples(cv_raw_path, purpose='train', eeg_source='eeg',
                    eeg_epoch_width_in_s=4, fold=0):
    files = [file for file in os.listdir(cv_raw_path) if
             '{0}_{1}_ew{2}_f{3}'.format(purpose, eeg_source,
                                         str(eeg_epoch_width_in_s), str(fold))
             in file]
    return len(files)


def generate_arrays_from_file(cv_raw_path, purpose='train', eeg_source='eeg',
                              eeg_epoch_width_in_s=4, fold=0, num_classes=4,
                              batch_size=32, shuffle=True):
    to_cat = tf.keras.utils.to_categorical
    template = '{0}_{1}_ew{2}_f{3}_{4}.csv'
    num_samples = get_num_samples(cv_raw_path, purpose, eeg_source,
                                  eeg_epoch_width_in_s, fold)
    file_ids = [i for i in range(num_samples)]
    if shuffle:
        random.shuffle(file_ids)
    while True:
        eeg_epochs = []
        eeg_labels = []
        for id in file_ids:
            data_filename = (cv_raw_path +
                             template.format(purpose, eeg_source,
                                             str(eeg_epoch_width_in_s),
                                             str(fold),
                                             str(id)))
            with open(data_filename) as datafile:
                datareader = csv.reader(datafile)
                data = next(datareader)
                eeg_labels.append(int(data[-1]))
                data = [float(i) for i in data[0:-1]]
                eeg_epochs.append(data)
            if len(eeg_epochs) == batch_size or id == file_ids[-1]:
                # convert datasets to numpy arrays
                eeg_shape = (len(eeg_epochs), len(eeg_epochs[0]), 1)
                eeg_epochs = np.array(eeg_epochs).reshape(eeg_shape)
                eeg_labels = np.array(eeg_labels, dtype=int)
                eeg_labels = to_cat(eeg_labels, num_classes=num_classes)
                yield (eeg_epochs, eeg_labels)
                eeg_epochs = []
                eeg_labels = []


def read_label_from_file(cv_raw_path, purpose='train', eeg_source='eeg',
                         eeg_epoch_width_in_s=4, fold=0):
    template = '{0}_{1}_ew{2}_f{3}_{4}.csv'
    num_samples = get_num_samples(cv_raw_path, purpose, eeg_source,
                                  eeg_epoch_width_in_s, fold)
    file_ids = [i for i in range(num_samples)]
    eeg_labels = []
    for id in file_ids:
        data_filename = (cv_raw_path +
                         template.format(purpose, eeg_source,
                                         str(eeg_epoch_width_in_s),
                                         str(fold),
                                         str(id)))
        with open(data_filename) as datafile:
            datareader = csv.reader(datafile)
            data = next(datareader)
            eeg_labels.append(int(data[-1]))
    eeg_labels = np.array(eeg_labels, dtype=int)
    return eeg_labels


# parameters to be varied
eeg_epoch_width_in_s = int(sys.argv[2])
eeg_source = sys.argv[1]
num_classes = int(sys.argv[3])
if num_classes == 2:
    target_names = ['Sham', 'TBI']
elif num_classes == 4:
    target_names = ['SW', 'SS', 'TW', 'TS']
elif num_classes == 6:
    target_names = ['SW', 'SN', 'SR', 'TW', 'TN', 'TR']

# set up model and training parameters from file
models_path = 'models/'
config_file = sys.argv[4]
config_params = yaml.load(models_path + config_file)
accuracies = []
reports = []

for fold in range(1):
    # setup the model
    # model is based on Ordonez et al., 2016,
    # http://dx.doi.org/10.3390/s16010115
    filters = [64, 64, 64, 64]
    kernel_size = config_params['kernel_size']
    l2 = tf.keras.regularizers.l2
    reg_rate = 0.01
    kinitializer = 'lecun_uniform'
    num_tsteps = eeg_epoch_width_in_s * 1024 // 4
    lstm_dimensions = [128, 128]
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.BatchNormalization(input_shape=(num_tsteps, 1)))
    model.add(
        tf.keras.layers.Reshape(target_shape=(num_tsteps, 1, 1)))
    for filter in filters:
        model.add(tf.keras.layers.Conv2D(filter, kernel_size=(kernel_size, 1),
                                         padding='same',
                                         kernel_regularizer=l2(reg_rate),
                                         kernel_initializer=kinitializer))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Reshape(target_shape=(num_tsteps,
              filters[-1])))
    for dim in lstm_dimensions:
        model.add(tf.keras.layers.LSTM(units=dim, return_sequences=True,
                                       activation='tanh'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(units=num_classes,
                              kernel_regularizer=l2(reg_rate))
    ))
    model.add(tf.keras.layers.Activation('softmax'))
    model.add(tf.keras.layers.Lambda(lambda x: x[:, -1, :],
                                     output_shape=[num_classes]))

    # define optimizers
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001)

    # compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # set up data path
    cv_raw_path = 'data/cv_raw_{0}c/'.format(str(num_classes))

    # set up training parameters
    batch_size = 1024 * 4 // eeg_epoch_width_in_s
    num_train_samples = get_num_samples(cv_raw_path, 'train', eeg_source,
                                        eeg_epoch_width_in_s, fold)
    num_test_samples = get_num_samples(cv_raw_path, 'test', eeg_source,
                                       eeg_epoch_width_in_s, fold)
    train_steps_per_epoch = int(math.ceil(num_train_samples / batch_size))
    test_steps_per_epoch = int(math.ceil(num_test_samples / batch_size))
    epochs = config_params['epochs']

    # set up tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard()
    tensorboard.log_dir = 'tb_logs/{0}'.format(config_params['config_name'])
    # tensorboard.histogram_freq = epochs / 1
    # tensorboard.write_grads = True
    # tensorboard.batch_size = batch_size
    # tensorboard.update_freq = 'epoch'

    # load previously saved model
    if len(sys.argv) == 6:
        print('loading previous model = ', sys.argv[5])
        model = tf.keras.models.load_model(sys.argv[5])

    # train the model
    train_gen = generate_arrays_from_file(cv_raw_path, 'train', eeg_source,
                                          eeg_epoch_width_in_s, fold,
                                          num_classes, batch_size, True)
    test_gen = generate_arrays_from_file(cv_raw_path, 'test', eeg_source,
                                         eeg_epoch_width_in_s, fold,
                                         num_classes, batch_size, True)
    model.fit_generator(train_gen, train_steps_per_epoch, epochs, verbose=2,
                        callbacks=[tensorboard],
                        validation_data=test_gen,
                        validation_steps=test_steps_per_epoch,
                        max_queue_size=1)

    # save model
    model.save('models/{0}.h5'.format(config_params['config_name']))

    # evaluate accuracy
    test_loss, test_acc = model.evaluate_generator(test_gen,
                                                   test_steps_per_epoch,
                                                   max_queue_size=1)
    accuracies.append(test_acc)
    print('Fold = ' + str(fold) + ' Accuracy = ' + str(test_acc))

    # calculate confusion matrix
    test_gen = generate_arrays_from_file(cv_raw_path, 'test', eeg_source,
                                         eeg_epoch_width_in_s, fold,
                                         num_classes, batch_size, False)
    predict_labels = model.predict_generator(test_gen,
                                             test_steps_per_epoch,
                                             max_queue_size=1)
    predict_labels = predict_labels.argmax(axis=1)
    test_labels = read_label_from_file(cv_raw_path, 'test', eeg_source,
                                       eeg_epoch_width_in_s, fold)

    test_labels = test_labels.argmax(axis=1)
    print(confusion_matrix(test_labels, predict_labels))

    # print report
    report = classification_report(test_labels, predict_labels,
                                   target_names=target_names,
                                   output_dict=True)
    reports.append(report)
    print(classification_report(test_labels, predict_labels,
                                target_names=target_names))

# print out results summary
if len(accuracies) > 1:
    print('Mean  accuracy = ' + str(statistics.mean(accuracies)))
    print('Stdev accuracy = ' + str(statistics.stdev(accuracies)))
    for name in target_names:
        precisions = [reports[i][name]['precision'] for i in range(
            len(reports))]
        print('Mean  prec ' + name + '  = ' + str(statistics.mean(precisions)))
        print('Stdev prec ' + name + '  = ' + str(statistics.stdev(
              precisions)))
    for name in target_names:
        recalls = [reports[i][name]['recall'] for i in range(len(reports))]
        print('Mean  recll ' + name + ' = ' + str(statistics.mean(recalls)))
        print('Stdev recll ' + name + ' = ' + str(statistics.stdev(recalls)))
