import csv
import math
import numpy as np
import os
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import sourcedata as sd
import statistics
import sys
import tensorflow as tf
from time import time


def generate_arrays_from_file(eeg_data_file, eeg_labels_file, num_classes,
                              batch_size):
    to_cat = tf.keras.utils.to_categorical
    while True:
        data_file = open(eeg_data_file, mode='r', newline='')
        labels_file = open(eeg_labels_file, mode='r', newline='')
        data_reader = csv.reader(data_file)
        labels_reader = csv.reader(labels_file)
        end_of_file = False
        while not end_of_file:
            eeg_epochs = []
            eeg_labels = []
            while len(eeg_epochs) < batch_size:
                try:
                    data = next(data_reader)
                    label = next(labels_reader)
                    # convert datasets to numpy arrays
                    eeg_epoch = [float(i) for i in data]
                    eeg_label = [float(i) for i in label] 
                    eeg_epochs.append(eeg_epoch)
                    eeg_labels.append(eeg_label)
                except StopIteration:
                    end_of_file = True
                    break
            # convert datasets to numpy arrays
            eeg_shape = (len(eeg_epochs), len(eeg_epochs[0]), 1)
            eeg_epochs = np.array(eeg_epochs).reshape(eeg_shape)
            eeg_labels = np.array(eeg_labels, dtype=int)
            eeg_labels = to_cat(eeg_labels, num_classes=num_classes) 
            yield (eeg_epochs, eeg_labels)
        labels_file.close()
        data_file.close()


def get_num_samples(filename):
    with open(filename, mode='r', newline='') as csvfile:
        for i, line in enumerate(csvfile):
            pass 
    return i + 1 


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

# read dataset per fold
accuracies = []
reports = []
template = '{0}_{1}_ew{2}_f{3}.csv'
common_labels = None

for fold in range(1):
    # set up data and label files
    common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(fold)]
    cv_path = 'data/cv_{0}c/'.format(str(num_classes))
    train_data_file = cv_path + template.format('train_data', *common_labels)
    train_labels_file = cv_path + template.format('train_labels',
                                                  *common_labels)
    test_data_file = cv_path + template.format('test_data', *common_labels)
    test_labels_file = cv_path + template.format('test_labels', *common_labels)

    # get the total number of samples in training and test sets
    num_train_samples = get_num_samples(train_labels_file)
    num_test_samples = get_num_samples(test_labels_file)

    # setup the model
    # model is based on Ordonez et al., 2016,
    # http://dx.doi.org/10.3390/s16010115
    filters = [64, 64, 64, 64]
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
        model.add(tf.keras.layers.Conv2D(filter, kernel_size=(5, 1),
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

    # set up training parameters
    batch_size = 1024
    train_steps_per_epoch = int(math.ceil(num_train_samples / batch_size))
    test_steps_per_epoch = int(math.ceil(num_test_samples / batch_size))
    epochs = 200

    # set up tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard()
    tensorboard.log_dir = 'tb_logs/{0}'.format(time())
    # tensorboard.histogram_freq = epochs / 1
    # tensorboard.write_grads = True
    # tensorboard.batch_size = batch_size
    # tensorboard.update_freq = 'epoch'

    # load previously saved model
    if len(sys.argv) == 5:
        print('loading previous model = ', sys.argv[4])
        model = tf.keras.models.load_model(sys.argv[4])

    # train the model
    train_gen = generate_arrays_from_file(train_data_file, train_labels_file,
                                          num_classes, batch_size)
    test_gen = generate_arrays_from_file(test_data_file, test_labels_file,
                                         num_classes, batch_size)
    model.fit_generator(train_gen, train_steps_per_epoch, epochs, verbose=1,
                        callbacks=[tensorboard],
                        validation_data=test_gen,
                        validation_steps=test_steps_per_epoch)

    # save model
    model.save('models/convlstm_{0}.h5'.format(time()))

    # evaluate accuracy
    test_loss, test_acc = model.evaluate_generator(test_gen,
                                                   test_steps_per_epoch)
    accuracies.append(test_acc)
    print('Fold = ' + str(fold) + ' Accuracy = ' + str(test_acc))

    # calculate confusion matrix
    predict_labels = model.predict_generator(test_gen,
                                             test_steps_per_epoch)
    predict_labels = predict_labels.argmax(axis=1)
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
