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

for fold in range(2):
    common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(fold)]
    cv_path = 'data/cv_{0}c/'.format(str(num_classes))
    train_epochs = []
    train_labels = []
    test_epochs = []
    test_labels = []
    input_filename = template.format('train_data', *common_labels)
    sd.read_data(cv_path + input_filename, train_epochs)
    input_filename = template.format('train_labels', *common_labels)
    sd.read_data(cv_path + input_filename, train_labels)
    input_filename = template.format('test_data', *common_labels)
    sd.read_data(cv_path + input_filename, test_epochs)
    input_filename = template.format('test_labels', *common_labels)
    sd.read_data(cv_path + input_filename, test_labels)

    # convert datasets to numpy arrays
    to_cat = tf.keras.utils.to_categorical
    train_shape = (len(train_epochs), len(train_epochs[0]), 1)
    train_epochs = np.array(train_epochs).reshape(train_shape)
    train_labels = np.array(train_labels, dtype=int)
    train_labels = to_cat(train_labels, num_classes=num_classes)
    test_shape = (len(test_epochs), len(test_epochs[0]), 1)
    test_epochs = np.array(test_epochs).reshape(test_shape)
    test_labels = np.array(test_labels, dtype=int)
    test_labels = to_cat(test_labels, num_classes=num_classes)

    # setup the model
    # model is based on Ordonez et al., 2016,
    # http://dx.doi.org/10.3390/s16010115
    filters = [4, 4, 4]
    l2 = tf.keras.regularizers.l2
    reg_rate = 0.01
    kinitializer = 'lecun_uniform'
    num_tsteps = train_epochs.shape[1]
    lstm_dimensions = [4, 4, 4]
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.BatchNormalization(input_shape=(num_tsteps, 1)))
    model.add(
        tf.keras.layers.Reshape(target_shape=(num_tsteps, 1, 1)))
    for filter in filters:
        model.add(tf.keras.layers.Conv2D(filter, kernel_size=(3, 1),
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
    optimizer = tf.keras.optimizers.Adam(lr=0.0001)

    # compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # set up training parameters
    batch_size = 32
    epochs = 100

    # set up tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard()
    tensorboard.log_dir = 'tb_logs/{}'.format(time())
    tensorboard.histogram_freq = epochs / 10
    tensorboard.write_grads = True
    tensorboard.batch_size = batch_size
    tensorboard.update_freq = 'epoch'

    # train the model
    model.fit(train_epochs, train_labels, batch_size, epochs,
              validation_data=(test_epochs, test_labels), verbose=0,
              callbacks=[tensorboard])

    # save model
    model.save('convlstm_{0}.h5'.format(time()))

    # evaluate accuracy
    test_loss, test_acc = model.evaluate(test_epochs, test_labels)
    accuracies.append(test_acc)
    print('Fold = ' + str(fold) + ' Accuracy = ' + str(test_acc))

    # calculate confusion matrix
    predict_labels = model.predict(test_epochs)
    predict_labels = predict_labels.argmax(axis=1)
    print(confusion_matrix(test_labels, predict_labels))

    # print report
    report = classification_report(test_labels, predict_labels,
                                   target_names=target_names,
                                   output_dict=True)
    reports.append(report)
    print(classification_report(test_labels, predict_labels,
                                target_names=target_names))

# print out results summary
print('Mean  accuracy = ' + str(statistics.mean(accuracies)))
print('Stdev accuracy = ' + str(statistics.stdev(accuracies)))
for name in target_names:
    precisions = [reports[i][name]['precision'] for i in range(len(reports))]
    print('Mean  prec ' + name + '  = ' + str(statistics.mean(precisions)))
    print('Stdev prec ' + name + '  = ' + str(statistics.stdev(precisions)))
for name in target_names:
    recalls = [reports[i][name]['recall'] for i in range(len(reports))]
    print('Mean  recll ' + name + ' = ' + str(statistics.mean(recalls)))
    print('Stdev recll ' + name + ' = ' + str(statistics.stdev(recalls)))
