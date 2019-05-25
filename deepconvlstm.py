import csv
import datagenerator as dg
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import sourcedata as sd
import statistics
import sys
import tensorflow as tf
from time import time
import yaml


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
config_file = '{0}.yaml'.format(sys.argv[4])
with open(models_path + config_file) as cfile:
    config_params = yaml.safe_load(cfile)
accuracies = []
reports = []

# load previously saved model if requested
if len(sys.argv) == 6:
    print('loading previous model = ', sys.argv[5])
    model = tf.keras.models.load_model(sys.argv[5])
else:
    # setup the model
    # model is based on Ordonez et al., 2016,
    # http://dx.doi.org/10.3390/s16010115
    # code is based on mcfly
    # https://mcfly.readthedocs.io/en/latest/installation.html
    filters = config_params['filters']
    kernel_size = config_params['kernel_size']
    l2 = tf.keras.regularizers.l2
    reg_rate = 0.01
    kinitializer = 'lecun_uniform'
    num_tsteps = eeg_epoch_width_in_s * 1024 // 4
    lstm_dimensions = [128, 128]
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.BatchNormalization(input_shape=(num_tsteps, 1)))
        model.add(
            tf.keras.layers.Reshape(target_shape=(num_tsteps, 1, 1)))
        for filter in filters:
            model.add(tf.keras.layers.Conv2D(filter,
                                             kernel_size=(kernel_size, 1),
                                             padding='same',
                                             kernel_regularizer=l2(reg_rate),
                                             kernel_initializer=kinitializer))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Reshape(target_shape=(num_tsteps,
                  filters[-1])))
        for dim in lstm_dimensions:
            model.add(tf.keras.layers.LSTM(units=dim,
                                           return_sequences=True,
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
batch_size = 1024 * 4 // eeg_epoch_width_in_s
epochs = config_params['epochs']

# set up tensorboard
tensorboard = tf.keras.callbacks.TensorBoard()
log_dir = 'tb_logs/{0}_{1}'.format(config_params['config_name'],
                                   config_params['epochs'])
tensorboard.log_dir = log_dir
# tensorboard.histogram_freq = epochs / 1
# tensorboard.write_grads = True
# tensorboard.batch_size = batch_size
# tensorboard.update_freq = 'epoch'

for fold in range(1):
    # set up checkpoints
    filepath = 'models/{0}_{1}_best.h5'.format(config_params['config_name'],
                                               str(fold))
    ckpt_best = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                   monitor='val_acc',
                                                   save_best_only=True,
                                                   mode='max')
    filepath = 'models/{0}_{1}_'.format(config_params['config_name'],
                                        str(fold))
    filepath += '{epoch:04d}.h5'
    ckpt_reg = tf.keras.callbacks.ModelCheckpoint(filepath, period=25)

    # set callbacks
    callbacks = [ckpt_best, ckpt_reg, tensorboard]

    # set up data path
    cv_raw_path = 'data/cv_raw_{0}c/'.format(str(num_classes))

    # train the model
    common_template = '{0}_{1}_ew{2}_f{3}_'
    train_template = common_template.format('train', eeg_source,
                                            str(eeg_epoch_width_in_s),
                                            str(fold))
    train_template += '{0}.csv'
    test_template = common_template.format('test', eeg_source,
                                           str(eeg_epoch_width_in_s),
                                           str(fold))
    test_template += '{0}.csv'
    train_gen = dg.DataGenerator(cv_raw_path, train_template, batch_size,
                                 num_classes, True)
    test_gen = dg.DataGenerator(cv_raw_path, test_template, batch_size,
                                num_classes, True)
    model.fit_generator(train_gen,
                        epochs=epochs, verbose=1,
                        callbacks=callbacks,
                        validation_data=test_gen,
                        max_queue_size=1)

    # calculate accuracy and confusion matrix
    test_gen = dg.DataGenerator(cv_raw_path, test_template, batch_size,
                                num_classes, False)
    predict_labels = model.predict_generator(test_gen,
                                             max_queue_size=1)
    predict_labels = predict_labels.argmax(axis=1)
    test_labels = test_gen.read_label_from_file()
    test_acc = accuracy_score(test_labels, predict_labels)
    accuracies.append(test_acc)
    print('Fold = ' + str(fold) + ' Accuracy = ' +
          '{:.3f}'.format(test_acc))
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
    print('Mean  accuracy = ' + '{:.3f}'.format(statistics.mean(accuracies)))
    print('Stdev accuracy = ' + '{:.3f}'.format(statistics.stdev(accuracies)))
    for name in target_names:
        precisions = [reports[i][name]['precision'] for i in range(
            len(reports))]
        print('Mean  prec ' + name + '  = ' +
              '{:.3f}'.format(statistics.mean(precisions)))
        print('Stdev prec ' + name + '  = ' +
              '{:.3f}'.format(statistics.stdev(precisions)))
    for name in target_names:
        recalls = [reports[i][name]['recall'] for i in range(len(reports))]
        print('Mean  recll ' + name + ' = ' +
              '{:.3f}'.format(statistics.mean(recalls)))
        print('Stdev recll ' + name + ' = ' +
              '{:.3f}'.format(statistics.stdev(recalls)))
