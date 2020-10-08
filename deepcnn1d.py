import csv
import datagenerator as dg
from models import get_baseline_convolutional_encoder
from models import get_cnn1d_with_attention
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import statistics
import sys
import tensorflow as tf
import yaml


"""Train Deep CNN (Species Aware version)"""
# set up model and training parameters from file
models_path = 'models/'
config_file = '{0}.yaml'.format(sys.argv[1])
with open(models_path + config_file) as cfile:
    config_params = yaml.safe_load(cfile)
eeg_source = 'eeg'  # use raw EEG epochs
eeg_epoch_width_in_s = config_params['epoch_width']
num_classes = config_params['num_classes']
target_names = ['SW', 'SS', 'TW', 'TS']
decimate_factor = config_params['decimate']
accuracies = []
reports = []

# setup the model
filters = config_params['filters']
embedding_dimension = config_params['embedding_dimension']
dropout = config_params['dropout']
num_tsteps = eeg_epoch_width_in_s * 1024 // (4 * decimate_factor)

# set up training parameters
batch_size = 1024 * 4 * decimate_factor // eeg_epoch_width_in_s
epochs = config_params['epochs']

# set up data source
dataset_folds = [line.rstrip().split(',') for line in
                 open('cv_folds3.txt')]
if config_params['overlap']:
    data_path = 'data/epochs_{}c'.format(str(num_classes))
else:
    data_path = 'data/epochs_novl_{}c'.format(str(num_classes))
file_template = '{}_BL5_' + 'ew{}.h5'.format(str(eeg_epoch_width_in_s))

for fold in range(len(dataset_folds)):
    with tf.device('/cpu:0'):
        # define model
        # 2 models are supported: base Deep CNN and Deep CNN with Attention
        if config_params['model'] == 'basecnn':
            model = get_baseline_convolutional_encoder(filters,
                                                       embedding_dimension,
                                                       (num_tsteps, 1),
                                                       dropout)
            model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        elif config_params['model'] == 'attcnn':
            kernel_width = int(config_params['kernel_width'] * 256 //
                               decimate_factor)
            model = get_cnn1d_with_attention(num_classes, filters,
                                             embedding_dimension, (num_tsteps, 1),
                                             dropout, kernel_width)
        # define optimizers
        optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        # load previously saved model if requested
        if len(sys.argv) == 3:
            print('loading previous model = ', sys.argv[2])
            model.load_weights(sys.argv[2])

    # compile the model
    # uncomment line below if using multiple GPUs for training
    # pmodel = tf.keras.utils.multi_gpu_model(model, gpus=2)
    # comment line below if using multiple GPUs for training
    pmodel = model
    pmodel.compile(optimizer=optimizer,
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    # set up tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard()
    log_dir = 'tb_logs/{}_{}c_ew{}_{}_{}'.format(config_params['config_name'],
                                                 config_params['num_classes'],
                                                 config_params['epoch_width'],
                                                 config_params['epochs'],
                                                 str(fold))
    tensorboard.log_dir = log_dir
    # tensorboard.histogram_freq = epochs / 1
    # tensorboard.write_grads = True
    # tensorboard.batch_size = batch_size
    # tensorboard.update_freq = 'epoch'

    # set up checkpoint for best model
    filepath = 'models/{}_{}c_ew{}_{}_{}_best.h5'.format(
        config_params['config_name'],
        config_params['num_classes'],
        config_params['epoch_width'],
        config_params['epochs'],
        str(fold))
    ckpt_best = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                   monitor='val_loss',
                                                   save_best_only=True,
                                                   mode='min')
    # set up regular checkpoint for training
    filepath = 'models/{}_{}c_ew{}_{}_{}'.format(
        config_params['config_name'],
        config_params['num_classes'],
        config_params['epoch_width'],
        config_params['epochs'],
        str(fold))
    filepath += '{epoch:04d}.h5'
    ckpt_reg = tf.keras.callbacks.ModelCheckpoint(filepath, period=5)

    # set callbacks
    callbacks = [ckpt_best, ckpt_reg, tensorboard]

    # set up train and test sets
    train_set = dataset_folds[fold][0:9]
    test_set = dataset_folds[fold][9:]

    # train the model
    train_gen = dg.DataGenerator(data_path, file_template, train_set,
                                 'train', batch_size, num_classes,
                                 decimate=decimate_factor,
                                 test_percent=0,
                                 val_percent=12.5,
                                 overlap=config_params['overlap'],
                                 num_samples=config_params['num_samples'])
    val_gen = dg.DataGenerator(data_path, file_template, train_set,
                               'validation', batch_size, num_classes,
                               decimate=decimate_factor,
                               test_percent=0,
                               val_percent=12.5,
                               overlap=config_params['overlap'],
                               num_samples=config_params['num_samples'])
    pmodel.fit_generator(train_gen,
                         epochs=epochs, verbose=1,
                         callbacks=callbacks,
                         validation_data=val_gen,
                         max_queue_size=1)

    # calculate accuracy and confusion matrix
    test_gen = dg.DataGenerator(data_path, file_template, test_set,
                                'test', batch_size, num_classes,
                                shuffle=False,
                                decimate=decimate_factor,
                                test_percent=99.9,
                                val_percent=0,
                                overlap=config_params['overlap'],
                                num_samples=config_params['num_samples'])
    filepath = 'models/{}_{}c_ew{}_{}_{}_best.h5'.format(
        config_params['config_name'],
        config_params['num_classes'],
        config_params['epoch_width'],
        config_params['epochs'],
        str(fold))
    pmodel = tf.keras.models.load_model(filepath)
    predict_labels = pmodel.predict_generator(test_gen,
                                              max_queue_size=1)
    predict_labels = predict_labels.argmax(axis=1)
    test_labels = test_gen.get_labels()
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
