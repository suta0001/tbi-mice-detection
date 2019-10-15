import csv
import pairdatageneratorrs as dg
from models import get_baseline_convolutional_encoder, build_siamese_net
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import statistics
import sys
import tensorflow as tf
from time import time
import yaml


# parameters to be varied
eeg_epoch_width_in_s = int(sys.argv[2])
eeg_source = sys.argv[1]
num_classes = int(sys.argv[3])
target_names = ['same', 'diff']
decimate_factor = 4

# set up model and training parameters from file
models_path = 'models/'
config_file = '{0}.yaml'.format(sys.argv[4])
with open(models_path + config_file) as cfile:
    config_params = yaml.safe_load(cfile)
accuracies = []
reports = []


def contrastive_loss(y_true, y_pred):
    margin = 1.25
    K = tf.keras.backend
    return K.mean((1 - y_true) * K.square(y_pred) + y_true *
                  K.square(K.maximum(margin - y_pred, 0)))


# load previously saved model if requested
if len(sys.argv) == 6:
    print('loading previous model = ', sys.argv[5])
    model = tf.keras.models.load_model(sys.argv[5])
else:
    # setup the model
    # model is from https://github.com/oscarknagg/voicemap
    filters = config_params['filters']
    embedding_dimension = config_params['embedding_dimension']
    dropout = config_params['dropout']
    num_tsteps = eeg_epoch_width_in_s * 1024 // (4 * decimate_factor)
    with tf.device('/cpu:0'):
        encoder = get_baseline_convolutional_encoder(filters,
                                                     embedding_dimension,
                                                     dropout=dropout)
        model = build_siamese_net(encoder, (num_tsteps, 1),
                                  distance_metric='uni_euc_cont_loss')
        # define optimizers
        optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)

    # compile the model
    pmodel = tf.keras.utils.multi_gpu_model(model, gpus=2)
    # pmodel = model
    pmodel.compile(optimizer=optimizer,
                   loss=contrastive_loss,
                   metrics=['accuracy'])

# set up training parameters
num_samples = config_params['num_samples']
test_percent = config_params['test_percent']
batch_size = 1024 * 8 * decimate_factor // eeg_epoch_width_in_s
epochs = config_params['epochs']

# set up tensorboard
tensorboard = tf.keras.callbacks.TensorBoard()
log_dir = 'tb_logs/{}_{}'.format(config_params['config_name'],
                                 config_params['epochs'])
tensorboard.log_dir = log_dir
# tensorboard.histogram_freq = epochs / 1
# tensorboard.write_grads = True
# tensorboard.batch_size = batch_size
# tensorboard.update_freq = 'epoch'

dataset_folds = [line.rstrip().split(',') for line in open('cv_folds.txt')]
data_path = 'data/epochs_{}c'.format(str(num_classes))
file_template = '{}_BL5_' + 'ew{}.h5'.format(str(eeg_epoch_width_in_s))
for fold in range(1):
    # set up checkpoints
    filepath = 'models/{}_{}_best.h5'.format(config_params['config_name'],
                                             str(fold))
    ckpt_best = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                   monitor='val_acc',
                                                   save_best_only=True,
                                                   mode='max')
    filepath = 'models/{}_{}_'.format(config_params['config_name'],
                                      str(fold))
    filepath += '{epoch:04d}.h5'
    ckpt_reg = tf.keras.callbacks.ModelCheckpoint(filepath, period=5)

    # set callbacks
    callbacks = [ckpt_best, ckpt_reg, tensorboard]

    # set up train and test sets
    sham_set = dataset_folds[fold][0:4] + dataset_folds[fold][7:9]
    tbi_set = dataset_folds[fold][4:7] + dataset_folds[fold][9:11]

    # train the model
    train_gen = dg.PairDataGeneratorRS(data_path, file_template, sham_set,
                                       tbi_set, 'train', batch_size,
                                       num_classes, num_samples,
                                       decimate=decimate_factor,
                                       test_percent=test_percent)
    test_gen = dg.PairDataGeneratorRS(data_path, file_template, sham_set,
                                      tbi_set, 'test', batch_size, num_classes,
                                      num_samples, decimate=decimate_factor,
                                      test_percent=test_percent)
    pmodel.fit_generator(train_gen,
                         epochs=epochs, verbose=1,
                         callbacks=callbacks,
                         validation_data=test_gen,
                         max_queue_size=1)

    # calculate accuracy and confusion matrix
    test_gen = dg.PairDataGeneratorRS(data_path, file_template, sham_set,
                                      tbi_set, 'test', batch_size, num_classes,
                                      num_samples, shuffle=False,
                                      decimate=decimate_factor,
                                      test_percent=test_percent)
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