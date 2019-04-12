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


# parameters to be varied
eeg_epoch_width_in_s = int(sys.argv[1])
eeg_source = 'pp2'
target_names = ['SW', 'SS', 'TW', 'TS']

# read dataset per fold
accuracies = []
reports = []
for fold in range(10):
    cv_path = 'data/cv/'
    template = '{0}_{1}_ew{2}_f{3}_sxx.csv'
    common_labels = [eeg_source, str(eeg_epoch_width_in_s), str(fold)]
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
    scaler = StandardScaler()
    width = (math.floor(math.pow(2, math.log2(eeg_epoch_width_in_s) - 3)) +
             eeg_epoch_width_in_s)
    train_epochs = scaler.fit_transform(train_epochs)
    train_epochs = np.array(train_epochs).reshape((-1, 41, width, 1))
    train_labels = np.array(train_labels, dtype=int)
    test_epochs = scaler.fit_transform(test_epochs)
    test_epochs = np.array(test_epochs).reshape((-1, 41, width, 1))
    test_labels = np.array(test_labels, dtype=int)

    # setup the model
    num_classes = 4
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3),
                               input_shape=train_epochs.shape[1:],
                               padding='same',
                               activation=tf.nn.relu),
        tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                               activation=tf.nn.relu),
        tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])

    # define optimizers
    optimizer = tf.keras.optimizers.SGD(lr=0.01, nesterov='True')

    # compile the model
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # train the model
    batch_size = 32
    epochs = 100
    model.fit(train_epochs, train_labels, batch_size, epochs,
              validation_data=(test_epochs, test_labels), verbose=0)

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
