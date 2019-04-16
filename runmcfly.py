from keras.models import load_model
from keras.utils import to_categorical
from mcfly import modelgen, find_architecture, storage
import numpy as np
import os
import pandas as pd
import sourcedata as sd
import sys


# parameters to be varied
eeg_epoch_width_in_s = int(sys.argv[1])
target_names = ['SW', 'SS', 'TW', 'TS']

# load data
epochs_path = 'data/epochs/'
epoch_template = '{0}_ew{1}.csv'
epoch_common_labels = [str(eeg_epoch_width_in_s)]
dataset_train = ['Sham104', 'TBI103', 'Sham105', 'TBI104', 'Sham107']
dataset_val = ['Sham108', 'TBI106']
dataset_test = ['Sham 102', 'TBI101', 'Sham103', 'TBI102']

train_epochs = []
train_labels = []
for species in dataset_train:
    eeg_epochs = []
    stage_epochs = []
    input_filename = epoch_template.format(species + '_BL5_eeg',
                                           *epoch_common_labels)
    sd.read_data(epochs_path + input_filename, eeg_epochs)
    input_filename = epoch_template.format(species + '_BL5_labels',
                                           *epoch_common_labels)
    sd.read_data(epochs_path + input_filename, stage_epochs)
    train_epochs.extend(eeg_epochs)
    train_labels.extend(stage_epochs)

val_epochs = []
val_labels = []
for species in dataset_val:
    eeg_epochs = []
    stage_epochs = []
    input_filename = epoch_template.format(species + '_BL5_eeg',
                                           *epoch_common_labels)
    sd.read_data(epochs_path + input_filename, eeg_epochs)
    input_filename = epoch_template.format(species + '_BL5_labels',
                                           *epoch_common_labels)
    sd.read_data(epochs_path + input_filename, stage_epochs)
    val_epochs.extend(eeg_epochs)
    val_labels.extend(stage_epochs)

test_epochs = []
test_labels = []
for species in dataset_val:
    eeg_epochs = []
    stage_epochs = []
    input_filename = epoch_template.format(species + '_BL5_eeg',
                                           *epoch_common_labels)
    sd.read_data(epochs_path + input_filename, eeg_epochs)
    input_filename = epoch_template.format(species + '_BL5_labels',
                                           *epoch_common_labels)
    sd.read_data(epochs_path + input_filename, stage_epochs)
    test_epochs.extend(eeg_epochs)
    test_labels.extend(stage_epochs)

# convert data to numpy arrays
num_classes = 4
train_shape = (len(train_epochs), len(train_epochs[0]), 1)
train_epochs = np.array(train_epochs).reshape(train_shape)
train_labels = to_categorical(train_labels, num_classes=num_classes, dtype=int)
val_shape = (len(val_epochs), len(val_epochs[0]), 1)
val_epochs = np.array(val_epochs).reshape(val_shape)
val_labels = to_categorical(val_labels, num_classes=num_classes, dtype=int)
test_shape = (len(test_epochs), len(test_epochs[0]), 1)
test_epochs = np.array(test_epochs).reshape(test_shape)
test_labels = to_categorical(test_labels, num_classes=num_classes, dtype=int)

# run mcfly magic
opath = 'data/mcfly/'
mpath = 'data/mcfly/models/'
outfile = opath + 'models.json'
if not os.path.exists(opath):
    os.makedirs(opath)
if not os.path.exists(mpath):
    os.makedirs(mpath)
subset_size = 21598 * 2
find_arch = find_architecture.find_best_architecture
model, params, mtype, acc = find_arch(train_epochs, train_labels,
                                      val_epochs, val_labels,
                                      verbose=True,
                                      number_of_models=8,
                                      nr_epochs=5,
                                      subset_size=subset_size,
                                      outputpath=outfile,
                                      model_path=mpath)
print('Best model = ' + mtype)
print(params)
print('kNN acc = ' + acc)

# train best model on full dataset
nr_epochs = 50
history = model.fit(train_epochs, train_labels, epochs=nr_epochs,
                    validation_data=(val_epochs, val_labels))

# save models
modelname = 'bestmodel.h5'
model_path = mpath + modelname
model.save(model_path)

# evaluate accuracy
test_loss, test_acc = model.evaluate(test_epochs, test_labels)
print('Accuracy = ' + str(test_acc))

# calculate confusion matrix
predict_labels = model.predict(test_epochs)
predict_labels = predict_labels.argmax(axis=1)
print(confusion_matrix(test_labels, predict_labels))

# print report
print(classification_report(test_labels, predict_labels,
                            target_names=target_names))
