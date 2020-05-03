from collections import Counter
import datautil as du
import h5py
import numpy as np
import os
import pandas as pd
import random
from scipy.signal import decimate
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    """
    This class generates the data used to train/test a deep neural network.
    The data division into train and test sets is done such that each class
    and species are represented as uniformly as possible.

    Currently, it only support num_classes = 4.

    Args:
        file_path: location of HDF5 files containing the EEG epochs
        file_template: template of filenamem, e.g., {}_BL5_ew32.h5
        species_set: set of mice species
        purpose: purpose of generator - train or test
        batch_size: batch size
        num_classes: number of species classes
        regenerate: if True, new samples will be regenerated
        shuffle: if True, dataset are shuffled after each epoch
        decimate: decimation factor
        test_percent: percentage of pair samples used for test set
        overlap: if True, use overlapping epochs
    """
    def __init__(self, file_path, file_template, species_set, purpose='train',
                 batch_size=32, num_classes=4, regenerate=False, shuffle=True,
                 decimate=1, test_percent=20, overlap=True):
        self.file_path = file_path
        self.file_template = file_template
        self.species_set = species_set
        assert purpose in ('train', 'test'),\
            'purpose must be either train or test'
        self.purpose = purpose
        self.decimate = decimate
        assert test_percent >= 0 and test_percent <= 100,\
            'test_percent must be between 0 and 100'
        self.test_percent = test_percent
        # check that num_classes is set to 4
        assert num_classes == 4,\
            'Only num_classes = 4 is supported currently'
        self.num_classes = num_classes
        if num_classes == 4:
            self.stages = ['wake', 'sleep']
        elif num_classes == 6:
            self.stages = ['wake', 'nrem', 'rem']
        self.num_samples = self._get_total_num_samples()
        assert batch_size <= self.num_samples,\
            'Batch size must be <= number of (train or test) samples'
        self.batch_size = batch_size

        # read from existing index file for generated samples
        # if regenerate = False; generate new index file if it does not exist
        if overlap:
            self.out_file = file_template[:-3].format('data') +\
                '_{}_{}_{}_{}.h5'.format(self.num_classes, self.batch_size,
                                         self.num_samples, self.test_percent)
        else:
            self.out_file = file_template[:-3].format('data_novl') +\
                '_{}_{}_{}_{}.h5'.format(self.num_classes, self.batch_size,
                                         self.num_samples, self.test_percent)
        if not os.path.exists(self.out_file) or regenerate:
            self._generate_labeled_samples()

        # set the generator to be either train or test data generator
        num_test_samples = int(np.round(self.test_percent *
                                        self.num_samples / 100))
        if self.purpose == 'test':
            self.num_samples = num_test_samples
            self.df = pd.read_hdf(self.out_file, 'data_index/test', mode='r')
        else:
            self.num_samples = num_samples - num_test_samples
            self.df = pd.read_hdf(self.out_file, 'data_index/train', mode='r')

        # shuffle data if shuffle=True
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        # set index range
        # max index is num_samples - 1 for last batch
        min_index = index * self.batch_size
        if (index + 1) * self.batch_size < self.num_samples:
            max_index = (index + 1) * self.batch_size - 1
        else:
            max_index = self.num_samples - 1
        # generate the batch
        data = []
        labels = []
        for pidx in self.indexes[min_index:max_index + 1]:
            df = self.df
            epoch = self.data_from_hdf5(df.at[pidx, 'species'],
                                        df.at[pidx, 'stage'],
                                        df.at[pidx, 'index'])
            if self.decimate > 1:
                epoch = decimate(epoch, self.decimate)
            data.append(epoch)
            labels.append(df.at[pidx, 'label'])
        # convert datasets to numpy arrays
        shape = (len(data), len(data[0]), 1)
        data = np.array(data).reshape(shape)
        labels = np.array(labels, dtype=int)

        return data, labels

    def get_labels(self):
        labels = self.df['label'].tolist()[0:self.num_samples]
        return np.array(labels, dtype=int)

    def get_num_samples(self, species, stage):
        datafile = os.path.join(self.file_path,
                                self.file_template.format(species))
        with h5py.File(datafile, 'r') as datafile:
            num_epoch_samples = datafile['eeg'][stage].shape[0]
        return num_epoch_samples

    def on_epoch_end(self):
        self.indexes = [i for i in range(self.num_samples)]
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_from_hdf5(self, species, stage, idx):
        datafile = os.path.join(self.file_path,
                                self.file_template.format(species))
        data = []
        with h5py.File(datafile, 'r') as sfile:
            data = sfile['eeg'][stage][idx]
        return data

    def _generate_labeled_samples(self):
        if os.path.exists(self.out_file):
            os.remove(self.out_file)
        curr_train_index = 0
        curr_test_index = 0
        store = pd.HDFStore(self.out_file, mode='a', complevel=4,
                            complib='zlib')
        for species in self.species_set:
            for stage in self.stages:
                label = du.get_class_label(self.num_classes, species, stage)
                num_epoch_samples = self.get_num_samples(species, stage)
                indexes = [i for i in range(self.num_epoch_samples)]
                np.random.shuffle(indexes)
                num_test_samples = int(np.round(self.test_percent *
                                                num_epoch_samples / 100))
                num_train_samples = num_epoch_samples - num_test_samples
                df_train_index = list(range(curr_train_index,
                                            curr_train_index +
                                            num_train_samples))
                df_test_index = list(range(curr_test_index,
                                           curr_test_index +
                                           num_test_samples))
                sindex = indexes[:num_train_samples]
                store.append('data_index/train',
                             pd.DataFrame({'species': species,
                                           'stage': stage,
                                           'index': sindex,
                                           'label': label},
                                          index=df_train_index),
                             data_columns=True,
                             min_itemsize={'species': 7,
                                           'stage': 5})
                curr_train_index += num_train_samples
                sindex = indexes[num_train_samples:]
                store.append('data_index/test',
                             pd.DataFrame({'species': species,
                                           'stage': stage,
                                           'index': sindex,
                                           'label': label},
                                          index=df_test_index),
                             data_columns=True,
                             min_itemsize={'species': 7,
                                           'stage': 5})
                curr_test_index += num_test_samples
        store.close()

    def _get_total_num_samples(self):
        num_samples = 0
        for species in self.species_set:
            for stage in self.stages:
                num_samples += self.get_num_samples(species, stage)
        return num_samples
