import csv
import numpy as np
import os
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_path, file_template, batch_size=32, num_classes=4,
                 shuffle=True):
        self.batch_size = batch_size
        self.file_path = file_path
        self.file_template = file_template
        self.num_classes = num_classes
        self.num_samples = self._get_num_samples()
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        # set index range
        # max index is num_samples - 1 for last batch
        min_index = index * self.batch_size
        if (index + 1) * self.batch_size < self.num_samples - 1:
            max_index = (index + 1) * self.batch_size
        else:
            max_index = self.num_samples - 1
        # generate the batch
        data = []
        labels = []
        for id in file_ids[min_index:max_index + 1]:
            data_filename = (self.file_path +
                             self.file_template.format(str(id)))
            with open(data_filename) as datafile:
                datareader = csv.reader(datafile)
                read_data = next(datareader)
                labels.append(int(read_data[-1]))
                read_data = [float(i) for i in read_data[0:-1]]
                data.append(read_data)
        # convert datasets to numpy arrays
        shape = (len(data), len(data[0]), 1)
        data = np.array(data).reshape(shape)
        labels = np.array(labels, dtype=int)
        labels = tf.keras.utils.to_categorical(labels,
                                               num_classes=self.num_classes)
        return data, labels

    def read_label_from_file(self):
        labels = []
        for id in file_ids:
            data_filename = (self.file_path +
                             self.file_template.format(str(id)))
            with open(data_filename) as datafile:
                datareader = csv.reader(datafile)
                data = next(datareader)
                labels.append(int(data[-1]))
        return np.array(labels, dtype=int)

    def on_epoch_end(self):
        self.file_ids = [i for i in range(self.num_samples)]
        if self.shuffle:
            np.random.shuffle(self.file_ids)

    def _get_num_samples(self):
        files = [file for file in os.listdir(self.file_path) if
                 self.file_template[-7:-1] in file]
        return len(files)
