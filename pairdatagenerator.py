from collections import Counter
import h5py
from itertools import combinations, permutations, product
import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf


class PairDataGenerator(tf.keras.utils.Sequence):
    """
    This class generates the pair data used to train/test a Siamese network.

    The pair data is restricted to come from different mice species to improve
    generality of the embedding learnt by the Siamese network.
    Pair from same classes is labeled with 0.
    Pair from different classes is labeled with 1.
    The set of sham/TBI mice must contain at least 2 unique species.

    Currently, it only support num_classes = 4.  Number of samples generated
    may be more than num_samples to ensure pair samples are distributed
    evenly among available pairs.

    Args:
        file_path: location of HDF5 files containing the EEG epochs
        file_template: template of filenamem, e.g., {}_BL5_ew32.h5
        sham_set: set of mice species with sham injury
        tbi_set: set of mice species with TBI
        batch_size: training batch size
        num_classes: number of species classes
        num_samples: number of pair samples to generate
        regenerate: if True, new samples will be regenerated
        shuffle: if True, dataset are shuffled after each epoch
    """
    def __init__(self, file_path, file_template, sham_set, tbi_set,
                 batch_size=32, num_classes=4, num_samples=1024,
                 regenerate=False, shuffle=True):
        self.file_path = file_path
        self.file_template = file_template
        assert batch_size <= num_samples,\
            'Batch size must be <= number of samples'
        self.batch_size = batch_size
        self.num_samples = num_samples
        # check that num_classes is set to 4
        assert num_classes == 4,\
            'Only num_classes = 4 is supported currently'
        self.num_classes = num_classes
        if num_classes == 4:
            self.stages = ['wake', 'sleep']
            self.num_class_combs = 10
            self.num_same_pairs = 4
            self.num_diff_pairs = 6
        elif num_classes == 6:
            self.stages = ['wake', 'nrem', 'rem']
            self.num_class_combs = 27
            self.num_same_pairs = 6
            self.num_diff_pairs = 21

        # check that sham_set and tbi_set contain at least 2 different species
        sham_set = list(Counter(sham_set))
        tbi_set = list(Counter(tbi_set))
        assert len(sham_set) > 1,\
            "Sham set must contain at least 2 unique species"
        assert len(tbi_set) > 1,\
            "TBI set must contain at least 2 unique species"
        self.sham_set = sham_set
        self.tbi_set = tbi_set

        # read from existing index file for generated samples
        # if regenerate = False; generate new index file if it does not exist
        self.out_file = file_template[:-3].format('pairdata') +\
            '_{}_{}_{}.h5'.format(num_classes, batch_size, num_samples)
        if not os.path.exists(self.out_file) or regenerate:
            self._generate_labeled_pairs()
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
        data0 = []
        data1 = []
        labels = []
        store = pd.HDFStore(self.out_file, mode='r')
        for index in self.indexes[min_index:max_index + 1]:
            df = store.select('pair_index', start=index, stop=index + 1)
            epoch0, epoch1 = self.pair_data_from_hdf5(df.at[index, 'species0'],
                                                      df.at[index, 'species1'],
                                                      df.at[index, 'stage0'],
                                                      df.at[index, 'stage1'],
                                                      df.at[index, 'index0'],
                                                      df.at[index, 'index1'])
            data0.append(epoch0)
            data1.append(epoch1)
            labels.append(df.at[index, 'label'])
        store.close()
        # convert datasets to numpy arrays
        shape = (len(data0), len(data0[0]), 1)
        data0 = np.array(data0).reshape(shape)
        data1 = np.array(data1).reshape(shape)
        labels = np.array(labels, dtype=int)

        return [data0, data1], labels

    def get_labels(self):
        labels = []
        with pd.HDFStore(self.out_file, mode='r') as store:
            df = store.select('pair_index', columns=['label'])
            labels = df['label'].tolist()
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

    def pair_data_from_hdf5(self, species0, species1, stage0, stage1, idx0,
                            idx1):
        file0 = os.path.join(self.file_path,
                             self.file_template.format(species0))
        file1 = os.path.join(self.file_path,
                             self.file_template.format(species1))
        data0 = []
        data1 = []
        with h5py.File(file0, 'r') as file0, h5py.File(file1, 'r') as file1:
            data0 = file0['eeg'][stage0][idx0]
            data1 = file1['eeg'][stage1][idx1]
        return data0, data1

    def _generate_labeled_pairs(self):
        if os.path.exists(self.out_file):
            os.remove(self.out_file)
        curr_index = 0
        store = pd.HDFStore(self.out_file, mode='a', complevel=4,
                            complib='zlib')
        for type in ['Sham', 'TBI', 'Both']:
            if type == 'Both':
                species_combs = list(product(self.sham_set, self.tbi_set))
                # div_factor is set to 1 since each pair of stages account
                # for one stage combination
                div_factor = 1
            else:
                if type == 'Sham':
                    species_set = self.sham_set
                elif type == 'TBI':
                    species_set = self.tbi_set
                species_combs = list(combinations(species_set, 2))
                # div_factor is set to 2 for 2 pairs of different stages
                # account for same stage combination in one set case
                div_factor = 2
            stage_perms = list(product(self.stages, self.stages))
            for species0, species1 in species_combs:
                for stage0, stage1 in stage_perms:
                    num_epoch_samples0 = self.get_num_samples(species0, stage0)
                    num_epoch_samples1 = self.get_num_samples(species1, stage1)
                    if type == 'Both' or stage0 != stage1:
                        num_pair_samples = int(np.ceil(self.num_samples / 2 /
                                               self.num_diff_pairs /
                                               div_factor /
                                               len(species_combs)))
                        label = 1
                    else:
                        num_pair_samples = int(np.ceil(self.num_samples / 2 /
                                               self.num_same_pairs /
                                               len(species_combs)))
                        label = 0
                    temp_count = int(np.ceil(np.sqrt(num_pair_samples)))
                    index0 = random.sample(list(range(num_epoch_samples0)),
                                           temp_count)
                    index1 = random.sample(list(range(num_epoch_samples1)),
                                           int(np.ceil(num_pair_samples /
                                               temp_count)))
                    index_pair = random.sample(list(product(index0, index1)),
                                               num_pair_samples)
                    index_pair = [list(t) for t in zip(*index_pair)]
                    df_index = list(range(curr_index,
                                    curr_index + num_pair_samples))
                    store.append('pair_index',
                                 pd.DataFrame({'species0': species0,
                                               'species1': species1,
                                               'stage0': stage0,
                                               'stage1': stage1,
                                               'index0': index_pair[0],
                                               'index1': index_pair[1],
                                               'label': label},
                                              index=df_index),
                                 data_columns=True,
                                 min_itemsize={'species0': 7,
                                               'species1': 7,
                                               'stage0': 5,
                                               'stage1': 5})
                    curr_index += num_pair_samples
        store.close()
