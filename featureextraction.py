import datautil as du
from models import get_baseline_convolutional_encoder, build_siamese_net
import networkx as nx
import numpy as np
import pyentrp.entropy as entropy
import pywt
from scipy.signal import decimate
from scipy.stats import kurtosis
from scipy.stats import skew
import spectralanalysis as sa
import statistics
import tensorflow as tf
import timedanalysis as ta
import visibilitygraphs as vg
import yaml


def calc_distance_features(eeg_epochs, source, model_path):
    with open(model_path) as cfile:
        config_params = yaml.safe_load(cfile)
    avg_features = np.array(du.read_data('avg_{}_features.csv'.format(source)))
    siamese_features = generate_embeddings(eeg_epochs, model_path)
    features = []
    for siam_feature in siamese_features:
        feature = []
        for avg_feature in avg_features:
            distance = siam_feature - avg_feature
            feature.append(np.sqrt(np.sum(np.square(distance))))
        features.append(feature)
    return features


def calc_permutation_entropy(eeg_epochs, orders=[3], delays=[1]):
    features = []
    for eeg_epoch in eeg_epochs:
        feature = []
        for order in orders:
            for delay in delays:
                feature.append(entropy.permutation_entropy(eeg_epoch,
                               order=order, delay=delay, normalize=True))
        features.append(feature)
    return features


def create_visibility_graphs(eeg_epoch, vis_graph_type='normal'):
    vis_graph = None

    if vis_graph_type == 'normal':
        vis_graph = vg.visibility_graph(eeg_epoch)
    elif vis_graph_type == 'horizontal':
        vis_graph = vg.horizontal_visibility_graph(eeg_epoch)
    elif vis_graph_type == 'difference':
        normal_vg = vg.visibility_graph(eeg_epoch)
        horizontal_vg = vg.horizontal_visibility_graph(eeg_epoch)
        vis_graph = nx.difference(normal_vg, horizontal_vg)

    return vis_graph


def calc_vg_features(eeg_epochs, max_degree=11):
    features = []
    for eeg_epoch in eeg_epochs:
        feature = []
        nvg = create_visibility_graphs(eeg_epoch, 'normal')
        hvg = create_visibility_graphs(eeg_epoch, 'horizontal')
        dvg = nx.difference(nvg, hvg)
        feature.append(vg.calc_mean_degree(dvg))
        feature.append(vg.calc_mean_degree(hvg))
        feature.extend(vg.calc_degree_probabilities(dvg, max_degree))
        features.append(feature)
    return features


def calc_spectral_features(eeg_epochs, fs=256):
    features = []
    # define frequency bands
    eeg_bands = {
        'delta': (1.0, 3.5),
        'theta': (4.0, 7.5),
        'alpha': (8.0, 12.0),
        'sigma': (13.0, 16.0),
        'beta': (16.5, 25.0),
        'gamma': (30.0, 35.0)
    }
    bands = ['delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma']
    for eeg_epoch in eeg_epochs:
        feature = []
        f, psd = sa.calc_psd(eeg_epoch, fs)
        total_power = sa.calc_bandpower(psd, f, f[0], f[-1])
        power = [sa.calc_bandpower(psd, f, eeg_bands[band][0],
                 eeg_bands[band][1]) for band in bands]
        rel_power = [power[i] / total_power for i in range(len(power))]
        feature.extend(power)
        feature.extend(rel_power)
        feature.append(rel_power[0] / rel_power[1])
        feature.append(rel_power[0] / rel_power[2])
        feature.append(rel_power[0] / rel_power[3])
        feature.append(rel_power[0] / rel_power[4])
        feature.append(rel_power[0] / rel_power[5])
        feature.append(rel_power[1] / rel_power[2])
        feature.append(rel_power[1] / rel_power[3])
        feature.append(rel_power[1] / rel_power[4])
        feature.append(rel_power[1] / rel_power[5])
        feature.append(rel_power[2] / rel_power[3])
        feature.append(rel_power[2] / rel_power[4])
        feature.append(rel_power[2] / rel_power[5])
        feature.append(rel_power[3] / rel_power[4])
        feature.append(rel_power[3] / rel_power[5])
        feature.append(rel_power[4] / rel_power[5])
        feature.append((rel_power[0] + rel_power[1]) /
                       (rel_power[2] + rel_power[3] + rel_power[4] +
                       rel_power[5]))
        feature.append(sa.calc_edge_frequency(psd, f, 0.95))
        feature.append(sa.calc_edge_frequency(psd, f, 0.50))
        feature.append(f[np.argmax(psd)])
        feature.append(sa.calc_spectral_entropy(psd))
        feature.append(statistics.mean(psd))
        feature.append(statistics.variance(psd))
        feature.append(skew(psd))
        feature.append(kurtosis(psd))
        features.append(feature)
    return features


def calc_time_domain_features(eeg_epochs):
    features = []
    for eeg_epoch in eeg_epochs:
        feature = []
        feature.append(statistics.mean(eeg_epoch))
        feature.append(statistics.variance(eeg_epoch))
        feature.append(skew(eeg_epoch))
        feature.append(kurtosis(eeg_epoch))
        feature.append(ta.zero_crossing_rate(eeg_epoch))
        feature.extend(ta.calc_hjorth_params(eeg_epoch))
        feature.append(np.percentile(eeg_epoch, 75))
        features.append(feature)
    return features


def calc_wavelet_pe_features(eeg_epochs):
    features = []
    for eeg_epoch in eeg_epochs:
        feature = []
        # extract wavelet components
        coeffs = pywt.wavedec(eeg_epoch, 'db4', level=6)
        wcs = []
        n = len(eeg_epoch)
        wcs.append(pywt.upcoeff('a', coeffs[0], 'db4', level=6, take=n))
        for i in range(1, 7):
            wcs.append(pywt.upcoef('d', coeffs[i], 'db4', level=6 - i, take=n))
        # calculate permutation entropy for each wavelet component
        for wc in wcs:
            for order in [3, 5, 7]:
                for delay in [1, 5, 10]:
                    feature.append(entropy.permutation_entropy(wc,
                                   order=order, delay=delay, normalize=True))
        features.append(feature)
    return features


def generate_embeddings(eeg_epochs, model_path):
    # due to tensorflow/keras issue, we cannot load model directly from file
    # so, we are forced to hardcode the model
    with open(model_path) as cfile:
        config_params = yaml.safe_load(cfile)
    filters = config_params['filters']
    embedding_dimension = config_params['embedding_dimension']
    dropout = config_params['dropout']
    num_tsteps = len(eeg_epochs[0])
    num_samples = len(eeg_epochs)
    with tf.device('/cpu:0'):
        model = get_baseline_convolutional_encoder(filters,
                                                   embedding_dimension,
                                                   dropout=dropout,
                                                   input_shape=(num_tsteps, 1))
        model = build_siamese_net(model, (num_tsteps, 1),
                                  distance_metric='uni_euc_cont_loss')
        net_model = 'models/{}_{}c_ew{}_{}_0_best.h5'
        net_model = net_model.format(config_params['config_name'],
                                     config_params['num_classes'],
                                     config_params['epoch_width'],
                                     config_params['epochs'])
        model.load_weights(net_model)
        model = model.layers[2]
    shape = (num_samples, num_tsteps, 1)
    features = model.predict(np.array(eeg_epochs).reshape(shape))
    return features


def process(eeg_epochs, method):
    if method == 'pe':
        op = calc_permutation_entropy
        kwargs = {'orders': [3, 5, 7], 'delays': [1, 5, 10]}
    elif method == 'vg':
        op = calc_vg_features
        kwargs = {'max_degree': 11}
    elif method == 'spectral':
        op = calc_spectral_features
        kwargs = {'fs': 256}
    elif method == 'timed':
        op = calc_time_domain_features
        kwargs = {}
    elif method == 'wpe':
        op = calc_wavelet_pe_features
        kwargs = {}
    elif method == 'siamese':
        op = generate_embeddings
        kwargs = {'model_path': 'models/basesiam_4c_ew32_1000.yaml'}
    elif method == 'siamesers':
        op = generate_embeddings
        kwargs = {'model_path': 'models/basesiamrs_4c_ew32_1000.yaml'}
    elif method == 'siamdist':
        op = calc_distance_features
        kwargs = {'source': 'siamese',
                  'model_path': 'models/basesiam_4c_ew32_1000.yaml'}
    elif method == 'siamrsdist':
        op = calc_distance_features
        kwargs = {'source': 'siamesers',
                  'model_path': 'models/siamrs1_4c_ew32_50.yaml'}
    return op(eeg_epochs, **kwargs)
