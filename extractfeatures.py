import networkx as nx
import numpy as np
import pyentrp.entropy as entropy
import pywt
from scipy.stats import kurtosis
from scipy.stats import skew
import sourcedata as sd
import spectralanalysis as sa
import statistics
import timedanalysis as ta
import visibilitygraphs as vg


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
