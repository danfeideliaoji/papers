#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris, load_wine, fetch_california_housing  # load_boston,

import os
import pandas as pd

import wget
import numpy as np

# 'boston',
DATASETS = ['iris', 'wine', 'california', 'parkinsons', \
            'climate_model_crashes', 'concrete_compression', \
            'yacht_hydrodynamics', 'airfoil_self_noise', \
            'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', \
            'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', \
            'blood_transfusion', 'breast_cancer_diagnostic', \
            'connectionist_bench_vowel', 'concrete_slump', \
            'wine_quality_red', 'wine_quality_white']


def uci_dataset_loader(dataset):
    """
    Data loading utility for a subset of UCI ML repository datasets. Assumes
    datasets are located in './datasets'. If the called for dataset is not in
    this folder, it is downloaded from the UCI ML repo.

    Parameters
    ----------

    dataset : str
        Name of the dataset to retrieve.
        Valid values: see DATASETS.

    Returns
    ------
    X : ndarray
        Data values (predictive values only).
    """
    assert dataset in DATASETS, f"Dataset not supported: {dataset}"
    X2 = None
    if dataset in DATASETS:
        if dataset == 'iris':
            X, Y = load_iris()['data'], load_iris()['target']
        elif dataset == 'wine':
            X, Y = load_wine()['data'], load_wine()['target']
        elif dataset == 'boston':
            X, Y = load_boston()['data'], load_boston()['target']
        elif dataset == 'california':
            X, Y = fetch_california_housing()['data'], fetch_california_housing()['target']
        elif dataset == 'parkinsons':
            X, Y = fetch_parkinsons()['data'], fetch_parkinsons()['target']
        elif dataset == 'climate_model_crashes':
            X, Y = fetch_climate_model_crashes()['data'], fetch_climate_model_crashes()['target']
        elif dataset == 'concrete_compression':
            X, Y = fetch_concrete_compression()['data'], fetch_concrete_compression()['target']
        elif dataset == 'yacht_hydrodynamics':
            X, Y = fetch_yacht_hydrodynamics()['data'], fetch_yacht_hydrodynamics()['target']
        elif dataset == 'airfoil_self_noise':
            X, Y = fetch_airfoil_self_noise()['data'], fetch_airfoil_self_noise()['target']
        elif dataset == 'connectionist_bench_sonar':
            X, Y = fetch_connectionist_bench_sonar()['data'], fetch_connectionist_bench_sonar()['target']
        elif dataset == 'ionosphere':
            X, Y = fetch_ionosphere()['data'], fetch_ionosphere()['target']
        elif dataset == 'qsar_biodegradation':
            X, Y = fetch_qsar_biodegradation()['data'], fetch_qsar_biodegradation()['target']
        elif dataset == 'seeds':
            X, Y = fetch_seeds()['data'], fetch_seeds()['target']
        elif dataset == 'glass':
            X, Y = fetch_glass()['data'], fetch_glass()['target']
        elif dataset == 'ecoli':
            X, Y = fetch_ecoli()['data'], fetch_ecoli()['target']
        elif dataset == 'yeast':
            X, Y = fetch_yeast()['data'], fetch_yeast()['target']
        elif dataset == 'libras':
            X, Y = fetch_libras()['data'], fetch_libras()['target']
        elif dataset == 'planning_relax':
            X, Y = fetch_planning_relax()['data'], fetch_planning_relax()['target']
        elif dataset == 'blood_transfusion':
            X, Y = fetch_blood_transfusion()['data'], fetch_blood_transfusion()['target']
        elif dataset == 'breast_cancer_diagnostic':
            X, Y = fetch_breast_cancer_diagnostic()['data'], fetch_breast_cancer_diagnostic()['target']
        elif dataset == 'connectionist_bench_vowel':
            X, Y = fetch_connectionist_bench_vowel()['data'], fetch_connectionist_bench_vowel()['target']
        elif dataset == 'concrete_slump':
            X, Y = fetch_concrete_slump()['data'], fetch_concrete_slump()['target']
        elif dataset == 'wine_quality_red':
            X, Y = fetch_wine_quality_red()['data'], fetch_wine_quality_red()['target']
        elif dataset == 'wine_quality_white':
            X, Y = fetch_wine_quality_white()['data'], fetch_wine_quality_white()['target']
        else:
            raise ValueError(f"Dataset not supported: {dataset}")
        return X, Y


def fetch_parkinsons():
    if not os.path.isdir('data/parkinsons'):
        os.mkdir('data/parkinsons')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
        wget.download(url, out='data/parkinsons/')

    with open('data/parkinsons/parkinsons.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=0)
        Xy = {}
        Xy['data'] = df.values[:, 1:].astype('float')
        Xy['target'] = df.values[:, 0]

    return Xy


def fetch_climate_model_crashes():
    if not os.path.isdir('data/climate_model_crashes'):
        os.mkdir('data/climate_model_crashes')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat'
        wget.download(url, out='data/climate_model_crashes/')

    with open('data/climate_model_crashes/pop_failures.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=0)
        Xy = {}
        Xy['data'] = df.values[:, 2:-1]
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_concrete_compression():
    if not os.path.isdir('data/concrete_compression'):
        os.mkdir('data/concrete_compression')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
        wget.download(url, out='data/concrete_compression/')

    with open('data/concrete_compression/Concrete_Data.xls', 'rb') as f:
        df = pd.read_excel(io=f)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_yacht_hydrodynamics():
    if not os.path.isdir('data/yacht_hydrodynamics'):
        os.mkdir('data/yacht_hydrodynamics')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
        wget.download(url, out='data/yacht_hydrodynamics/')

    with open('data/yacht_hydrodynamics/yacht_hydrodynamics.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_airfoil_self_noise():
    if not os.path.isdir('data/airfoil_self_noise'):
        os.mkdir('data/airfoil_self_noise')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat'
        wget.download(url, out='data/airfoil_self_noise/')

    with open('data/airfoil_self_noise/airfoil_self_noise.dat', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1]
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_connectionist_bench_sonar():
    if not os.path.isdir('data/connectionist_bench_sonar'):
        os.mkdir('data/connectionist_bench_sonar')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
        wget.download(url, out='data/connectionist_bench_sonar/')

    with open('data/connectionist_bench_sonar/sonar.all-data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_ionosphere():
    if not os.path.isdir('data/ionosphere'):
        os.mkdir('data/ionosphere')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
        wget.download(url, out='data/ionosphere/')

    with open('data/ionosphere/ionosphere.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_qsar_biodegradation():
    if not os.path.isdir('data/qsar_biodegradation'):
        os.mkdir('data/qsar_biodegradation')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00254/biodeg.csv'
        wget.download(url, out='data/qsar_biodegradation/')

    with open('data/qsar_biodegradation/biodeg.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_seeds():
    if not os.path.isdir('data/seeds'):
        os.mkdir('data/seeds')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'
        wget.download(url, out='data/seeds/')

    with open('data/seeds/seeds_dataset.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_glass():
    if not os.path.isdir('data/glass'):
        os.mkdir('data/glass')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
        wget.download(url, out='data/glass/')

    with open('data/glass/glass.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_ecoli():
    if not os.path.isdir('data/ecoli'):
        os.mkdir('data/ecoli')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
        wget.download(url, out='data/ecoli/')

    with open('data/ecoli/ecoli.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_yeast():
    if not os.path.isdir('data/yeast'):
        os.mkdir('data/yeast')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data'
        wget.download(url, out='data/yeast/')

    with open('data/yeast/yeast.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_libras():
    if not os.path.isdir('data/libras'):
        os.mkdir('data/libras')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data'
        wget.download(url, out='data/libras/')

    with open('data/libras/movement_libras.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_planning_relax():
    if not os.path.isdir('data/planning_relax'):
        os.mkdir('data/planning_relax')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00230/plrx.txt'
        wget.download(url, out='data/planning_relax/')

    with open('data/planning_relax/plrx.txt', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_blood_transfusion():
    if not os.path.isdir('data/blood_transfusion'):
        os.mkdir('data/blood_transfusion')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data'
        wget.download(url, out='data/blood_transfusion/')

    with open('data/blood_transfusion/transfusion.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_breast_cancer_diagnostic():
    if not os.path.isdir('data/breast_cancer_diagnostic'):
        os.mkdir('data/breast_cancer_diagnostic')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
        wget.download(url, out='data/breast_cancer_diagnostic/')

    with open('data/breast_cancer_diagnostic/wdbc.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 2:].astype('float')
        Xy['target'] = df.values[:, 1]

    return Xy


def fetch_connectionist_bench_vowel():
    if not os.path.isdir('data/connectionist_bench_vowel'):
        os.mkdir('data/connectionist_bench_vowel')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/vowel/vowel-context.data'
        wget.download(url, out='data/connectionist_bench_vowel/')

    with open('data/connectionist_bench_vowel/vowel-context.data', 'rb') as f:
        df = pd.read_csv(f, delimiter='\s+', header=None)
        Xy = {}
        Xy['data'] = df.values[:, 3:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_concrete_slump():
    if not os.path.isdir('data/concrete_slump'):
        os.mkdir('data/concrete_slump')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
        wget.download(url, out='data/concrete_slump/')

    with open('data/concrete_slump/slump_test.data', 'rb') as f:
        df = pd.read_csv(f, delimiter=',')
        Xy = {}
        Xy['data'] = df.values[:, 1:-3].astype('float')
        Xy['target'] = df.values[:, -3:]

    return Xy


def fetch_wine_quality_red():
    if not os.path.isdir('data/wine_quality_red'):
        os.mkdir('data/wine_quality_red')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
        wget.download(url, out='data/wine_quality_red/')

    with open('data/wine_quality_red/winequality-red.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, 1:-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy


def fetch_wine_quality_white():
    if not os.path.isdir('data/wine_quality_white'):
        os.mkdir('data/wine_quality_white')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
        wget.download(url, out='data/wine_quality_white/')

    with open('data/wine_quality_white/winequality-white.csv', 'rb') as f:
        df = pd.read_csv(f, delimiter=';')
        Xy = {}
        Xy['data'] = df.values[:, :-1].astype('float')
        Xy['target'] = df.values[:, -1]

    return Xy

if __name__ == '__main__':
    for data in DATASETS:
        x, y = uci_dataset_loader(data)
        print(data, x.shape,y.shape)
#
# if name == 'california':
#     train_size, cal_size, test_size_iid, test_size_per_pattern = 10000, 1000, 1000, 100
# elif name == 'climate_model_crashes':
#     train_size, cal_size, test_size_iid, test_size_per_pattern = 340, 100, 100, 100
# elif name == 'concrete_compression':
#     train_size, cal_size, test_size_iid, test_size_per_pattern = 630, 200, 100, 100
# elif name == 'airfoil_self_noise':
#     train_size, cal_size, test_size_iid, test_size_per_pattern = 1203, 200, 100, 100
# elif name == 'qsar_biodegradation':
#     train_size, cal_size, test_size_iid, test_size_per_pattern = 755, 200, 100, 100
# elif name == 'yeast':
#     train_size, cal_size, test_size_iid, test_size_per_pattern = 1084, 200, 100, 100
# elif name == 'blood_transfusion':
#     train_size, cal_size, test_size_iid, test_size_per_pattern = 548, 100, 100, 100
# elif name == 'connectionist_bench_vowel':
#     train_size, cal_size, test_size_iid, test_size_per_pattern = 690, 200, 100, 100
# elif name == 'wine_quality_red':

# iris (150, 4) (150,)
# wine (178, 13) (178,)
# california (20640, 8) (20640,)
# parkinsons (195, 23) (195,)
# climate_model_crashes (540, 18) (540,)
# concrete_compression (1030, 8) (1030,)
# yacht_hydrodynamics (308, 6) (308,)
# airfoil_self_noise (1503, 5) (1503,)
# connectionist_bench_sonar (208, 60) (208,)
# ionosphere (351, 34) (351,)
# qsar_biodegradation (1055, 41) (1055,)
# seeds (210, 7) (210,)
# glass (214, 9) (214,)
# ecoli (336, 7) (336,)
# yeast (1484, 8) (1484,)
# libras (360, 90) (360,)
# planning_relax (182, 12) (182,)
# blood_transfusion (748, 4) (748,)
# breast_cancer_diagnostic (569, 30) (569,)
# connectionist_bench_vowel (990, 10) (990,)
# concrete_slump (103, 7) (103, 3)
# wine_quality_red (1599, 10) (1599,)
# wine_quality_white (4898, 11) (4898,)

