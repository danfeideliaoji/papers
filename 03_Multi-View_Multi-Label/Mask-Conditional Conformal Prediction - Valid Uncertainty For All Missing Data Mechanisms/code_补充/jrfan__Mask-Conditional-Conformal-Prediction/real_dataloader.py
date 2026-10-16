# ------------------------------------------------------------------------------
# This file includes code adapted from:
#   https://github.com/mzaffran/ConformalPredictionMissingValues
#   "Conformal Prediction with Missing Values" by M. Zaffran, A. Dieuleveut, J. Josse, and Y. Romano
#   (ICML 2023)

# License: MIT License (original repository)
# Some methods, including CP-MDA-Exact and CP-MDA-Nested, are based on the original implementation.
# ------------------------------------------------------------------------------

import random
import numpy as np
from tqdm import tqdm
import copy
import utils
import os
import pickle

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
import warnings
warnings.filterwarnings("ignore")

def get_setting(dim=3, params_reg={'regression':'Linear'}, params_noise={'noise':'Gaussian'}, params_missing={}):

    regression  = params_reg['regression']
    assert regression in ['Linear'], 'regression must be Linear.'

    if 'mean' in params_reg:
        mean = params_reg['mean']
    else:
        mean = 1
    if 'phi' in params_reg:
        scale = params_reg['phi']
    else:
        scale = 1
    if params_reg['beta'] is not None:
        beta = params_reg['beta']
    else:
        beta = np.full(dim,1)
    if dim < 10:
        name = 'Linear_d_'+str(dim)+'_beta_'+'_'.join(str(x) for x in beta)+'_Gaussian_Mean_'+str(mean)
    else:
        name = 'Linear_d_'+str(dim)+'_beta_varies_Gaussian_Mean_'+str(mean)+'_Scale_'+str(scale)

    if 'prob_missing' in params_missing:
        prob_missing = params_missing['prob_missing']
    else:
        prob_missing = 0.2

    if 'mechanism' in params_missing:
        if params_missing['mechanism'] == 'MNAR_mask_quantiles':
            name = name + '_' + params_missing['mechanism'] + '_q_' + str(params_missing['q']) + '_'
        else:
            name = name + '_' + params_missing['mechanism'] + '_'
            if 'id_setting' in params_missing:
                name = name + 'id_'+str(params_missing['id_setting'])+'_'
        name += ''.join([str(i) for i in params_missing['var_missing']]) + '_'
    else:
        name = name + '_MCAR_'

    name = name + str(prob_missing)

    return name

def get_name_data_imputed(train_size, cal_size, params_test, imputation,
                          dim=3, params_reg={}, params_noise={}, dataset=None, params_missing={}, seed=1):
    """
    Parameters
    ----------
    n : experiment sample size
    dim : dimension of the covariates (i.e. X lies in R^dim)
    regression : regression model, should be Linear
    noise : noise type, can be Gaussian
    params_reg : parameters of the regression part
    params_noise : parameters of the noise, e.g. a dictionary {'ar': [1, ar1], 'ma':[1]}
                   to generate an AR(1) noise with coefficient -ar1
    seed : random seed for reproducibility used in the experiment

    Returns
    -------
    name : name of the file containing (if existing)
    the generated data with the given parameters of simulations
    """

    name = get_name_data(train_size, cal_size, params_test, dim=dim,
                         params_reg=params_reg, params_noise=params_noise, dataset=dataset, params_missing=params_missing, seed=seed)

    if imputation is not None:
        name = name + '_imputation_' + imputation

    return name

def get_name_data(train_size, cal_size, params_test, dim=3,
                  params_reg={}, params_noise={}, dataset=None, params_missing={}, seed=1):
    """
    Parameters
    ----------
    n : experiment sample size
    dim : dimension of the covariates (i.e. X lies in R^dim)
    regression : regression model, should be Linear
    noise : noise type, can be Gaussian
    params_reg : parameters of the regression part
    params_noise : parameters of the noise, e.g. a dictionary {'ar': [1, ar1], 'ma':[1]}
                   to generate an AR(1) noise with coefficient -ar1
    seed : random seed for reproducibility used in the experiment

    Returns
    -------
    name : name of the file containing (if existing)
    the generated data with the given parameters of simulations
    """

    max_test_size = np.max(params_test['test_size'])

    if dataset is None:

        regression = params_reg['regression']

        assert regression in ['Linear'], 'regression must be Linear.'

        name = get_setting(dim=dim, params_reg=params_reg, params_noise=params_noise, params_missing=params_missing)

    else:
        name = dataset

    name = name + '_seed_' + str(seed) + '_train_' + str(train_size) + '_cal_' + str(cal_size) + '_test_' + str(max_test_size)

    if 'prob_missing' in list(params_missing.keys()):
        name = name + '_prob_' + str(params_missing['prob_missing'])

    return name

def GetDataset(name, base_path):
    if name == "concrete_compression":
        dataset = np.loadtxt(open(base_path + 'Concrete_Data.csv', "rb"), delimiter=",", skiprows=1)
        data = pd.DataFrame(data=dataset)
        response_name = 8
        d = data.shape[1] - 1
        continuous_var = np.full(d, 1)

    if name == "bike":
        # https://www.kaggle.com/rajmehra03/bike-sharing-demand-rmsle-0-3194
        df = pd.read_csv(base_path + 'bike_train.csv')

        # # seperating season as per values. this is bcoz this will enhance features.
        season = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season], axis=1)

        # # # same for weather. this is bcoz this will enhance features.
        weather = pd.get_dummies(df['weather'], prefix='weather')
        df = pd.concat([df, weather], axis=1)

        # # # now can drop weather and season.
        df.drop(['season', 'weather'], inplace=True, axis=1)
        df.head()

        df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
        df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
        df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
        df['year'] = df['year'].map({2011: 0, 2012: 1})

        df.drop('datetime', axis=1, inplace=True)
        df.drop(['casual', 'registered'], axis=1, inplace=True)
        df.columns.to_series().groupby(df.dtypes).groups

        features_names = ['temp', 'atemp', 'humidity', 'windspeed', 'holiday', 'workingday',
                          'season_1', 'season_2', 'season_3', 'season_4', 'weather_1',
                          'weather_2', 'weather_3', 'weather_4', 'hour', 'day', 'month', 'year']
        response_name = 'count'

        d = len(features_names)
        continuous_var = np.append(np.full(4, 1), np.full(d - 5, 0))

        col_names = np.append(features_names, response_name)

        data = df[col_names]

    if name == "meps_19":
        df = pd.read_csv(base_path + 'meps_19_reg.csv')

        response_name = "UTILIZATION_reg"

        features_names = ['AGE', 'PCS42', 'MCS42', 'K6SUM42', 'PERWT15F', 'REGION=1',
                          'REGION=2', 'REGION=3', 'REGION=4', 'SEX=1', 'SEX=2', 'MARRY=1',
                          'MARRY=2', 'MARRY=3', 'MARRY=4', 'MARRY=5', 'MARRY=6', 'MARRY=7',
                          'MARRY=8', 'MARRY=9', 'MARRY=10', 'FTSTU=-1', 'FTSTU=1', 'FTSTU=2',
                          'FTSTU=3', 'ACTDTY=1', 'ACTDTY=2', 'ACTDTY=3', 'ACTDTY=4',
                          'HONRDC=1', 'HONRDC=2', 'HONRDC=3', 'HONRDC=4', 'RTHLTH=-1',
                          'RTHLTH=1', 'RTHLTH=2', 'RTHLTH=3', 'RTHLTH=4', 'RTHLTH=5',
                          'MNHLTH=-1', 'MNHLTH=1', 'MNHLTH=2', 'MNHLTH=3', 'MNHLTH=4',
                          'MNHLTH=5', 'HIBPDX=-1', 'HIBPDX=1', 'HIBPDX=2', 'CHDDX=-1',
                          'CHDDX=1', 'CHDDX=2', 'ANGIDX=-1', 'ANGIDX=1', 'ANGIDX=2',
                          'MIDX=-1', 'MIDX=1', 'MIDX=2', 'OHRTDX=-1', 'OHRTDX=1', 'OHRTDX=2',
                          'STRKDX=-1', 'STRKDX=1', 'STRKDX=2', 'EMPHDX=-1', 'EMPHDX=1',
                          'EMPHDX=2', 'CHBRON=-1', 'CHBRON=1', 'CHBRON=2', 'CHOLDX=-1',
                          'CHOLDX=1', 'CHOLDX=2', 'CANCERDX=-1', 'CANCERDX=1', 'CANCERDX=2',
                          'DIABDX=-1', 'DIABDX=1', 'DIABDX=2', 'JTPAIN=-1', 'JTPAIN=1',
                          'JTPAIN=2', 'ARTHDX=-1', 'ARTHDX=1', 'ARTHDX=2', 'ARTHTYPE=-1',
                          'ARTHTYPE=1', 'ARTHTYPE=2', 'ARTHTYPE=3', 'ASTHDX=1', 'ASTHDX=2',
                          'ADHDADDX=-1', 'ADHDADDX=1', 'ADHDADDX=2', 'PREGNT=-1', 'PREGNT=1',
                          'PREGNT=2', 'WLKLIM=-1', 'WLKLIM=1', 'WLKLIM=2', 'ACTLIM=-1',
                          'ACTLIM=1', 'ACTLIM=2', 'SOCLIM=-1', 'SOCLIM=1', 'SOCLIM=2',
                          'COGLIM=-1', 'COGLIM=1', 'COGLIM=2', 'DFHEAR42=-1', 'DFHEAR42=1',
                          'DFHEAR42=2', 'DFSEE42=-1', 'DFSEE42=1', 'DFSEE42=2',
                          'ADSMOK42=-1', 'ADSMOK42=1', 'ADSMOK42=2', 'PHQ242=-1', 'PHQ242=0',
                          'PHQ242=1', 'PHQ242=2', 'PHQ242=3', 'PHQ242=4', 'PHQ242=5',
                          'PHQ242=6', 'EMPST=-1', 'EMPST=1', 'EMPST=2', 'EMPST=3', 'EMPST=4',
                          'POVCAT=1', 'POVCAT=2', 'POVCAT=3', 'POVCAT=4', 'POVCAT=5',
                          'INSCOV=1', 'INSCOV=2', 'INSCOV=3', 'RACE']

        d = len(features_names)
        continuous_var = np.append(np.full(5, 1), np.full(d - 5, 0))

        col_names = np.append(features_names, response_name)

        data = df[col_names]

    return data, response_name, continuous_var

def none_or_value(x,y):
    if x is None or x==0:
        return y
    return x

def process_test(params_test, d, params_missing={}):

    test_sizes = []
    mechanisms_test = []

    for mechanism in list(params_test.keys()):
        assert mechanism in ['iid', 'worst_pattern',
                             'best_pattern', 'test_pattern',
                             'fixed_nb_sample_pattern', 'fixed_nb_sample_pattern_size'], 'Test mechanism should be among iid, worst_pattern, best_pattern, test_pattern, fixed_nb_sample_pattern, fixed_nb_sample_pattern_size.'
        mechanisms_test = np.append(mechanisms_test, mechanism)
        if mechanism not in ['fixed_nb_sample_pattern', 'fixed_nb_sample_pattern_size']:
            assert 'test_size' in list(params_test[mechanism].keys()), 'test_size should be provided for each test mechanism.'
            test_sizes = np.append(test_sizes, int(params_test[mechanism]['test_size']))
        else:
            assert 'nb_sample_pattern' in list(params_test[mechanism].keys()), 'nb_sample_pattern should be provided for fixed_nb_sample_pattern mechanism.'
            nb_sample_pattern = params_test[mechanism]['nb_sample_pattern']

            if 'var_missing' in params_missing:
                var_missing = params_missing['var_missing']
            else:
                var_missing = np.full(d, 1)

            if mechanism == 'fixed_nb_sample_pattern':

                list_patterns = utils.create_patterns(d, var_missing)
                nb_pattern = len(list_patterns)
                test_size = nb_sample_pattern*nb_pattern
                test_sizes = np.append(test_sizes, int(test_size))
                params_test[mechanism]['test_size'] = test_size

            else:
                nb_pattern_size = np.sum(var_missing)
                test_size = nb_sample_pattern * nb_pattern_size
                test_sizes = np.append(test_sizes, int(test_size))
                params_test[mechanism]['test_size'] = test_size

    test_sizes = np.unique(test_sizes).astype(int)

    params_test['test_size'] = test_sizes
    params_test['mechanisms_test'] = mechanisms_test

    return params_test

def load_file(parent, name, ext):

    assert ext in ['pkl', 'xz'], 'ext must be pkl or xz.'
    path = parent + '/' + name + '.' + ext
    if ext == 'pkl':
        with open(path,'rb') as f:
            file = pickle.load(f)
    elif ext == 'xz':
        with lzma.open(path,'rb') as f:
            file = pickle.load(f)
    return file

class Semi_synthetic_data():

    def __init__(self,
                 n_rep=10,
                 prob_missing=0.2,
                 dataset_name='meps_19',
                 imputation='None',
                 seed=None,
                 cal_size=None,
                 mechanism='MCAR',
                 var_missing_new=None):
        self.n_rep = n_rep
        self.mechanism = mechanism
        dataset_base_path = "./data/cqr_datasets/"

        datasets_sizes = {'meps_19': {'train': 1000, 'cal': none_or_value(cal_size, 200), 'test_pattern': 100},
                          'meps_20': {'train': 1000, 'cal': none_or_value(cal_size, 500), 'test_pattern': 100},
                          'meps_21': {'train': 1000, 'cal': none_or_value(cal_size, 500), 'test_pattern': 100},
                          'bio': {'train': 1000, 'cal': none_or_value(cal_size, 500), 'test_pattern': 100},
                          'concrete_compression': {'train': 630, 'cal': none_or_value(cal_size, 200), 'test_pattern': 100},
                          'bike': {'train': 8000, 'cal': none_or_value(cal_size, 500), 'test_pattern': 100}}
        df, target, var_missing = GetDataset(dataset_name, dataset_base_path)
        if mechanism == 'MCAR':
            self.params_missing = {'prob_missing': prob_missing,
                                   'var_missing': var_missing,
                                   'mechanism':mechanism}
        else:
            self.params_missing = {'prob_missing': prob_missing,
                                   'var_missing': var_missing_new,
                                   'mechanism':mechanism}
        print(mechanism, self.params_missing)
        self.d = df.shape[1] - 1
        if dataset_name == 'concrete_compression':
            nb_sample_pattern = datasets_sizes[dataset_name]['test_pattern']
            self.params_test = {'iid': {'test_size': 200},
                           'fixed_nb_sample_pattern': {'nb_sample_pattern': nb_sample_pattern}}
        else:
            nb_sample_pattern = datasets_sizes[dataset_name]['test_pattern']
            self.params_test = {'iid': {'test_size': 2000},
                           'fixed_nb_sample_pattern': {'nb_sample_pattern': nb_sample_pattern},
                                'fixed_nb_sample_pattern_size': {'nb_sample_pattern': 100}
                                }
        self.test_size_iid = self.params_test['iid']['test_size']
        self.params_test = process_test(self.params_test, d=self.d, params_missing=self.params_missing)

        max_test_size = np.max(self.params_test['test_size'])

        train_size = datasets_sizes[dataset_name]['train']
        self.train_size = train_size
        cal_size = datasets_sizes[dataset_name]['cal']
        self.cal_size = cal_size

        name = get_name_data(train_size, cal_size, self.params_test,
                             dataset=dataset_name, params_missing=self.params_missing, seed=n_rep)
        if mechanism == 'MCAR':
            pass
        else:
            name = 'MAR_'+name
        print('name: ', name)
        if os.path.isfile('data/' + name + '.pkl'):
            print('data found')
            data = load_file('data', name, 'pkl')
        else:
            print('data not found')
            X, X_missing, M, Y = generate_multiple_real_data_MCAR(df, target, train_size=train_size,
                                                                      cal_size=cal_size, params_test=self.params_test,
                                                                      params_missing=self.params_missing, seed_max=n_rep)
            data = {'X': X, 'X_missing': X_missing, 'M': M, 'Y': Y}
            write_file('data', name, 'pkl', data)

        name_imputed = get_name_data_imputed(train_size, cal_size, self.params_test, imputation=imputation,
                                           dataset=dataset_name, params_missing=self.params_missing, seed=n_rep)
        if mechanism == 'MCAR':
            pass
        else:
            name_imputed = 'MAR_' + name_imputed

        if os.path.isfile('data/' + name_imputed + '.pkl'):
            print('imputation found')
            X_imp = load_file('data', name_imputed, 'pkl')
        else:
            print('imputation not found')
            if imputation == 'complete':
                X_imp = data['X']
            else:
                X_imp = impute(data, imputation)
            print('impute fini !')
            write_file('data', name_imputed, 'pkl', X_imp)
        self.X = data['X']
        self.X_missing = data['X_missing']
        self.X_imputed = X_imp
        self.M = data['M']
        self.Y = data['Y']
    def __getitem__(self, key):
        return self.__dict__[key]

def generate_multiple_real_data_MCAR(dataframe, target, train_size, cal_size, params_test, params_missing={}, seed_max=1):

    data_features = dataframe.loc[:, dataframe.columns != target]
    response = dataframe.loc[:, target]

    sets = ['Train', 'Cal', 'Test']
    mechanisms_test = params_test['mechanisms_test']
    max_test_size = np.max(params_test['test_size'])

    n = dataframe.shape[0]

    X = dict.fromkeys(sets)
    X_missing = dict.fromkeys(sets)
    M = dict.fromkeys(sets)
    Y = dict.fromkeys(sets)

    for k in range(seed_max):

        random.seed(k)
        np.random.seed(k)

        ido = random.sample(range(n), n)

        Xk = np.array(data_features.iloc[ido,:])
        Yk = np.array(response[ido])

        data = {'X': Xk, 'Y':Yk}

        Xk, Yk = generate_split(train_size, cal_size, params_test, data)
        if params_missing['mechanism']=='MCAR':
            Xk_missing, Mk_missing = generate_MCAR(Xk, params_test, params_missing, seed=k)
        elif params_missing['mechanism']=='MAR':
            Xk_missing, Mk_missing = generate_MAR(Xk, params_test, params_missing, seed=k)
        else:
            raise NotImplementedError

        for set in ['Train', 'Cal']:
            if k == 0:
                X[set] = np.expand_dims(Xk[set], axis=0)
                X_missing[set] = np.expand_dims(Xk_missing[set], axis=0)
                M[set] = np.expand_dims(Mk_missing[set], axis=0)
                Y[set] = Yk[set]
            else:
                X[set] = np.vstack((X[set], np.expand_dims(Xk[set], axis=0)))
                X_missing[set] = np.vstack((X_missing[set], np.expand_dims(Xk_missing[set], axis=0)))
                M[set] = np.vstack((M[set], np.expand_dims(Mk_missing[set], axis=0)))
                Y[set] = np.vstack((Y[set], np.array(Yk[set])))

        set = 'Test'
        if k == 0:
            X[set] = dict.fromkeys(mechanisms_test)
            X_missing[set] = dict.fromkeys(mechanisms_test)
            M[set] = dict.fromkeys(mechanisms_test)
            Y[set] = dict.fromkeys(mechanisms_test)
            for key in mechanisms_test:
                n_test = params_test[key]['test_size']
                X[set][key] = np.expand_dims(Xk[set][n_test], axis=0)
                X_missing[set][key] = np.expand_dims(Xk_missing[set][key], axis=0)
                M[set][key] = np.expand_dims(Mk_missing[set][key], axis=0)
                Y[set][key] = Yk[set][n_test]
        else:
            for key in mechanisms_test:
                n_test = params_test[key]['test_size']
                X[set][key] = np.vstack((X[set][key], np.expand_dims(Xk[set][n_test], axis=0)))
                X_missing[set][key] = np.vstack((X_missing[set][key], np.expand_dims(Xk_missing[set][key], axis=0)))
                M[set][key] = np.vstack((M[set][key], np.expand_dims(Mk_missing[set][key], axis=0)))
                Y[set][key] = np.vstack((Y[set][key], np.array(Yk[set][n_test])))

    return X, X_missing, M, Y

def generate_MCAR(X, params_test, params_missing={}, seed=1):

    random.seed(seed)
    np.random.seed(seed)

    d = X['Train'].shape[1]

    if 'prob_missing' in params_missing:
        prob_missing = params_missing['prob_missing']
    else:
        prob_missing = 0.2
    if 'var_missing' in params_missing:
        var_missing = params_missing['var_missing']
    else:
        var_missing = np.full(d, 1)

    nb_var_missing = np.sum(var_missing)

    train_size = X['Train'].shape[0]
    cal_size = X['Cal'].shape[0]

    M_mcar_train = np.full(X['Train'].shape, False)
    X_mcar_train = copy.deepcopy(X['Train'])

    M_mcar_cal = np.full(X['Cal'].shape, False)
    X_mcar_cal = copy.deepcopy(X['Cal'])

    M_mcar_train[:,np.where(np.array(var_missing) == 1)[0]] = (np.random.uniform(low=0,high=1,size=(train_size,nb_var_missing)) <= (prob_missing))
    X_mcar_train[M_mcar_train] = np.nan
    M_mcar_cal[:,np.where(np.array(var_missing) == 1)[0]] = (np.random.uniform(low=0,high=1,size=(cal_size,nb_var_missing)) <= (prob_missing))
    X_mcar_cal[M_mcar_cal] = np.nan

    mechanisms_test = params_test['mechanisms_test']

    M_mcar = {'Train': M_mcar_train, 'Cal': M_mcar_cal}
    M_mcar_test = dict.fromkeys(mechanisms_test)

    X_mcar = {'Train': X_mcar_train, 'Cal': X_mcar_cal}
    X_mcar_test = dict.fromkeys(mechanisms_test)

    if 'iid' in mechanisms_test:
        test_size = params_test['iid']['test_size']
        M_mcar_iid = np.full((test_size, d), False)
        M_mcar_iid[:,np.where(np.array(var_missing) == 1)[0]] = (np.random.uniform(low=0,high=1,size=(test_size,nb_var_missing)) <= (prob_missing))
        M_mcar_test['iid'] = M_mcar_iid
        X_mcar_iid = copy.deepcopy(X['Test'][test_size])
        X_mcar_iid[M_mcar_iid] = np.nan
        X_mcar_test['iid'] = X_mcar_iid
    for extreme in ['worst_pattern', 'best_pattern']:
        if extreme in mechanisms_test:
            test_size = params_test[extreme]['test_size']
            test_pattern = params_test[extreme]['pattern']
            M_mcar_extreme = np.full((test_size, d), False)
            M_mcar_extreme[:,np.where(np.array(test_pattern) == 1)[0]] = 1
            M_mcar_test[extreme] = M_mcar_extreme
            X_mcar_extreme = copy.deepcopy(X['Test'][test_size])
            X_mcar_extreme[M_mcar_extreme] = np.nan
            X_mcar_test[extreme] = X_mcar_extreme
    if 'fixed_nb_sample_pattern' in mechanisms_test:
        list_patterns = utils.create_patterns(d, var_missing)
        test_size = params_test['fixed_nb_sample_pattern']['test_size']
        nb_sample_pattern = params_test['fixed_nb_sample_pattern']['nb_sample_pattern']
        M_mcar_fixed_sample_pattern = np.full((test_size, d), False)
        X_mcar_fixed_sample_pattern = copy.deepcopy(X['Test'][test_size])
        for idp, pattern in enumerate(list_patterns):
            M_mcar_fixed_sample_pattern[(idp*nb_sample_pattern):((idp+1)*nb_sample_pattern),np.where(np.array(pattern) == 1)[0]] = 1
        X_mcar_fixed_sample_pattern[M_mcar_fixed_sample_pattern] = np.nan
        M_mcar_test['fixed_nb_sample_pattern'] = M_mcar_fixed_sample_pattern
        X_mcar_test['fixed_nb_sample_pattern'] = X_mcar_fixed_sample_pattern
    if 'fixed_nb_sample_pattern_size' in mechanisms_test:
        list_pattern_sizes = np.arange(np.sum(var_missing))
        test_size = params_test['fixed_nb_sample_pattern_size']['test_size']
        nb_sample_pattern_size = params_test['fixed_nb_sample_pattern_size']['nb_sample_pattern']
        M_mcar_fixed_sample_pattern_size = np.full((test_size, d), False)
        X_mcar_fixed_sample_pattern_size = copy.deepcopy(X['Test'][test_size])

        list_patterns = utils.create_patterns(d, var_missing)
        size_to_ids = dict.fromkeys(np.arange(0, d))
        for k in np.arange(0, d):
            size_to_ids[k] = []
        for pattern in list_patterns:
            key_pattern = utils.pattern_to_id(pattern)
            size_pattern = utils.pattern_to_size(pattern)
            size_to_ids[size_pattern] = np.append(size_to_ids[size_pattern], key_pattern)

        for idp, pattern_size in enumerate(list_pattern_sizes):
            keys = random.choices(size_to_ids[pattern_size], k=nb_sample_pattern_size)
            unique_keys, count_keys = np.unique(keys, return_counts=True)
            min_ind = idp * nb_sample_pattern_size
            for idps, key in enumerate(unique_keys):
                nb_sample_pattern = count_keys[idps]
                pattern = utils.bin_to_vec(bin(int(key)), d)
                M_mcar_fixed_sample_pattern_size[min_ind:(min_ind+nb_sample_pattern),np.where(np.array(pattern) == 1)[0]] = 1
                min_ind = min_ind + nb_sample_pattern
        X_mcar_fixed_sample_pattern_size[M_mcar_fixed_sample_pattern_size] = np.nan
        M_mcar_test['fixed_nb_sample_pattern_size'] = M_mcar_fixed_sample_pattern_size
        X_mcar_test['fixed_nb_sample_pattern_size'] = X_mcar_fixed_sample_pattern_size

    X_mcar['Test'] = X_mcar_test
    M_mcar['Test'] = M_mcar_test

    return X_mcar, M_mcar

def generate_split(train_size, cal_size, params_test, data):

    X = data['X']
    X_train = X[:train_size,:]
    X_cal = X[train_size:(train_size+cal_size),:]

    Y = data['Y']
    Y_train = Y[:train_size]
    Y_cal = Y[train_size:(train_size+cal_size)]

    test_size = params_test['test_size']

    mechanisms_test = params_test['mechanisms_test']

    #if test_size is list:
    X_test = dict.fromkeys(test_size)
    Y_test = dict.fromkeys(test_size)
    for n_test in test_size:
        if (train_size+cal_size+n_test) <= X.shape[0]:
            X_test[n_test] = X[(train_size+cal_size):(train_size+cal_size+n_test),:]
            Y_test[n_test] = Y[(train_size+cal_size):(train_size+cal_size+n_test)]
        else:
            if 'iid' in mechanisms_test:
                assert params_test['iid']['test_size'] != n_test
            for extreme in ['worst_pattern', 'best_pattern']:
                if extreme in mechanisms_test:
                    assert params_test[extreme]['test_size'] != n_test
            for fixed in ['fixed_nb_sample_pattern','fixed_nb_sample_pattern_size']:
                if fixed in mechanisms_test and params_test[fixed]['test_size'] == n_test:
                    assert (train_size + cal_size + params_test[fixed]['nb_sample_pattern']) <= X.shape[0]

            X_test_created = np.empty((n_test,X.shape[1]))
            Y_test_created = np.empty((n_test))

            X_to_shuffle = copy.deepcopy(X[(train_size+cal_size):,:])
            Y_to_shuffle = copy.deepcopy(Y[(train_size+cal_size):])
            n_shuffle = X_to_shuffle.shape[0]
            nb_exact_shuffle = n_test//n_shuffle
            nb_rest = n_test%n_shuffle
            for k in range(nb_exact_shuffle):
                ido = random.sample(range(n_shuffle), n_shuffle)
                X_test_created[(k * n_shuffle):((k+1) * n_shuffle), :] = X_to_shuffle[ido, :]
                Y_test_created[(k * n_shuffle):((k + 1) * n_shuffle)] = Y_to_shuffle[ido]
            ido = random.sample(range(n_shuffle), nb_rest)
            X_test_created[((k+1) * n_shuffle):, :] = X_to_shuffle[ido, :]
            Y_test_created[((k+1) * n_shuffle):] = Y_to_shuffle[ido]
            X_test[n_test] = X_test_created
            Y_test[n_test] = Y_test_created

    X_split = {'Train': X_train, 'Cal': X_cal, 'Test': X_test}
    Y_split = {'Train': Y_train, 'Cal': Y_cal, 'Test': Y_test}

    return X_split, Y_split

def generate_MAR(X, params_test, params_missing={}, seed=1):

    random.seed(seed)
    np.random.seed(seed)

    d = X['Train'].shape[1]

    if 'prob_missing' in params_missing:
        prob_missing = params_missing['prob_missing']
    else:
        prob_missing = 0.2
    if 'var_missing' in params_missing:
        var_missing = np.array(params_missing['var_missing'])
    else:
        var_missing = np.full(d, 1)
    # print(params_missing,var_missing)

    nb_var_missing = np.sum(var_missing)

    train_size = X['Train'].shape[0]
    cal_size = X['Cal'].shape[0]

    # M_mar_train = np.full(X['Train'].shape, False)
    X_mar_train = copy.deepcopy(X['Train'])

    # M_mar_cal = np.full(X['Cal'].shape, False)
    X_mar_cal = copy.deepcopy(X['Cal'])

    M_mar_train = MAR_mask(X_mar_train, prob_missing, var_missing)
    X_mar_train[M_mar_train] = np.nan
    M_mar_cal = MAR_mask(X_mar_cal, prob_missing, var_missing)
    X_mar_cal[M_mar_cal] = np.nan


    mechanisms_test = params_test['mechanisms_test']

    M_mar = {'Train': M_mar_train, 'Cal': M_mar_cal}
    M_mar_test = dict.fromkeys(mechanisms_test)

    X_mar = {'Train': X_mar_train, 'Cal': X_mar_cal}
    X_mar_test = dict.fromkeys(mechanisms_test)

    if 'iid' in mechanisms_test:
        test_size = params_test['iid']['test_size']
        X_mar_iid = copy.deepcopy(X['Test'][test_size])

        M_mar_iid = MAR_mask(X_mar_iid, prob_missing, var_missing)
        M_mar_test['iid'] = M_mar_iid

        X_mar_iid[M_mar_iid] = np.nan
        X_mar_test['iid'] = X_mar_iid
    for extreme in ['worst_pattern', 'best_pattern']:
        if extreme in mechanisms_test:
            test_size = params_test[extreme]['test_size']
            test_pattern = params_test[extreme]['pattern']
            M_mar_extreme = np.full((test_size, d), False)
            M_mar_extreme[:,np.where(np.array(test_pattern) == 1)[0]] = 1
            M_mar_test[extreme] = M_mar_extreme
            X_mar_extreme = copy.deepcopy(X['Test'][test_size])
            X_mar_extreme[M_mar_extreme] = np.nan
            X_mar_test[extreme] = X_mar_extreme
    if 'fixed_nb_sample_pattern' in mechanisms_test:
        list_patterns = utils.create_patterns(d, var_missing)
        test_size = params_test['fixed_nb_sample_pattern']['test_size']
        nb_sample_pattern = params_test['fixed_nb_sample_pattern']['nb_sample_pattern']
        M_mar_fixed_sample_pattern = np.full((test_size, d), False)
        X_mar_fixed_sample_pattern = copy.deepcopy(X['Test'][test_size])
        for idp, pattern in enumerate(list_patterns):
            M_mar_fixed_sample_pattern[(idp*nb_sample_pattern):((idp+1)*nb_sample_pattern),np.where(np.array(pattern) == 1)[0]] = 1
        X_mar_fixed_sample_pattern[M_mar_fixed_sample_pattern] = np.nan
        M_mar_test['fixed_nb_sample_pattern'] = M_mar_fixed_sample_pattern
        X_mar_test['fixed_nb_sample_pattern'] = X_mar_fixed_sample_pattern
    if 'fixed_nb_sample_pattern_size' in mechanisms_test:
        list_pattern_sizes = np.arange(np.sum(var_missing))
        test_size = params_test['fixed_nb_sample_pattern_size']['test_size']
        nb_sample_pattern_size = params_test['fixed_nb_sample_pattern_size']['nb_sample_pattern']
        M_mcar_fixed_sample_pattern_size = np.full((test_size, d), False)
        X_mcar_fixed_sample_pattern_size = copy.deepcopy(X['Test'][test_size])

        list_patterns = utils.create_patterns(d, var_missing)
        size_to_ids = dict.fromkeys(np.arange(0, d))
        for k in np.arange(0, d):
            size_to_ids[k] = []
        for pattern in list_patterns:
            key_pattern = utils.pattern_to_id(pattern)
            size_pattern = utils.pattern_to_size(pattern)
            size_to_ids[size_pattern] = np.append(size_to_ids[size_pattern], key_pattern)

        for idp, pattern_size in enumerate(list_pattern_sizes):
            keys = random.choices(size_to_ids[pattern_size], k=nb_sample_pattern_size)
            unique_keys, count_keys = np.unique(keys, return_counts=True)
            min_ind = idp * nb_sample_pattern_size
            for idps, key in enumerate(unique_keys):
                nb_sample_pattern = count_keys[idps]
                pattern = utils.bin_to_vec(bin(int(key)), d)
                M_mcar_fixed_sample_pattern_size[min_ind:(min_ind+nb_sample_pattern),np.where(np.array(pattern) == 1)[0]] = 1
                min_ind = min_ind + nb_sample_pattern
        X_mcar_fixed_sample_pattern_size[M_mcar_fixed_sample_pattern_size] = np.nan
        M_mar_test['fixed_nb_sample_pattern_size'] = M_mcar_fixed_sample_pattern_size
        X_mar_test['fixed_nb_sample_pattern_size'] = X_mcar_fixed_sample_pattern_size

    X_mar['Test'] = X_mar_test
    M_mar['Test'] = M_mar_test

    return X_mar, M_mar

def MAR_mask(X, p, incomplete_vars, depend_vars=None):
    n, d = X.shape

    mask = np.zeros((n, d), dtype=bool)

    idxs_nas = np.array([i for i in range(d) if incomplete_vars[i] == 1])
    if depend_vars is None:
        idxs_obs = np.array([i for i in range(d) if i not in idxs_nas])
    else:
        idxs_obs = np.array(depend_vars)

    # Pick coefficients for the logistic model
    coeffs = utils.pick_coeffs(X, idxs_obs, idxs_nas)

    # Pick the intercepts to achieve the desired missingness level
    intercepts = utils.fit_intercepts(X[:, idxs_obs], coeffs, p)

    # Compute probabilities
    ps = 1 / (1 + np.exp(-(X[:, idxs_obs] @ coeffs + intercepts)))
    # Generate mask
    ber = np.random.uniform(size=(n, len(idxs_nas)))
    mask[:, idxs_nas] = ber < ps


    return mask

def impute(data, imputation):

    X_missing = data['X_missing']

    imputer = get_impute(imputation)

    n_rep = X_missing['Train'].shape[0]

    X_train_imp = np.empty(X_missing['Train'].shape)
    X_cal_imp = np.empty(X_missing['Cal'].shape)

    print('collect pattern')
    if type(X_missing['Test']) is dict:
        multiple_test = True
        keys_test = list(X_missing['Test'].keys())
        X_test_imp = dict.fromkeys(keys_test)
        for key in keys_test:
            X_test_imp[key] = np.empty(X_missing['Test'][key].shape)
    else:
        multiple_test = False
        X_test_imp = np.empty(X_missing['Test'].shape)

    for k in range(n_rep):
        print('imputing {} sur {}'.format(k, n_rep))
        imputer.fit(X_missing['Train'][k,:,:])
        print('fit finished')
        X_train_imp[k,:,:] = imputer.transform(X_missing['Train'][k,:,:])
        X_cal_imp[k,:,:] = imputer.transform(X_missing['Cal'][k,:,:])
        print('train and cal transformed')
        if multiple_test:
            for key in keys_test:
                X_test_imp[key][k,:,:] = imputer.transform(X_missing['Test'][key][k,:,:])
        else:
            X_test_imp[k,:,:] = imputer.transform(X_missing['Test'][k,:,:])
    X_imputed = {'Train': X_train_imp, 'Cal': X_cal_imp, 'Test': X_test_imp}

    return X_imputed

def get_impute(imputation):
    print('Initialize imputation')
    class No_impute():
        def __init__(self):
            pass
        def fit(self, x):
            return self
        def transform(self, x):
            return x

    if imputation in ['mean', 'constant']:
        imputer = SimpleImputer(missing_values=np.nan, strategy=imputation)
    elif imputation == 'MICE':
        imputer = IterativeImputer(missing_values=np.nan, sample_posterior=True, max_iter=20)
    elif imputation == 'iterative_ridge':
        imputer = IterativeImputer(missing_values=np.nan, sample_posterior=False)
    elif imputation == 'None':
        imputer = No_impute()
    elif imputation == 'missforest':
        from sklearn.ensemble import RandomForestRegressor
        imputer = IterativeImputer(estimator= RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        bootstrap=True,
        n_jobs=-1,
        random_state=0,
    ),missing_values=np.nan, sample_posterior=False)
    elif imputation =='Drf':
        from models import Drf_to_impute
        imputer = IterativeImputer(estimator=Drf_to_impute(),missing_values=np.nan, sample_posterior=True)
    else:
        print(imputation,)
        raise NotImplementedError
    return imputer


def write_file(parent, name, ext, file):

    assert ext in ['pkl', 'xz'], 'ext must be pkl or xz.'
    path = parent + '/' + name + '.' + ext
    if ext == 'pkl':
        if not os.path.isdir(parent):
            os.makedirs(parent)
        with open(path,'wb') as f:
            pickle.dump(file, f)
    elif ext == 'xz':
        if not os.path.isdir(parent):
            os.makedirs(parent)
        with lzma.open(path,'wb') as f:
            pickle.dump(file, f)