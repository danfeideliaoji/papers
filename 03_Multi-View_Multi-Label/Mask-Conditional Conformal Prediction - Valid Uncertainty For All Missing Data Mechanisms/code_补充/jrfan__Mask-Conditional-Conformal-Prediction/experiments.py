import utils
import numpy as np
from models import Data_for_mask, Miss_proba_simple_model, fit_basemodel,predict_basemodel, Data_for_mask_lr
import pickle
import wandb
from tqdm import tqdm
import copy
import warnings
import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

class Missing_experiments():

    def __init__(self, data, pre_impute, pre_imputation, model_imputation,
                 accept_reject,reweight, RN_model, RN_train_with_cal,
                  generalized_impute_train, generalize_pre_impute_iteration,
                 basemodel,alpha, cores, exact, other_model_parameters=None, ocacle_pre_impute=False,
                 weighted_search_range=0.5, weighted_grid_number=30):
        """
        Experiment runner class for conformal prediction under missingness.
        Supports various conformal strategies (Exact/Nested/Weighted/ARC-CP), and handles
        data imputation, risk estimation, and predictive interval calibration.
        :param data: Instance of a missing-data-aware data loader (e.g., Synthetic_linear_data).
        :param pre_impute: Whether to apply pre-imputation before modeling.
        :param pre_imputation: Pre-imputation method name (e.g., 'MICE', 'mean').
        :param model_imputation: Method used for test-time imputation.
        :param accept_reject: Whether to use accept-reject based CP (ARC-CP).
        :param reweight: Whether to apply reweighting in calibration (Weighted CP).
        :param RN_model: Model used to estimate missingness probabilities (e.g., 'xgboost').
        :param RN_train_with_cal: If True, include calibration set when training RN_model.
        :param generalized_impute_train: If True, use (X, Y) as input to the imputer during training.
        :param generalize_pre_impute_iteration: Number of iterations for pre-imputation on calibration data.
        :param basemodel: Base regression model (e.g., 'RF', 'Linear', 'XGBoost').
        :param alpha: Miscoverage level for conformal prediction (e.g., 0.1 for 90% intervals).
        :param cores: Number of cores to use for base model fitting.
        :param exact: If True, use Exact CP-MDA; otherwise Nested.
        :param other_model_parameters: Additional hyperparameters for base model.
        :param ocacle_pre_impute: If True, uses oracle (ground-truth) data for imputation (ablation).
        """
        self.data = data
        self.dataset_name = data.name
        self.pre_impute = pre_impute
        self.n_rep = self.data.n_rep
        self.is_tqdm = False
        self.pre_impute = pre_impute
        self.pre_imputation = pre_imputation
        self.model_imputation = model_imputation
        self.generalized_pre_impute = pre_impute
        self.generalized_impute_train = generalized_impute_train
        self.generalize_pre_impute_iteration = generalize_pre_impute_iteration
        self.accept_reject = accept_reject
        self.reweight = reweight
        self.RN_model = RN_model
        self.RN_train_with_cal = RN_train_with_cal
        self.basemodel = basemodel
        self.alpha = alpha
        self.params_basemodel = {'cores': cores}
        if other_model_parameters is not None:
            self.params_basemodel.update(other_model_parameters)
        self.exact = exact
        self.ocacle_pre_impute = ocacle_pre_impute
        self.weighted_search_range = weighted_search_range
        self.weighted_grid_number = weighted_grid_number

    def run_experiments(self):
        """
        Run the full pipeline for one experimental configuration over multiple replications.
        For each replication:
            - Apply pre-imputation (if specified).
            - Fit a model for missingness (density ratio prediction model).
            - Fit a base regression model on imputed data.
            - Calibrate conformal intervals using exact/nested CP or weighted/accept-reject CP.
            - Evaluate performance on test sets with different missing patterns.

        Results (per-replication prediction intervals) are aggregated and stored in `self.results`.

        :return: Dictionary containing aggregated prediction intervals over all replications,
                 with keys corresponding to test dataset variants (e.g., 'iid', 'fixed_nb_sample_pattern').
        """
        results = None
        self.init_hyper()

        for k in tqdm(range(self.n_rep), disable=not self.is_tqdm):
            self.k = k
            if not self.is_tqdm:
                print('{} rep over {}'.format(k,self.n_rep))

            X_mis_train = self.data['X_missing']['Train'][k, :, :]
            X_mis_cal = self.data['X_missing']['Cal'][k, :, :]

            M_train = self.data['M']['Train'][k, :, :]
            M_train_unchanged = M_train.copy()
            M_cal = self.data['M']['Cal'][k, :, :]
            M_cal_unchanged = M_cal.copy()

            Y_train = self.data['Y']['Train'][k, :]
            Y_cal = self.data['Y']['Cal'][k, :]

            keys_test = list(self.data['X_missing']['Test'].keys())
            X_mis_test = dict.fromkeys(keys_test)
            M_test = dict.fromkeys(keys_test)
            Y_test = dict.fromkeys(keys_test)

            for key in keys_test:
                X_mis_test[key] = self.data['X_missing']['Test'][key][k, :, :]
                M_test[key] = self.data['M']['Test'][key][k, :, :]
                Y_test[key] = self.data['Y']['Test'][key][k, :]

            target = 'Quantiles'

            if self.pre_impute and self.pre_imputation != 'oracle':
                pre_imputer = utils.get_impute(self.pre_imputation)
            else:
                pre_imputer = utils.get_impute('None')

            model_imputer = utils.get_impute(self.model_imputation)

            model_imputer.fit(X_mis_train)


            if self.pre_impute and self.pre_imputation == 'oracle':
                print('oracle imputation')
                class Oracle_pre_imputer():
                    def __init__(self, parent, k):
                        self.parent = parent
                        self.k = k
                        print(self.parent.data['X']['Train'].shape)
                        assert self.parent.data['X']['Train'].shape[1] != self.parent.data['X']['Cal'].shape[1]

                    def fit(self,X):
                        pass

                    def transform(self,X):
                        if X.shape[0] == self.parent.data['X']['Train'].shape[1]:
                            return self.parent.data['X']['Train'][self.k]
                        elif X.shape[0] == self.parent.data['X']['Cal'].shape[1]:
                            return self.parent.data['X']['Cal'][self.k]
                        else:
                            raise NotImplementedError
                pre_imputer = Oracle_pre_imputer(self, self.k)

            else:
                if self.generalized_pre_impute:
                    pre_imputer.fit(np.column_stack((X_mis_train, Y_train)))
                else:
                    pre_imputer.fit(X_mis_train)

            X_imp_train = model_imputer.transform(X_mis_train)
            if self.generalized_pre_impute and self.generalized_impute_train:
                X_imp_train_with_Y = pre_imputer.transform(np.column_stack((X_mis_train,Y_train)))[:, :-1]

            X_imp_test = dict.fromkeys(keys_test)
            if self.pre_impute:
                if self.generalized_pre_impute:
                    if self.generalize_pre_impute_iteration > 0:
                        x_cal_tmp = []
                        m_cal_tmp = []
                        y_cal_tmp = []
                        for i in range(self.generalize_pre_impute_iteration):
                            X_mis_cal = pre_imputer.transform(np.column_stack((X_mis_cal, Y_cal)))[:, :-1]
                            M_cal = np.zeros_like(M_cal)
                            x_cal_tmp.append(X_mis_cal)
                            m_cal_tmp.append(M_cal)
                            y_cal_tmp.append(Y_cal)
                        X_mis_cal = np.concatenate(x_cal_tmp, axis=0)
                        M_cal = np.concatenate(m_cal_tmp, axis=0)
                        Y_cal = np.concatenate(y_cal_tmp, axis=0)
                    else:
                        X_mis_cal = pre_imputer.transform(np.column_stack((X_mis_cal, Y_cal)))[:, :-1]
                        M_cal = np.zeros_like(M_cal)

                else:
                    X_mis_cal = pre_imputer.transform(X_mis_cal)
                    M_cal = np.zeros_like(M_cal)

            self.model_RN_derivative = None
            assert not (self.accept_reject and self.reweight), f"Invalid state: bool_var1={self.accept_reject}, bool_var2={self.reweight}"

            self.mask_class_model = None
            if self.accept_reject or self.reweight:
                if self.generalized_pre_impute:
                    train_imputed_train_C = pre_imputer.transform(
                        np.column_stack((X_mis_train, Y_train)))[:, :-1]
                else:
                    train_imputed_train_C = pre_imputer.transform(X_mis_train)

                train_concat = np.column_stack([train_imputed_train_C, Y_train])
                cal_concat = np.column_stack([X_mis_cal, Y_cal])

                if self.RN_train_with_cal:
                    train_concat = np.vstack(( train_concat,cal_concat))
                    M_train_concat = np.vstack((M_train, M_cal_unchanged))
                else:
                    M_train_concat = M_train.copy()

                self.mask_class_data = Data_for_mask(
                    train_concat,
                    M_train_concat,
                    batch_size=128,
                    contain_y=True,
                    dataset=self.dataset_name)

                assert self.RN_model.lower() in ['histgbdt', 'xgboost', 'logistic'], '{} not correct'.format(self.RN_model)
                self.mask_class_model = Miss_proba_simple_model(self.mask_class_data, model=self.RN_model, epoch=1000)
                self.mask_class_model.train()

            for key in keys_test:
                X_imp_test[key] = model_imputer.transform(X_mis_test[key])
                # this is to transform X_mis_test to a complete one, using original imputation

            if self.generalized_pre_impute and self.generalized_impute_train:
                features_train = X_imp_train_with_Y
            else:
                features_train = X_imp_train
            features_test = X_imp_test

            trained_model = fit_basemodel(features_train, Y_train, target=target,
                                          basemodel=self.basemodel, alpha=self.alpha,
                                          params_basemodel=self.params_basemodel)

            groups_test = dict.fromkeys(keys_test)
            for key in keys_test:
                groups_test[key] = list(map(utils.pattern_to_id, M_test[key].astype(int)))
            preds = dict.fromkeys(keys_test)

            for key in keys_test:
                pred = self.calibrate_predict(fitted_basemodel=trained_model,
                                            model_imputer=model_imputer,
                                            X_cal=X_mis_cal,
                                            M_cal=M_cal,
                                            Y_cal=Y_cal,
                                            X_mis_test=X_mis_test[key],
                                            features_test=features_test[key],
                                            M_test=M_test[key],
                                            groups_test=groups_test[key],
                                            target=target,
                                            M_cal_unchanged=M_cal_unchanged
                                           )
                preds[key] = pred

            if results is None:
                results = dict.fromkeys(keys_test)
                for key in keys_test:
                    results[key] = {'Y_inf': np.array(preds[key]['y_inf']),
                                    'Y_sup': np.array(preds[key]['y_sup'])}
            else:
                for key in keys_test:
                    results[key]['Y_inf'] = np.vstack(
                        (results[key]['Y_inf'], np.array(preds[key]['y_inf'])))
                    results[key]['Y_sup'] = np.vstack(
                        (results[key]['Y_sup'], np.array(preds[key]['y_sup'])))

        self.results = results
        return results

    def run_simple_synthetic(self):
        """
        Run a simplified experimental pipeline tailored for synthetic datasets with known generative processes.

        This method differs from `run_experiments` in that:
        - The pre-imputation is performed via a simple perturbation-based method.
        - Oracle likelihood ratio estimation can be optionally used.
        - Internal validation includes correlation between true and estimated likelihood ratios.

        The overall steps mirror `run_experiments`, but assume the presence of oracle knowledge.

        :return: Dictionary of predictive intervals across replications for each test scenario.
        """
        results = None

        for k in tqdm(range(self.n_rep), disable=not self.is_tqdm):
            self.k = k
            if not self.is_tqdm:
                print('{} rep over {}'.format(k,self.n_rep))

            X_mis_train = self.data['X_missing']['Train'][k, :, :]
            X_mis_cal = self.data['X_missing']['Cal'][k, :, :]

            M_train = self.data['M']['Train'][k, :, :]
            M_train_unchanged = M_train.copy()
            M_cal = self.data['M']['Cal'][k, :, :]
            M_cal_unchanged = M_cal.copy()

            Y_train = self.data['Y']['Train'][k, :]
            Y_cal = self.data['Y']['Cal'][k, :]

            keys_test = list(self.data['X_missing']['Test'].keys())
            X_mis_test = dict.fromkeys(keys_test)
            M_test = dict.fromkeys(keys_test)
            Y_test = dict.fromkeys(keys_test)

            for key in keys_test:
                X_mis_test[key] = self.data['X_missing']['Test'][key][k, :, :]
                M_test[key] = self.data['M']['Test'][key][k, :, :]
                Y_test[key] = self.data['Y']['Test'][key][k, :]

            target = 'Quantiles'

            if self.pre_impute:

                class Simple_preimputer:
                    def __init__(self, loader):
                        self.loader = loader

                    def transform(self,X):
                        Y = X[:, -1]
                        X = X[:, :-1]
                        return np.column_stack([self.loader.impute_with_perturbation(X,Y),Y])

                    def fit(self,X):
                        pass

                pre_imputer = Simple_preimputer(self.data)

            model_imputer = utils.get_impute(self.model_imputation)

            model_imputer.fit(X_mis_train)


            X_imp_train = model_imputer.transform(X_mis_train)
            if self.generalized_pre_impute and self.generalized_impute_train:
                X_imp_train_with_Y = pre_imputer.transform(np.column_stack((X_mis_train,Y_train)))[:, :-1]

            X_imp_test = dict.fromkeys(keys_test)
            X_mis_cal = pre_imputer.transform(np.column_stack((X_mis_cal, Y_cal)))[:, :-1]
            M_cal = np.zeros_like(M_cal)

            self.model_RN_derivative = None
            assert not (self.accept_reject and self.reweight), f"Invalid state: bool_var1={self.accept_reject}, bool_var2={self.reweight}"

            self.mask_class_model = None
            if self.accept_reject or self.reweight:
                if self.generalized_pre_impute:
                    train_imputed_train_C = pre_imputer.transform(
                        np.column_stack((X_mis_train, Y_train)))[:, :-1]
                else:
                    train_imputed_train_C = pre_imputer.transform(X_mis_train)

                train_concat = np.column_stack([train_imputed_train_C, Y_train])
                cal_concat = np.column_stack([X_mis_cal, Y_cal])

                train_concat = np.vstack(( train_concat,cal_concat))
                M_train_concat = np.vstack((M_train, M_cal_unchanged))

                self.mask_class_data = Data_for_mask_lr(
                    train_concat,
                    M_train_concat,
                    batch_size=128,
                    contain_y=True,
                    dataset=self.dataset_name)

                assert self.RN_model.lower() in ['histgbdt', 'xgboost', 'logistic', 'oracle'], '{} not correct'.format(self.RN_model)

                if self.RN_model.lower() == 'oracle':
                    class oracle_RN:
                        def __init__(self, dataloader):
                            self.dataloader = dataloader

                        def eval(self, x):
                            Y = x[:, -1]
                            X = x[:, :-1]
                            M = np.isnan(X).astype(int)
                            lrs = []
                            for i in range(X.shape[0]):
                                obs_index = [i for i, val in enumerate(M[i]) if val == 0]
                                lrs.append(self.dataloader.likelihood_ratio_estimation(X[i][obs_index], Y[i], obs_index))
                            return np.array(lrs)
                    self.mask_class_model = oracle_RN(self.data)
                else:
                    self.mask_class_model = Miss_proba_simple_model(self.mask_class_data, model=self.RN_model, epoch=1000)
                    self.mask_class_model.train()

                if self.RN_model.lower() != 'oracle':
                    M_lr = [1, 1, 0]
                    M_index = [i for i, val in enumerate(M_lr) if val == 1]
                    obs_index = [i for i, val in enumerate(M_lr) if val == 0]
                    n_test = len(X_mis_cal)
                    lr_esti_vals = []
                    lr_true_vals = []

                    for i in range(n_test):
                        x_missing = X_mis_cal[i].copy()
                        x_missing[M_index] = np.nan

                        x = X_mis_cal[:, obs_index][i]
                        y = Y_cal[i]

                        lr_estimation = self.mask_class_model.eval(np.append(x_missing, y).reshape(1, -1))
                        lr_true = self.data.likelihood_ratio_estimation(x, y, obs_index)

                        lr_esti_vals.append(lr_estimation[0])
                        lr_true_vals.append(lr_true)
                    print("correlation", np.corrcoef(lr_true_vals, lr_esti_vals)[0, 1])

            for key in keys_test:
                X_imp_test[key] = model_imputer.transform(X_mis_test[key])
                # this is to transform X_mis_test to a complete one, using original imputation

            if self.generalized_pre_impute and self.generalized_impute_train:
                features_train = X_imp_train_with_Y
            else:
                features_train = X_imp_train
            features_test = X_imp_test

            trained_model = fit_basemodel(features_train, Y_train, target=target,
                                          basemodel=self.basemodel, alpha=self.alpha,
                                          params_basemodel=self.params_basemodel)

            groups_test = dict.fromkeys(keys_test)
            for key in keys_test:
                groups_test[key] = list(map(utils.pattern_to_id, M_test[key].astype(int)))
            preds = dict.fromkeys(keys_test)

            for key in keys_test:
                pred = self.calibrate_predict(fitted_basemodel=trained_model,
                                            model_imputer=model_imputer,
                                            X_cal=X_mis_cal,
                                            M_cal=M_cal,
                                            Y_cal=Y_cal,
                                            X_mis_test=X_mis_test[key],
                                            features_test=features_test[key],
                                            M_test=M_test[key],
                                            groups_test=groups_test[key],
                                            target=target,
                                            M_cal_unchanged=M_cal_unchanged
                                           )

                preds[key] = pred

            if results is None:
                results = dict.fromkeys(keys_test)
                for key in keys_test:
                    results[key] = {'Y_inf': np.array(preds[key]['y_inf']),
                                    'Y_sup': np.array(preds[key]['y_sup'])}
            else:
                for key in keys_test:
                    results[key]['Y_inf'] = np.vstack(
                        (results[key]['Y_inf'], np.array(preds[key]['y_inf'])))
                    results[key]['Y_sup'] = np.vstack(
                        (results[key]['Y_sup'], np.array(preds[key]['y_sup'])))

        self.results = results
        return results

    def save(self, file_name):
        """
        Save the current state of the experiment instance to disk using pickle.
        :param file_name: Name (without extension) to save the experiment object.
        """
        file = open('./data/' + file_name + '.pkl', 'wb')
        pickle.dump(self.__dict__, file)
        file.close()

    def load(self, file_name):
        """try load self.name.txt"""
        file = open('./data/' + file_name + '.pkl', 'rb')
        self.__dict__ = pickle.load(file)
        file.close()

    def init_hyper(self):
        if 'meps19' in self.dataset_name:
            self.weighted_search_range = 200
            self.weighted_grid_number = 200

    def calculate_results_pattern(self):

        key = -1

        data, results = self.data, self.results
        contains, lengths = utils.compute_PI_metrics(data, results, 'iid')
        metrics = utils.compute_metrics_cond(self.n_rep, data, results,
                                             'fixed_nb_sample_pattern',
                                             cond='Pattern')

        dict_cov = {}
        dict_len = {}

        dict_cov[key] = np.mean(contains, axis=1)
        dict_len[key] = np.mean(lengths, axis=1)

        avg_cov = {}
        for key_pattern in metrics.keys():
            dict_cov[key_pattern] = metrics[key_pattern]['avg_cov']
            dict_len[key_pattern] = metrics[key_pattern]['avg_len']
            avg_cov[key_pattern] = np.mean(metrics[key_pattern]['avg_cov'])
            print(key_pattern,
                  avg_cov[key_pattern], np.std(metrics[key_pattern]['avg_cov'],ddof=1),
                  np.mean(metrics[key_pattern]['avg_len']), np.std(metrics[key_pattern]['avg_len'],ddof=1))

        worst_index = np.argmin(list(avg_cov.values()))
        worst_key = list(avg_cov.keys())[worst_index]
        worst_coverage = dict_cov[worst_key]
        worst_len = dict_len[worst_key]

        best_index = np.argmax(list(avg_cov.values()))
        best_key = list(avg_cov.keys())[best_index]
        best_coverage = dict_cov[best_key]
        best_len = dict_len[best_key]

        table_cln = ['marginal_cov', 'marginal_cov_len']
        table_value = [dict_cov[-1], dict_len[-1]]

        for i, j in enumerate(metrics.keys()):
            table_cln.extend(['mask_cond_cov_num_mask_{}'.format(j),
                              'mask_cond_cov_len_num_mask_{}'.format(j), ])

            table_value.extend([dict_cov[j], dict_len[j], ])

        tbl = pd.DataFrame(np.array(table_value).T, columns=table_cln)

        results_dict = {}
        results_dict['marg_coverage'] = np.mean(dict_cov[-1])
        results_dict['marg_coverage_std'] = np.std(dict_cov[-1], ddof=1)

        results_dict['marg_cov_len'] = np.mean(dict_len[-1])
        results_dict['marg_cov_len_std'] = np.std(dict_len[-1], ddof=1)

        results_dict['mask_cov_best'] = np.mean(best_coverage)
        results_dict['mask_cov_best_std'] = np.std(best_coverage, ddof=1)

        results_dict['mask_cov_best_len'] = np.mean(best_len)
        results_dict['mask_cov_best_len_std'] = np.std(best_len, ddof=1)

        results_dict['mask_cov_worst'] = np.mean(worst_coverage)
        results_dict['mask_cov_worst_std'] = np.std(worst_coverage, ddof=1)

        results_dict['mask_cov_worst_len'] = np.mean(worst_len)
        results_dict['mask_cov_worst_len_std'] = np.std(worst_len, ddof=1)

        results_dict['results'] = tbl
        print(results_dict)
        return tbl, results_dict

    def calibrate_predict(self, fitted_basemodel, model_imputer,
                          X_cal, M_cal, Y_cal,
                          X_mis_test, features_test, M_test,
                          groups_test, target='Quantiles',
                          M_cal_unchanged=None,
                          reweight_with_y=False):
        """
        Calibrate predictive intervals on the test data using conformal prediction under missingness.
        - Exact CP (pattern-matched calibration sets)
        - Nested CP (cross-pattern calibration)
        - Weighted CP using estimated likelihood ratios
        - Accept-Reject CP based on reweighting scores
        :param fitted_basemodel: Trained base regression model (supports quantile prediction).
        :param model_imputer: Imputer fitted on training data.
        :param X_cal: Calibration features (with missing values).
        :param M_cal: Missing masks for calibration data.
        :param Y_cal: Calibration labels.
        :param X_mis_test: Test data with missing values.
        :param features_test: Test data after imputation (used for prediction).
        :param M_test: Mask matrix for test data.
        :param groups_test: List of test sample pattern IDs (via utils.pattern_to_id).
        :param target: Must be 'Quantiles' (CP currently assumes quantile models).
        :param M_cal_unchanged: Unmodified calibration masks (for ARC-CP).
        :param reweight_with_y: If True, fallback to symmetric conformal intervals even under reweighting.
        :return: A dictionary with keys:
                 - 'y_inf': Lower bounds of prediction intervals (size: n_test)
                 - 'y_sup': Upper bounds of prediction intervals (size: n_test)
        """
        exact = self.exact
        alpha = self.alpha
        mechanism = self.data['mechanism']
        accept_reject = self.accept_reject
        reweight = self.reweight
        basemodel = self.basemodel
        mask_class_model = self.mask_class_model
        # X_cal: calibration with missing value( if impute first then imputed cal data)

        assert target in ['Quantiles']
        assert basemodel in ['Linear', 'RF', 'NNet', 'XGBoost', 'histgbdt', 'NNet_missing']

        impute_and_mask_conditional = False
        if impute_and_mask_conditional:
            M_test = np.zeros_like(M_test)
            M_cal = np.zeros_like(M_cal)
            X_cal = model_imputer.transform(X_cal.copy())
            X_test = model_imputer.transform(X_mis_test.copy())
            print(X_test[0], features_test[0])
            assert np.array_equal(X_test, features_test), "Arrays are not equal"

        patterns = np.unique(M_test, axis=0)
        ids = list(map(utils.pattern_to_id, patterns.astype(int)))

        n_test = features_test.shape[0]
        q_scores_test = np.empty(n_test)

        if not exact:
            pred_test = {'y_inf': np.empty(n_test),
                         'y_sup': np.empty(n_test)}

        if reweight:
            pred_test_reweight = {'y_inf': np.ones(n_test) * np.inf,
                                  'y_sup': np.ones(n_test) * -np.inf}

        for idp, id_pattern in enumerate(ids):  # iteration over test missingness patterns
            test_index = (np.array(groups_test) == id_pattern).flatten()
            test_index_list = np.where(test_index)[0]

            X_imp_cal_masking = copy.deepcopy(X_cal)
            M_cal_masking = copy.deepcopy(M_cal)
            Y_cal_masking = copy.deepcopy(Y_cal)

            pattern = patterns[idp]
            empty = False

            if exact:
                # Only keep cal rows where observed features match those in the test pattern
                ind_subsample = np.all(M_cal[:, pattern == 0] == 0, axis=1)
                if np.sum(ind_subsample) == 0:
                    empty = True
                X_imp_cal_masking = X_imp_cal_masking[ind_subsample, :]
                M_cal_masking = M_cal_masking[ind_subsample, :]
                Y_cal_masking = Y_cal_masking[ind_subsample]

            if X_imp_cal_masking.shape[1] > len(pattern):
                nb = X_imp_cal_masking.shape[1] - len(pattern)
                pattern_ext = np.append(pattern, np.full(nb, 0))
            else:
                pattern_ext = pattern

            # Apply the same missingness pattern to calibration data
            X_imp_cal_masking[:, pattern_ext == 1] = np.nan
            M_cal_masking[:, pattern == 1] = 1

            if reweight:
                shift_scores = mask_class_model.eval(np.column_stack([X_imp_cal_masking, Y_cal]))
                # Apply the same pattern to test features to construct X_ampute_test
                X_ampute_test = features_test.copy()
                X_ampute_test[M_test == 1] = np.nan
                X_ampute_test = X_ampute_test[test_index]

            if accept_reject:
                shift_scores = mask_class_model.normalize_eval(np.column_stack([X_imp_cal_masking, Y_cal]))
                random_uniform = np.random.uniform(0, 1, size=len(shift_scores))
                if mechanism == 'MCAR':
                    include_mask = utils.check_condition(M_cal_unchanged, pattern_ext)
                else:
                    include_mask = np.zeros_like(random_uniform, dtype=bool)
                accepted_indices = np.where((random_uniform < shift_scores) | include_mask)[0]
                X_imp_cal_masking = X_imp_cal_masking[accepted_indices]
                M_cal_masking = M_cal_masking[accepted_indices]
                Y_cal_masking = Y_cal_masking[accepted_indices]

            if not empty:
                X_imp_cal_masking = model_imputer.transform(X_imp_cal_masking)
                features_cal = X_imp_cal_masking
                cal_predictions = predict_basemodel(fitted_basemodel, features_cal, target, basemodel, alpha)
                # Calibration score = max of (prediction error above/below interval), CQR
                scores = np.maximum(cal_predictions['y_inf'] - Y_cal_masking, Y_cal_masking - cal_predictions['y_sup'])

            if exact == True:

                if not empty:

                    q_scores_cal = utils.quantile_corrected(scores, alpha=alpha)

                    if reweight:
                        X_to_pred = copy.deepcopy(X_mis_test[(np.array(groups_test) == id_pattern).flatten(), :])
                        X_to_pred_masking = copy.deepcopy(X_to_pred)
                        X_to_pred_masking = model_imputer.transform(X_to_pred_masking)

                        features_test_pattern = X_to_pred_masking

                        preds_this_test = predict_basemodel(fitted_basemodel, features_test_pattern, target, basemodel,
                                                            alpha)
                        min_score = min(scores) - self.weighted_search_range
                        max_score = max(scores) + self.weighted_search_range
                        grid_number = self.weighted_grid_number
                        each_grid = (-min_score + max_score) / grid_number
                        # print('each_grid', each_grid)

                        for grid in np.linspace(-max_score, -min_score, grid_number):
                            # Evaluate LR on test sample + y_shift
                            # Compute weighted quantile across calibration scores
                            # Update prediction bounds if the test score is below threshold
                            y_inf = preds_this_test['y_inf'] + grid
                            shift_scores_y = mask_class_model.eval(np.column_stack([X_ampute_test, y_inf.reshape(-1, 1)]))
                            test_len = shift_scores_y.shape[0]
                            array_n = np.asarray(shift_scores).reshape(1, -1)
                            array_m = np.asarray(shift_scores_y).reshape(-1, 1)

                            combined_y = np.hstack([array_n.repeat(len(array_m), axis=0), array_m])
                            weights_y = combined_y / combined_y.sum(axis=1, keepdims=True)

                            expanded_array_y = np.tile(scores, (test_len, 1))  # Repeat array m times
                            expanded_array_y = np.hstack(
                                (expanded_array_y, np.full((test_len, 1), np.nan)))  # Add a column of NaN

                            quantiles_y = utils.weighted_quantile_matrix(expanded_array_y, weights_y, q=1-alpha)

                            score_y = np.maximum(preds_this_test['y_inf'] - y_inf, y_inf - preds_this_test['y_sup'])
                            indices = np.where(score_y <= quantiles_y)[0]
                            pred_test_reweight['y_inf'][test_index_list[indices]] = np.minimum(pred_test_reweight['y_inf'][test_index_list[indices]], y_inf[indices])

                            y_sup = preds_this_test['y_sup'] - grid
                            shift_scores_y = mask_class_model.eval(np.column_stack([X_ampute_test,
                                                                                    y_sup.reshape(-1, 1)
                                                                                    ]))
                            test_len = shift_scores_y.shape[0]
                            array_n = np.asarray(shift_scores).reshape(1, -1)
                            array_m = np.asarray(shift_scores_y).reshape(-1, 1)

                            combined_y = np.hstack([array_n.repeat(len(array_m), axis=0), array_m])
                            weights_y = combined_y / combined_y.sum(axis=1, keepdims=True)

                            expanded_array_y = np.tile(scores, (test_len, 1))  # Repeat array m times
                            expanded_array_y = np.hstack(
                                (expanded_array_y, np.full((test_len, 1), np.nan)))  # Add a column of NaN

                            quantiles_y = utils.weighted_quantile_matrix(expanded_array_y, weights_y, q=1 - alpha)
                            score_y = np.maximum(preds_this_test['y_inf'] - y_sup, y_sup - preds_this_test['y_sup'])
                            indices = np.where(score_y <= quantiles_y)[0]
                            pred_test_reweight['y_sup'][test_index_list[indices]] = np.maximum(pred_test_reweight['y_sup'][test_index_list[indices]], y_sup[indices])

                        index_inf_to_fill = np.where(pred_test_reweight['y_inf'][test_index_list] == np.inf)[0]
                        pred_test_reweight['y_inf'][test_index_list[index_inf_to_fill]] = y_inf[index_inf_to_fill]

                        index_sup_to_fill = np.where(pred_test_reweight['y_sup'][test_index_list] == -np.inf)[0]
                        pred_test_reweight['y_sup'][test_index_list[index_sup_to_fill]] = y_sup[index_sup_to_fill]

                        pred_test_reweight['y_sup'][test_index_list] = pred_test_reweight['y_sup'][test_index_list] + each_grid
                        pred_test_reweight['y_inf'][test_index_list] = pred_test_reweight['y_inf'][test_index_list] - each_grid

                    else:
                        q_scores_test[(np.array(groups_test) == id_pattern).flatten()] = q_scores_cal

                else:
                    q_scores_test[(np.array(groups_test) == id_pattern).flatten()] = np.inf

            else:
                assert not exact
                X_to_pred = copy.deepcopy(X_mis_test[(np.array(groups_test) == id_pattern).flatten(), :])
                M_to_pred = copy.deepcopy(M_test[(np.array(groups_test) == id_pattern).flatten(), :])
                n_current = X_to_pred.shape[0]

                patterns_cal = np.unique(M_cal_masking, axis=0)
                ids_cal = list(map(utils.pattern_to_id, patterns_cal.astype(int)))

                groups_cal_masking = list(map(utils.pattern_to_id, M_cal_masking.astype(int)))
                nb_cal = len(scores)

                all_preds = {'y_inf': np.empty((n_current, nb_cal)),
                             'y_sup': np.empty((n_current, nb_cal))}

                for idp_cal, id_pattern_cal in enumerate(ids_cal):
                    # For each cal pattern, apply its pattern mask to test set and compute score
                    # Then aggregate quantiles across different masks
                    idx_cal_masking = (np.array(groups_cal_masking) == id_pattern_cal).flatten()
                    nb_mask = np.sum(idx_cal_masking)
                    all_preds['y_inf'][:, idx_cal_masking] = np.tile(scores[idx_cal_masking], (n_current, 1))
                    all_preds['y_sup'][:, idx_cal_masking] = np.tile(scores[idx_cal_masking], (n_current, 1))

                    pattern_masking = patterns_cal[idp_cal]

                    X_to_pred_masking = copy.deepcopy(X_to_pred)
                    M_to_pred_masking = copy.deepcopy(M_to_pred)
                    X_to_pred_masking[:, pattern_masking == 1] = np.nan
                    M_to_pred_masking[:, pattern_masking == 1] = 1

                    X_to_pred_masking = model_imputer.transform(X_to_pred_masking)
                    features_test_pattern = X_to_pred_masking

                    preds_k = predict_basemodel(fitted_basemodel, features_test_pattern, target, basemodel, alpha)
                    all_preds['y_inf'][:, idx_cal_masking] = -np.tile(preds_k['y_inf'], (nb_mask, 1)).T + all_preds['y_inf'][:,idx_cal_masking]
                    all_preds['y_sup'][:, idx_cal_masking] = np.tile(preds_k['y_sup'], (nb_mask, 1)).T + all_preds['y_sup'][:,idx_cal_masking]
                if (1 - alpha) * (1 + 1 / nb_cal) > 1:
                    pred_test['y_inf'][(np.array(groups_test) == id_pattern).flatten()] = [-np.inf] * n_current
                    pred_test['y_sup'][(np.array(groups_test) == id_pattern).flatten()] = [np.inf] * n_current
                else:
                    pred_test['y_inf'][(np.array(groups_test) == id_pattern).flatten()] = -np.quantile(all_preds['y_inf'], (1 - alpha) * (1 + 1 / nb_cal), axis=1)
                    pred_test['y_sup'][(np.array(groups_test) == id_pattern).flatten()] = np.quantile(all_preds['y_sup'], (1 - alpha) * (1 + 1 / nb_cal), axis=1)

        if exact == True:
            if reweight:
                if reweight_with_y:
                    pred_test = predict_basemodel(fitted_basemodel, features_test, target, basemodel, alpha)
                    interval_predictions = {'y_inf': pred_test['y_inf'] - q_scores_test,
                                            'y_sup': pred_test['y_sup'] + q_scores_test}
                else:
                    interval_predictions = {'y_inf': pred_test_reweight['y_inf'],
                                            'y_sup': pred_test_reweight['y_sup']}
                    # print(interval_predictions)

            else:
                pred_test = predict_basemodel(fitted_basemodel, features_test, target, basemodel, alpha)
                interval_predictions = {'y_inf': pred_test['y_inf'] - q_scores_test,
                                        'y_sup': pred_test['y_sup'] + q_scores_test}
        else:
            interval_predictions = {'y_inf': pred_test['y_inf'],
                                    'y_sup': pred_test['y_sup']}

        return interval_predictions

class Missing_experiments_MCAR():

    def __init__(self, data, pre_impute, pre_imputation, model_imputation,
                 accept_reject,reweight, RN_model, RN_train_with_cal,
                  generalized_impute_train, generalize_pre_impute_iteration,
                 basemodel,alpha, cores, exact, other_model_parameters=None, ocacle_pre_impute=False):
        """
        The same implementation of experiment class for MCAR
        """
        self.data = data
        self.dataset_name = data.name
        self.n_rep = self.data.n_rep
        self.is_tqdm = False
        self.pre_impute = pre_impute
        self.pre_imputation = pre_imputation
        self.model_imputation = model_imputation
        self.generalized_pre_impute = pre_impute
        self.generalized_impute_train = generalized_impute_train
        self.generalize_pre_impute_iteration = generalize_pre_impute_iteration
        self.accept_reject = accept_reject
        self.reweight = reweight
        self.RN_model = RN_model
        self.RN_train_with_cal = RN_train_with_cal
        self.basemodel = basemodel
        self.alpha = alpha
        self.params_basemodel = {'cores': cores}
        if other_model_parameters is not None:
            self.params_basemodel.update(other_model_parameters)
        self.exact = exact
        self.ocacle_pre_impute = ocacle_pre_impute

    def run_experiments(self):
        results = None

        for k in tqdm(range(self.n_rep), disable=not self.is_tqdm):
            self.k = k
            if not self.is_tqdm:
                print('{} rep over {}'.format(k,self.n_rep))

            X_mis_train = self.data['X_missing']['Train'][k, :, :]
            X_mis_cal = self.data['X_missing']['Cal'][k, :, :]

            M_train = self.data['M']['Train'][k, :, :]
            M_cal = self.data['M']['Cal'][k, :, :]
            M_cal_unchanged = M_cal.copy()

            Y_train = self.data['Y']['Train'][k, :]
            Y_cal = self.data['Y']['Cal'][k, :]

            keys_test = list(self.data['X_missing']['Test'].keys())
            X_mis_test = dict.fromkeys(keys_test)
            M_test = dict.fromkeys(keys_test)
            Y_test = dict.fromkeys(keys_test)

            for key in keys_test:
                X_mis_test[key] = self.data['X_missing']['Test'][key][k, :, :]
                M_test[key] = self.data['M']['Test'][key][k, :, :]
                Y_test[key] = self.data['Y']['Test'][key][k, :]

            target = 'Quantiles'

            if self.pre_impute and self.pre_imputation != 'oracle':
                pre_imputer = utils.get_impute(self.pre_imputation)
            else:
                pre_imputer = utils.get_impute('None')

            model_imputer = utils.get_impute(self.model_imputation)

            model_imputer.fit(X_mis_train)


            if self.pre_impute and self.pre_imputation == 'oracle':
                print('oracle imputation')
                class Oracle_pre_imputer():
                    def __init__(self, parent, k):
                        self.parent = parent
                        self.k = k
                        print(self.parent.data['X']['Train'].shape)
                        assert self.parent.data['X']['Train'].shape[1] != self.parent.data['X']['Cal'].shape[1]

                    def fit(self,X):
                        pass

                    def transform(self,X):
                        if X.shape[0] == self.parent.data['X']['Train'].shape[1]:
                            return self.parent.data['X']['Train'][self.k]
                        elif X.shape[0] == self.parent.data['X']['Cal'].shape[1]:
                            return self.parent.data['X']['Cal'][self.k]
                        else:
                            raise NotImplementedError
                pre_imputer = Oracle_pre_imputer(self, self.k)

            else:
                if self.generalized_pre_impute:
                    pre_imputer.fit(np.column_stack((X_mis_train, Y_train)))
                else:
                    pre_imputer.fit(X_mis_train)

            X_imp_train = model_imputer.transform(X_mis_train)
            if self.generalized_pre_impute and self.generalized_impute_train:
                X_imp_train_with_Y = pre_imputer.transform(np.column_stack((X_mis_train,Y_train)))[:, :-1]

            X_imp_test = dict.fromkeys(keys_test)
            if self.pre_impute:
                if self.generalized_pre_impute:
                    if self.generalize_pre_impute_iteration > 0:
                        x_cal_tmp = []
                        m_cal_tmp = []
                        y_cal_tmp = []
                        for i in range(self.generalize_pre_impute_iteration):
                            X_mis_cal = pre_imputer.transform(np.column_stack((X_mis_cal, Y_cal)))[:, :-1]
                            M_cal = np.zeros_like(M_cal)
                            x_cal_tmp.append(X_mis_cal)
                            m_cal_tmp.append(M_cal)
                            y_cal_tmp.append(Y_cal)
                        X_mis_cal = np.concatenate(x_cal_tmp, axis=0)
                        M_cal = np.concatenate(m_cal_tmp, axis=0)
                        Y_cal = np.concatenate(y_cal_tmp, axis=0)
                    else:
                        X_mis_cal = pre_imputer.transform(np.column_stack((X_mis_cal, Y_cal)))[:, :-1]
                        M_cal = np.zeros_like(M_cal)

                else:
                    X_mis_cal = pre_imputer.transform(X_mis_cal)
                    M_cal = np.zeros_like(M_cal)

            self.model_RN_derivative = None
            assert not (self.accept_reject and self.reweight), f"Invalid state: bool_var1={self.accept_reject}, bool_var2={self.reweight}"

            self.mask_class_model = None
            if self.accept_reject or self.reweight:
                if self.generalized_pre_impute:
                    train_imputed_train_C = pre_imputer.transform(
                        np.column_stack((X_mis_train, Y_train)))[:, :-1]
                else:
                    train_imputed_train_C = pre_imputer.transform(X_mis_train)

                train_concat = np.column_stack([train_imputed_train_C, Y_train])
                cal_concat = np.column_stack([X_mis_cal, Y_cal])

                if self.RN_train_with_cal:
                    train_concat = np.vstack(( train_concat,cal_concat))
                    M_train_concat = np.vstack((M_train, M_cal_unchanged))
                else:
                    M_train_concat = M_train.copy()

                self.mask_class_data = Data_for_mask(
                    train_concat,
                    M_train_concat,
                    batch_size=128,
                    contain_y=True,
                    dataset=self.dataset_name)

                assert self.RN_model.lower() in ['histgbdt', 'xgboost', 'logistic'], '{} not correct'.format(self.RN_model)
                if "concrete" in self.dataset_name:
                    self.mask_class_model = Miss_proba_simple_model(self.mask_class_data, model=self.RN_model, epoch=1000)
                else:
                    self.mask_class_model = Miss_proba_simple_model(self.mask_class_data, model=self.RN_model, epoch=1000)
                self.mask_class_model.train()

            for key in keys_test:
                X_imp_test[key] = model_imputer.transform(X_mis_test[key])
                # this is to transform X_mis_test to a complete one, using original imputation

            if self.generalized_pre_impute and self.generalized_impute_train:
                features_train = X_imp_train_with_Y
            else:
                features_train = X_imp_train
            features_test = X_imp_test

            trained_model = fit_basemodel(features_train, Y_train, target=target,
                                          basemodel=self.basemodel, alpha=self.alpha,
                                          params_basemodel=self.params_basemodel)

            groups_test = dict.fromkeys(keys_test)
            for key in keys_test:
                groups_test[key] = list(map(utils.pattern_to_id, M_test[key].astype(int)))
            preds = dict.fromkeys(keys_test)

            for key in keys_test:
                pred = self.calibrate_predict(fitted_basemodel=trained_model,
                                            model_imputer=model_imputer,
                                            X_cal=X_mis_cal,
                                            M_cal=M_cal,
                                            Y_cal=Y_cal,
                                            X_mis_test=X_mis_test[key],
                                            features_test=features_test[key],
                                            M_test=M_test[key],
                                            groups_test=groups_test[key],
                                            target=target,
                                            M_cal_unchanged=M_cal_unchanged
                                           )
                preds[key] = pred

            if results is None:
                results = dict.fromkeys(keys_test)
                for key in keys_test:
                    results[key] = {'Y_inf': np.array(preds[key]['y_inf']),
                                    'Y_sup': np.array(preds[key]['y_sup'])}
            else:
                for key in keys_test:
                    results[key]['Y_inf'] = np.vstack(
                        (results[key]['Y_inf'], np.array(preds[key]['y_inf'])))
                    results[key]['Y_sup'] = np.vstack(
                        (results[key]['Y_sup'], np.array(preds[key]['y_sup'])))

        self.results = results
        return results

    def run_simple_synthetic(self):
        results = None

        for k in tqdm(range(self.n_rep), disable=not self.is_tqdm):
            self.k = k
            if not self.is_tqdm:
                print('{} rep over {}'.format(k,self.n_rep))

            X_mis_train = self.data['X_missing']['Train'][k, :, :]
            X_mis_cal = self.data['X_missing']['Cal'][k, :, :]

            M_train = self.data['M']['Train'][k, :, :]
            M_train_unchanged = M_train.copy()
            M_cal = self.data['M']['Cal'][k, :, :]
            M_cal_unchanged = M_cal.copy()

            Y_train = self.data['Y']['Train'][k, :]
            Y_cal = self.data['Y']['Cal'][k, :]

            keys_test = list(self.data['X_missing']['Test'].keys())
            X_mis_test = dict.fromkeys(keys_test)
            M_test = dict.fromkeys(keys_test)
            Y_test = dict.fromkeys(keys_test)

            for key in keys_test:
                X_mis_test[key] = self.data['X_missing']['Test'][key][k, :, :]
                M_test[key] = self.data['M']['Test'][key][k, :, :]
                Y_test[key] = self.data['Y']['Test'][key][k, :]

            target = 'Quantiles'

            if self.pre_impute:

                class Simple_preimputer:
                    def __init__(self, loader):
                        self.loader = loader

                    def transform(self,X):
                        Y = X[:, -1]
                        X = X[:, :-1]
                        return np.column_stack([self.loader.impute_with_perturbation(X,Y),Y])

                    def fit(self,X):
                        pass

                pre_imputer = Simple_preimputer(self.data)

            model_imputer = utils.get_impute(self.model_imputation)

            model_imputer.fit(X_mis_train)


            X_imp_train = model_imputer.transform(X_mis_train)
            if self.generalized_pre_impute and self.generalized_impute_train:
                X_imp_train_with_Y = pre_imputer.transform(np.column_stack((X_mis_train,Y_train)))[:, :-1]

            X_imp_test = dict.fromkeys(keys_test)
            X_mis_cal = pre_imputer.transform(np.column_stack((X_mis_cal, Y_cal)))[:, :-1]
            M_cal = np.zeros_like(M_cal)

            self.model_RN_derivative = None
            assert not (self.accept_reject and self.reweight), f"Invalid state: bool_var1={self.accept_reject}, bool_var2={self.reweight}"

            self.mask_class_model = None
            if self.accept_reject or self.reweight:
                if self.generalized_pre_impute:
                    train_imputed_train_C = pre_imputer.transform(
                        np.column_stack((X_mis_train, Y_train)))[:, :-1]
                else:
                    train_imputed_train_C = pre_imputer.transform(X_mis_train)

                train_concat = np.column_stack([train_imputed_train_C, Y_train])
                cal_concat = np.column_stack([X_mis_cal, Y_cal])

                train_concat = np.vstack(( train_concat,cal_concat))
                M_train_concat = np.vstack((M_train, M_cal_unchanged))

                self.mask_class_data = Data_for_mask_lr(
                    train_concat,
                    M_train_concat,
                    batch_size=128,
                    contain_y=True,
                    dataset=self.dataset_name)

                assert self.RN_model.lower() in ['histgbdt', 'xgboost', 'logistic', 'oracle'], '{} not correct'.format(self.RN_model)

                if self.RN_model.lower() == 'oracle':
                    class oracle_RN:
                        def __init__(self, dataloader):
                            self.dataloader = dataloader

                        def eval(self, x):
                            Y = x[:, -1]
                            X = x[:, :-1]
                            M = np.isnan(X).astype(int)
                            lrs = []
                            for i in range(X.shape[0]):
                                obs_index = [i for i, val in enumerate(M[i]) if val == 0]
                                lrs.append(self.dataloader.likelihood_ratio_estimation(X[i][obs_index], Y[i], obs_index))
                            return np.array(lrs)
                    self.mask_class_model = oracle_RN(self.data)
                else:
                    self.mask_class_model = Miss_proba_simple_model(self.mask_class_data, model=self.RN_model, epoch=1000)
                    self.mask_class_model.train()

                if self.RN_model.lower() != 'oracle':
                    M_lr = [1, 1, 0]
                    M_index = [i for i, val in enumerate(M_lr) if val == 1]
                    obs_index = [i for i, val in enumerate(M_lr) if val == 0]
                    n_test = len(X_mis_cal)
                    lr_esti_vals = []
                    lr_true_vals = []

                    for i in range(n_test):
                        x_missing = X_mis_cal[i].copy()
                        x_missing[M_index] = np.nan

                        x = X_mis_cal[:, obs_index][i]
                        y = Y_cal[i]

                        lr_estimation = self.mask_class_model.eval(np.append(x_missing, y).reshape(1, -1))
                        lr_true = self.data.likelihood_ratio_estimation(x, y, obs_index)

                        lr_esti_vals.append(lr_estimation[0])
                        lr_true_vals.append(lr_true)
                    print("correlation", np.corrcoef(lr_true_vals, lr_esti_vals)[0, 1])

            for key in keys_test:
                X_imp_test[key] = model_imputer.transform(X_mis_test[key])
                # this is to transform X_mis_test to a complete one, using original imputation

            if self.generalized_pre_impute and self.generalized_impute_train:
                features_train = X_imp_train_with_Y
            else:
                features_train = X_imp_train
            features_test = X_imp_test

            trained_model = fit_basemodel(features_train, Y_train, target=target,
                                          basemodel=self.basemodel, alpha=self.alpha,
                                          params_basemodel=self.params_basemodel)

            groups_test = dict.fromkeys(keys_test)
            for key in keys_test:
                groups_test[key] = list(map(utils.pattern_to_id, M_test[key].astype(int)))
            preds = dict.fromkeys(keys_test)

            for key in keys_test:
                pred = self.calibrate_predict(fitted_basemodel=trained_model,
                                            model_imputer=model_imputer,
                                            X_cal=X_mis_cal,
                                            M_cal=M_cal,
                                            Y_cal=Y_cal,
                                            X_mis_test=X_mis_test[key],
                                            features_test=features_test[key],
                                            M_test=M_test[key],
                                            groups_test=groups_test[key],
                                            target=target,
                                            M_cal_unchanged=M_cal_unchanged
                                           )

                preds[key] = pred

            if results is None:
                results = dict.fromkeys(keys_test)
                for key in keys_test:
                    results[key] = {'Y_inf': np.array(preds[key]['y_inf']),
                                    'Y_sup': np.array(preds[key]['y_sup'])}
            else:
                for key in keys_test:
                    results[key]['Y_inf'] = np.vstack(
                        (results[key]['Y_inf'], np.array(preds[key]['y_inf'])))
                    results[key]['Y_sup'] = np.vstack(
                        (results[key]['Y_sup'], np.array(preds[key]['y_sup'])))

        self.results = results
        return results

    def save(self, file_name):
        """save class as self.name.txt"""
        file = open('./data/' + file_name + '.pkl', 'wb')
        pickle.dump(self.__dict__, file)
        file.close()

    def load(self, file_name):
        """try load self.name.txt"""
        file = open('./data/' + file_name + '.pkl', 'rb')
        self.__dict__ = pickle.load(file)
        file.close()

    def calculate_results_pattern(self):

        key = -1

        data, results = self.data, self.results
        contains, lengths = utils.compute_PI_metrics(data, results, 'iid')
        metrics = utils.compute_metrics_cond(self.n_rep, data, results,
                                             'fixed_nb_sample_pattern',
                                             cond='Pattern')

        dict_cov = {}
        dict_len = {}

        dict_cov[key] = np.mean(contains, axis=1)
        dict_len[key] = np.mean(lengths, axis=1)

        avg_cov = {}
        for key_pattern in metrics.keys():
            dict_cov[key_pattern] = metrics[key_pattern]['avg_cov']
            dict_len[key_pattern] = metrics[key_pattern]['avg_len']
            avg_cov[key_pattern] = np.mean(metrics[key_pattern]['avg_cov'])
            print(key_pattern,
                  avg_cov[key_pattern], np.std(metrics[key_pattern]['avg_cov'],ddof=1),
                  np.mean(metrics[key_pattern]['avg_len']), np.std(metrics[key_pattern]['avg_len'],ddof=1))

        worst_index = np.argmin(list(avg_cov.values()))
        worst_key = list(avg_cov.keys())[worst_index]
        worst_coverage = dict_cov[worst_key]
        worst_len = dict_len[worst_key]

        best_index = np.argmax(list(avg_cov.values()))
        best_key = list(avg_cov.keys())[best_index]
        best_coverage = dict_cov[best_key]
        best_len = dict_len[best_key]

        table_cln = ['marginal_cov', 'marginal_cov_len']
        table_value = [dict_cov[-1], dict_len[-1]]

        for i, j in enumerate(metrics.keys()):
            table_cln.extend(['mask_cond_cov_num_mask_{}'.format(j),
                              'mask_cond_cov_len_num_mask_{}'.format(j), ])

            table_value.extend([dict_cov[j], dict_len[j], ])

        tbl = pd.DataFrame(np.array(table_value).T, columns=table_cln)

        results_dict = {}
        results_dict['marg_coverage'] = np.mean(dict_cov[-1])
        results_dict['marg_coverage_std'] = np.std(dict_cov[-1], ddof=1)

        results_dict['marg_cov_len'] = np.mean(dict_len[-1])
        results_dict['marg_cov_len_std'] = np.std(dict_len[-1], ddof=1)

        results_dict['mask_cov_best'] = np.mean(best_coverage)
        results_dict['mask_cov_best_std'] = np.std(best_coverage, ddof=1)

        results_dict['mask_cov_best_len'] = np.mean(best_len)
        results_dict['mask_cov_best_len_std'] = np.std(best_len, ddof=1)

        results_dict['mask_cov_worst'] = np.mean(worst_coverage)
        results_dict['mask_cov_worst_std'] = np.std(worst_coverage, ddof=1)

        results_dict['mask_cov_worst_len'] = np.mean(worst_len)
        results_dict['mask_cov_worst_len_std'] = np.std(worst_len, ddof=1)

        # results_dict['results'] = tbl
        print(results_dict)
        return tbl, results_dict

    def calibrate_predict(self, fitted_basemodel, model_imputer,
                          X_cal, M_cal, Y_cal,
                          X_mis_test, features_test, M_test,
                          groups_test, target='Quantiles',
                          M_cal_unchanged=None,
                          reweight_with_y=False):
        exact = self.exact
        alpha = self.alpha
        mechanism = self.data['mechanism']
        accept_reject = self.accept_reject
        reweight = self.reweight
        basemodel = self.basemodel
        mask_class_model = self.mask_class_model
        # X_cal: calibration with missing value( if impute first then imputed cal data)

        assert target in ['Quantiles']
        assert basemodel in ['Linear', 'RF', 'NNet', 'XGBoost', 'histgbdt', 'NNet_missing']

        impute_and_mask_conditional = False
        if impute_and_mask_conditional:
            M_test = np.zeros_like(M_test)
            M_cal = np.zeros_like(M_cal)
            X_cal = model_imputer.transform(X_cal.copy())
            X_test = model_imputer.transform(X_mis_test.copy())
            print(X_test[0], features_test[0])
            assert np.array_equal(X_test, features_test), "Arrays are not equal"

        patterns = np.unique(M_test, axis=0)
        ids = list(map(utils.pattern_to_id, patterns.astype(int)))

        n_test = features_test.shape[0]
        q_scores_test = np.empty(n_test)

        if not exact:
            pred_test = {'y_inf': np.empty(n_test),
                         'y_sup': np.empty(n_test)}

        if reweight:
            pred_test_reweight = {'y_inf': np.ones(n_test) * np.inf,
                                  'y_sup': np.ones(n_test) * -np.inf}

        for idp, id_pattern in enumerate(ids):  # iteration over test missingness patterns
            test_index = (np.array(groups_test) == id_pattern).flatten()
            test_index_list = np.where(test_index)[0]

            X_imp_cal_masking = copy.deepcopy(X_cal)
            M_cal_masking = copy.deepcopy(M_cal)
            Y_cal_masking = copy.deepcopy(Y_cal)

            pattern = patterns[idp]
            empty = False

            if exact:
                ind_subsample = np.all(M_cal[:, pattern == 0] == 0, axis=1)
                if np.sum(ind_subsample) == 0:
                    empty = True
                X_imp_cal_masking = X_imp_cal_masking[ind_subsample, :]
                M_cal_masking = M_cal_masking[ind_subsample, :]
                Y_cal_masking = Y_cal_masking[ind_subsample]

            if X_imp_cal_masking.shape[1] > len(pattern):
                nb = X_imp_cal_masking.shape[1] - len(pattern)
                pattern_ext = np.append(pattern, np.full(nb, 0))
            else:
                pattern_ext = pattern

            X_imp_cal_masking[:, pattern_ext == 1] = np.nan
            M_cal_masking[:, pattern == 1] = 1

            if reweight:
                shift_scores = mask_class_model.eval(np.column_stack([X_imp_cal_masking, Y_cal]))
                X_ampute_test = features_test.copy()
                X_ampute_test[M_test == 1] = np.nan
                X_ampute_test = X_ampute_test[test_index]

            if accept_reject:
                shift_scores = mask_class_model.normalize_eval(np.column_stack([X_imp_cal_masking, Y_cal]))
                random_uniform = np.random.uniform(0, 1, size=len(shift_scores))
                if mechanism == 'MCAR':
                    include_mask = utils.check_condition(M_cal_unchanged, pattern_ext)
                else:
                    include_mask = np.zeros_like(random_uniform, dtype=bool)
                accepted_indices = np.where((random_uniform < shift_scores) | include_mask)[0]
                X_imp_cal_masking = X_imp_cal_masking[accepted_indices]
                M_cal_masking = M_cal_masking[accepted_indices]
                Y_cal_masking = Y_cal_masking[accepted_indices]

            if not empty:
                X_imp_cal_masking = model_imputer.transform(X_imp_cal_masking)
                features_cal = X_imp_cal_masking
                cal_predictions = predict_basemodel(fitted_basemodel, features_cal, target, basemodel, alpha)
                scores = np.maximum(cal_predictions['y_inf'] - Y_cal_masking, Y_cal_masking - cal_predictions['y_sup'])

            if exact == True:

                if not empty:

                    q_scores_cal = utils.quantile_corrected(scores, alpha=alpha)

                    if reweight:
                        X_to_pred = copy.deepcopy(X_mis_test[(np.array(groups_test) == id_pattern).flatten(), :])
                        X_to_pred_masking = copy.deepcopy(X_to_pred)
                        X_to_pred_masking = model_imputer.transform(X_to_pred_masking)

                        features_test_pattern = X_to_pred_masking

                        preds_this_test = predict_basemodel(fitted_basemodel, features_test_pattern, target, basemodel,
                                                            alpha)
                        if "synthetic" in self.dataset_name:
                            min_score = min(scores) - 0.5
                            max_score = max(scores) + 0.5
                            grid_number = 30
                        else:
                            # For real data, since we don't know the scale to use grid search,
                            # we modify them to guaranty the robusteness
                            min_score = min(scores) - 15
                            max_score = max(scores) + 15
                            grid_number = 10
                        each_grid = (-min_score + max_score) / grid_number

                        for grid in np.linspace(-max_score, -min_score, grid_number):
                            y_inf = preds_this_test['y_inf'] + grid
                            shift_scores_y = mask_class_model.eval(np.column_stack([X_ampute_test,
                                                                                    y_inf.reshape(-1, 1)
                                                                                    ]))
                            test_len = shift_scores_y.shape[0]
                            array_n = np.asarray(shift_scores).reshape(1, -1)
                            array_m = np.asarray(shift_scores_y).reshape(-1, 1)
                            # print(grid, shift_scores, shift_scores_y)

                            combined_y = np.hstack([array_n.repeat(len(array_m), axis=0), array_m])
                            weights_y = combined_y / combined_y.sum(axis=1, keepdims=True)

                            expanded_array_y = np.tile(scores, (test_len, 1))  # Repeat array m times
                            expanded_array_y = np.hstack(
                                (expanded_array_y, np.full((test_len, 1), np.nan)))  # Add a column of NaN

                            quantiles_y = utils.weighted_quantile_matrix(expanded_array_y, weights_y, q=1-alpha)

                            score_y = np.maximum(preds_this_test['y_inf'] - y_inf, y_inf - preds_this_test['y_sup'])
                            indices = np.where(score_y <= quantiles_y)[0]
                            pred_test_reweight['y_inf'][test_index_list[indices]] = np.minimum(pred_test_reweight['y_inf'][test_index_list[indices]], y_inf[indices])

                            y_sup = preds_this_test['y_sup'] - grid
                            shift_scores_y = mask_class_model.eval(np.column_stack([X_ampute_test,
                                                                                    y_sup.reshape(-1, 1)
                                                                                    ]))
                            test_len = shift_scores_y.shape[0]
                            array_n = np.asarray(shift_scores).reshape(1, -1)
                            array_m = np.asarray(shift_scores_y).reshape(-1, 1)

                            combined_y = np.hstack([array_n.repeat(len(array_m), axis=0), array_m])
                            weights_y = combined_y / combined_y.sum(axis=1, keepdims=True)

                            expanded_array_y = np.tile(scores, (test_len, 1))  # Repeat array m times
                            expanded_array_y = np.hstack(
                                (expanded_array_y, np.full((test_len, 1), np.nan)))  # Add a column of NaN

                            quantiles_y = utils.weighted_quantile_matrix(expanded_array_y, weights_y, q=1 - alpha)
                            score_y = np.maximum(preds_this_test['y_inf'] - y_sup, y_sup - preds_this_test['y_sup'])
                            indices = np.where(score_y <= quantiles_y)[0]
                            pred_test_reweight['y_sup'][test_index_list[indices]] = np.maximum(pred_test_reweight['y_sup'][test_index_list[indices]], y_sup[indices])

                        index_inf_to_fill = np.where(pred_test_reweight['y_inf'][test_index_list] == np.inf)[0]
                        pred_test_reweight['y_inf'][test_index_list[index_inf_to_fill]] = y_inf[index_inf_to_fill]

                        index_sup_to_fill = np.where(pred_test_reweight['y_sup'][test_index_list] == -np.inf)[0]
                        pred_test_reweight['y_sup'][test_index_list[index_sup_to_fill]] = y_sup[index_sup_to_fill]

                        pred_test_reweight['y_sup'][test_index_list] = pred_test_reweight['y_sup'][test_index_list] + each_grid
                        pred_test_reweight['y_inf'][test_index_list] = pred_test_reweight['y_inf'][test_index_list] - each_grid

                    else:
                        q_scores_test[(np.array(groups_test) == id_pattern).flatten()] = q_scores_cal

                else:
                    q_scores_test[(np.array(groups_test) == id_pattern).flatten()] = np.inf

            else:
                assert not exact
                X_to_pred = copy.deepcopy(X_mis_test[(np.array(groups_test) == id_pattern).flatten(), :])
                M_to_pred = copy.deepcopy(M_test[(np.array(groups_test) == id_pattern).flatten(), :])
                n_current = X_to_pred.shape[0]

                patterns_cal = np.unique(M_cal_masking, axis=0)
                ids_cal = list(map(utils.pattern_to_id, patterns_cal.astype(int)))

                groups_cal_masking = list(map(utils.pattern_to_id, M_cal_masking.astype(int)))
                nb_cal = len(scores)

                all_preds = {'y_inf': np.empty((n_current, nb_cal)),
                             'y_sup': np.empty((n_current, nb_cal))}

                for idp_cal, id_pattern_cal in enumerate(ids_cal):

                    idx_cal_masking = (np.array(groups_cal_masking) == id_pattern_cal).flatten()
                    nb_mask = np.sum(idx_cal_masking)
                    all_preds['y_inf'][:, idx_cal_masking] = np.tile(scores[idx_cal_masking], (n_current, 1))
                    all_preds['y_sup'][:, idx_cal_masking] = np.tile(scores[idx_cal_masking], (n_current, 1))

                    pattern_masking = patterns_cal[idp_cal]

                    X_to_pred_masking = copy.deepcopy(X_to_pred)
                    M_to_pred_masking = copy.deepcopy(M_to_pred)
                    X_to_pred_masking[:, pattern_masking == 1] = np.nan
                    M_to_pred_masking[:, pattern_masking == 1] = 1

                    X_to_pred_masking = model_imputer.transform(X_to_pred_masking)
                    features_test_pattern = X_to_pred_masking

                    preds_k = predict_basemodel(fitted_basemodel, features_test_pattern, target, basemodel, alpha)
                    all_preds['y_inf'][:, idx_cal_masking] = -np.tile(preds_k['y_inf'], (nb_mask, 1)).T + all_preds['y_inf'][:,idx_cal_masking]
                    all_preds['y_sup'][:, idx_cal_masking] = np.tile(preds_k['y_sup'], (nb_mask, 1)).T + all_preds['y_sup'][:,idx_cal_masking]
                if (1 - alpha) * (1 + 1 / nb_cal) > 1:
                    pred_test['y_inf'][(np.array(groups_test) == id_pattern).flatten()] = [-np.inf] * n_current
                    pred_test['y_sup'][(np.array(groups_test) == id_pattern).flatten()] = [np.inf] * n_current
                else:
                    pred_test['y_inf'][(np.array(groups_test) == id_pattern).flatten()] = -np.quantile(all_preds['y_inf'], (1 - alpha) * (1 + 1 / nb_cal), axis=1)
                    pred_test['y_sup'][(np.array(groups_test) == id_pattern).flatten()] = np.quantile(all_preds['y_sup'], (1 - alpha) * (1 + 1 / nb_cal), axis=1)

        if exact == True:
            if reweight:
                if reweight_with_y:
                    pred_test = predict_basemodel(fitted_basemodel, features_test, target, basemodel, alpha)
                    interval_predictions = {'y_inf': pred_test['y_inf'] - q_scores_test,
                                            'y_sup': pred_test['y_sup'] + q_scores_test}
                else:
                    interval_predictions = {'y_inf': pred_test_reweight['y_inf'],
                                            'y_sup': pred_test_reweight['y_sup']}
                    # print(interval_predictions)

            else:
                pred_test = predict_basemodel(fitted_basemodel, features_test, target, basemodel, alpha)
                interval_predictions = {'y_inf': pred_test['y_inf'] - q_scores_test,
                                        'y_sup': pred_test['y_sup'] + q_scores_test}
        else:
            interval_predictions = {'y_inf': pred_test['y_inf'],
                                    'y_sup': pred_test['y_sup']}

        return interval_predictions



if __name__ == '__main__':
    from dataloader import Synthetic_linear_data
    data = Synthetic_linear_data(100,500, 100, 2000, 100,
                                 'MCAR', 0.5, [1,1,1,1,1,1,1,1,1,1],[])
    data.generate_multiple_data()

    print(data.M['Cal'][0].sum(axis=1))
    print(data.M['Cal'][0].sum(axis=0))

    data.X_missing['Cal'] = data.X['Cal']
    data.M['Cal'] = np.zeros_like(data.M['Cal'])

    print(data.M['Cal'][0].sum(axis=1))
    print(data.M['Cal'][0].sum(axis=0))

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from tqdm import tqdm


    def quantile_regression_with_mask_evaluation_v2(data, alpha=0.1):
        results_by_pattern = {}

        for k in tqdm(range(data.n_rep)):
            X_train = np.concatenate((data.X['Train'][k], data.X['Cal'][k]), axis=0)
            y_train = np.concatenate((data.Y['Train'][k], data.Y['Cal'][k]), axis=0)
            X_test_dict = data.X['Test']
            y_test_dict = data.Y['Test']
            M_test_dict = data.M['Test']

            model_low = HistGradientBoostingRegressor(loss="quantile", quantile=alpha / 2).fit(X_train, y_train)
            model_high = HistGradientBoostingRegressor(loss="quantile", quantile=1 - alpha / 2).fit(X_train, y_train)

            for key in X_test_dict.keys():
                X_test = X_test_dict[key][k]
                y_test = y_test_dict[key][k]
                M_test = M_test_dict[key][k]

                y_lower = model_low.predict(X_test)
                y_upper = model_high.predict(X_test)
                y_in_interval = (y_test >= y_lower) & (y_test <= y_upper)
                interval_len = y_upper - y_lower

                for i in range(len(y_test)):
                    pattern = tuple(M_test[i])
                    if pattern not in results_by_pattern:
                        results_by_pattern[pattern] = {'coverages': [], 'lengths': []}
                    results_by_pattern[pattern]['coverages'].append(y_in_interval[i])
                    results_by_pattern[pattern]['lengths'].append(interval_len[i])

        summary = {
            'mask_pattern': [],
            'avg_coverage': [],
            'std_coverage': [],
            'avg_length': [],
            'std_length': []
        }

        for pattern, stats in results_by_pattern.items():
            summary['mask_pattern'].append(pattern)
            summary['avg_coverage'].append(np.mean(stats['coverages']))
            summary['std_coverage'].append(np.std(stats['coverages'], ddof=1))
            summary['avg_length'].append(np.mean(stats['lengths']))
            summary['std_length'].append(np.std(stats['lengths'], ddof=1))

        df_summary = pd.DataFrame(summary)
        # tools.display_dataframe_to_user(name="Quantile Regression Mask-Wise Results", dataframe=df_summary)
        return df_summary

    quantile_regression_with_mask_evaluation_v2(data)
    # exp = Missing_experiments_MCAR(data, None, True, 'MICE', 'iterative_ridge',
    #                                False, False, 'histgbdt', False,
    #                                False, False, 0,
    #                           'RF', 0.1, 8, True,
    #                                {'n_estimators': 100, 'min_samples_leaf': 3, 'max_features': 'sqrt'},
    #                                ocacle_pre_impute=True)
    # exp.run_experiments()
    # exp.calculate_results_pattern()
