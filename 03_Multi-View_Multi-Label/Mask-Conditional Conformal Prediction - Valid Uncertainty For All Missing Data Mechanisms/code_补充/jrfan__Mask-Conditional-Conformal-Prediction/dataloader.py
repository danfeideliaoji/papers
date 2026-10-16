import random
import numpy as np
import copy
import utils
import os
import pickle
from sklearn.experimental import enable_iterative_imputer  # noqa
from uci_dataloader import uci_dataset_loader
from real_dataloader import Semi_synthetic_data
np.set_printoptions(
        threshold=6,
        edgeitems=15,
        linewidth=200
    )
from real_dataloader import GetDataset

class Missing_wrapper():

    def __init__(self, name, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern,
                 mechanism, missing_prob, var_missing, var_dependent,missing_function=None, seed=0):
        self.name = name
        self.n_rep = n_rep
        self.train_size = train_size
        self.cal_size = cal_size
        self.test_size_iid = test_size_iid
        self.test_size_per_pattern = test_size_per_pattern
        assert mechanism in ('MCAR','MAR','MNAR')
        assert 0<=missing_prob<1
        self.mechanism = mechanism
        self.missing_prob = missing_prob
        self.var_missing = var_missing
        self.total_number_missing = int(2**(sum(self.var_missing)))
        self.var_dependent = var_dependent
        self.missng_function = missing_function #to implement
        self.fixed_MAR_MNAR_coeff = False
        # Create the unique name to save this data into pickle file
        self.unique_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            name, str(n_rep), str(train_size), str(cal_size), str(test_size_iid),
            str(test_size_per_pattern), mechanism, str(missing_prob*100), ''.join([str(i) for i in var_missing]),
            ''.join([str(i) for i in var_dependent]), self.fixed_MAR_MNAR_coeff,
        )
        self.raw_data_X, self.raw_data_Y = None, None
        # print(self.unique_name)

    def __getitem__(self, key):
        return self.__dict__[key]

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

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def generate_data(self, n, data_type):
        pass

    def generate_split(self):
        """
        :return: Generate X_train and Y_train
        """
        X_train, Y_train = self.generate_data(self.train_size, data_type='train')
        X_cal, Y_cal = self.generate_data(self.cal_size, data_type='cal')

        test_size = [self.test_size_iid, self.test_size_per_pattern*self.total_number_missing]
        X_test = dict.fromkeys(test_size)
        Y_test = dict.fromkeys(test_size)

        for n_test in test_size:
            X_test[n_test], Y_test[n_test] = self.generate_data(n_test, data_type='test')

        X_split = {'Train': X_train, 'Cal': X_cal, 'Test': X_test}
        Y_split = {'Train': Y_train, 'Cal': Y_cal, 'Test': Y_test}
        return X_split, Y_split

    def split_indice(self):
        """
        to split the training, calibration and test indices.
        :return:
        """
        if self.raw_data_X is None:
            self.all_indice = np.random.permutation(self.train_size+self.cal_size+self.test_size_iid)
        else:
            self.all_indice = np.random.permutation(self.raw_data_X.shape[0])

        self.train_indice = self.all_indice[:self.train_size]
        self.cal_indice = self.all_indice[self.train_size:self.train_size+self.cal_size]
        self.test_indice = self.all_indice[self.train_size+self.cal_size:]
        return

    def generate_multiple_data(self):
        """
        Depending on the missing mechanism, geneate all data into the desired forms, for each repilication of data,
        we fix a seed and add missing values on them.
        :return:
        """

        print(self.unique_name)
        if os.path.isfile('./data/' + self.unique_name + '.pkl'):
            self.load(self.unique_name)
            return self.X, self.X_missing, self.M, self.Y
        else:
            sets = ['Train', 'Cal', 'Test']
            X = dict.fromkeys(sets)
            X_missing = dict.fromkeys(sets)
            M = dict.fromkeys(sets)
            Y = dict.fromkeys(sets)
            # whether fix one single parameter for all the replications, in our experiments, we have tried both.
            # If we don't fix it, it means that for each replication, we will have different parameters for MAR or MNAR
            # meaning that the experiments cross over various MAR or MNAR settings
            if self.fixed_MAR_MNAR_coeff:
                self.set_seed(0)
                self.split_indice()
                Xk, Yk = self.generate_split()
                if self.mechanism == 'MAR':
                    coeffs, intercepts = self.generate_MAR(Xk, Yk, only_coeff=True)
                elif self.mechanism == 'MNAR':
                    coeffs, intercepts = self.generate_MNAR(Xk, Yk, only_coeff=True)

            for k in range(self.n_rep):
                print(k,'/',self.n_rep)
                self.set_seed(k)
                # For each replication, we generate indices independantly.
                self.split_indice()
                # For each replictaion, we generate original dataset independantly.
                Xk, Yk = self.generate_split()
                if self.mechanism == 'MCAR':
                    Xk_missing, Mk_missing = self.generate_MCAR(Xk)
                elif self.mechanism == 'MAR':
                    if self.fixed_MAR_MNAR_coeff:
                        Xk_missing, Mk_missing, Yk = self.generate_MAR(Xk, Yk, coeffs=coeffs, intercepts=intercepts)
                    else:
                        Xk_missing, Mk_missing, Yk = self.generate_MAR(Xk, Yk)
                elif self.mechanism == 'MNAR':
                    if self.fixed_MAR_MNAR_coeff:
                        Xk_missing, Mk_missing, Yk = self.generate_MNAR(Xk, Yk, coeffs=coeffs, intercepts=intercepts)
                    else:
                        Xk_missing, Mk_missing, Yk = self.generate_MNAR(Xk, Yk)
                else:
                    raise NotImplementedError

                # Traing and calibration data
                for set in ['Train', 'Cal']:
                    if k == 0:
                        X[set] = np.expand_dims(Xk[set], axis=0)
                        X_missing[set] = np.expand_dims(Xk_missing[set], axis=0)
                        M[set] = np.expand_dims(Mk_missing[set], axis=0)
                        Y[set] = Yk[set]
                    else:
                        X[set] = np.vstack((X[set],np.expand_dims(Xk[set], axis=0)))
                        X_missing[set] = np.vstack((X_missing[set],np.expand_dims(Xk_missing[set], axis=0)))
                        M[set] = np.vstack((M[set],np.expand_dims(Mk_missing[set], axis=0)))
                        Y[set] = np.vstack((Y[set],np.array(Yk[set])))

                # Test data, which requires a dict to save all the arrays.
                set = 'Test'
                mechanisms_test = ['iid','fixed_nb_sample_pattern']
                mechanisms_test_dict = {'iid':self.test_size_iid,
                                        'fixed_nb_sample_pattern':self.test_size_per_pattern*self.total_number_missing}
                if k == 0:
                    X[set] = dict.fromkeys(mechanisms_test)
                    X_missing[set] = dict.fromkeys(mechanisms_test)
                    M[set] = dict.fromkeys(mechanisms_test)
                    Y[set] = dict.fromkeys(mechanisms_test)
                    for key in mechanisms_test_dict:
                        n_test = mechanisms_test_dict[key]
                        X[set][key] = np.expand_dims(Xk[set][n_test], axis=0)
                        Y[set][key] = Yk[set][n_test]
                        X_missing[set][key] = np.expand_dims(Xk_missing[set][key], axis=0)
                        M[set][key] = np.expand_dims(Mk_missing[set][key], axis=0)
                else:
                    for key in mechanisms_test_dict:
                        n_test =mechanisms_test_dict[key]
                        X[set][key] = np.vstack((X[set][key], np.expand_dims(Xk[set][n_test], axis=0)))
                        Y[set][key] = np.vstack((Y[set][key], np.array(Yk[set][n_test])))
                        X_missing[set][key] = np.vstack((X_missing[set][key], np.expand_dims(Xk_missing[set][key], axis=0)))
                        M[set][key] = np.vstack((M[set][key], np.expand_dims(Mk_missing[set][key], axis=0)))

            self.X, self.X_missing, self.M, self.Y = X, X_missing, M, Y
            self.save(self.unique_name)
            return X, X_missing, M, Y

    def generate_MAR(self, X, Yk, coeffs=None, intercepts=None, only_coeff=False):
        """
        Generate MAR is more difficult than MCAR, because we need to sample
        from unlimited data to create required missingness for test data, thus we need to
        generate X and Y together each time.
        :param X: The covariate of data, of a dict form, with train, cal, test as keys.
        :param Yk: The labels of data
        :param coeffs: If None, a new coefficient for MAR will be created
        :param intercepts: If None, a new coefficient for MAR will be created
        :param only_coeff: If true, only a new coefficient of MAR will be returned.
        :return: the maksed covariates X, mask M and the label Y.
        """
        d = X['Train'].shape[1]

        prob_missing = self.missing_prob
        var_missing = self.var_missing

        X_mar_train = copy.deepcopy(X['Train'])
        X_mar_cal = copy.deepcopy(X['Cal'])
        if only_coeff:
            M_mar_train, coeffs, intercepts = utils.MAR_mask(X_mar_train, prob_missing, var_missing, return_vars=True)
            return coeffs, intercepts

        if coeffs is None or intercepts is None:
            M_mar_train, coeffs, intercepts = utils.MAR_mask(X_mar_train, prob_missing, var_missing, return_vars=True)
        else:
            M_mar_train = utils.MAR_mask(X_mar_train, prob_missing, var_missing, coeffs=coeffs, intercepts=intercepts)

        X_mar_train[M_mar_train] = np.nan

        M_mar_cal = utils.MAR_mask(X_mar_cal, prob_missing, var_missing, coeffs=coeffs, intercepts=intercepts)
        X_mar_cal[M_mar_cal] = np.nan

        mechanisms_test = ['iid','fixed_nb_sample_pattern']

        M_mar = {'Train': M_mar_train, 'Cal': M_mar_cal}
        M_mar_test = dict.fromkeys(mechanisms_test)

        X_mar = {'Train': X_mar_train, 'Cal': X_mar_cal}
        X_mar_test = dict.fromkeys(mechanisms_test)

        X_mar_iid = copy.deepcopy(X['Test'][ self.test_size_iid])
        M_mar_iid = utils.MAR_mask(X_mar_iid, prob_missing, var_missing, coeffs=coeffs, intercepts=intercepts)
        M_mar_test['iid'] = M_mar_iid
        X_mar_iid[M_mar_iid] = np.nan
        X_mar_test['iid'] = X_mar_iid

        # if 'fixed_nb_sample_pattern' in mechanisms_test:
        list_patterns = utils.create_patterns(d, var_missing)
        test_size = self.test_size_per_pattern * self.total_number_missing
        nb_sample_pattern = self.test_size_per_pattern

        M_mar_fixed_sample_pattern = np.full((test_size, d), False)
        X_mar_fixed_sample_pattern = np.empty(X['Test'][test_size].shape)
        Y_mar_fixed_sample_pattern = np.empty(Yk['Test'][test_size].shape)

        for idp, pattern in enumerate(list_patterns):
            M_mar_fixed_sample_pattern[(idp * nb_sample_pattern):((idp + 1) * nb_sample_pattern),
            np.where(np.array(pattern) == 1)[0]] = 1
            X_simulated_M, Y_simulated_M = self.simulate_X_obs_given_M(coeffs=coeffs, intercepts=intercepts,
                                                        m=pattern, n_samples=nb_sample_pattern)
            X_mar_fixed_sample_pattern[(idp * nb_sample_pattern):((idp + 1) * nb_sample_pattern),:]=X_simulated_M
            Y_mar_fixed_sample_pattern[(idp * nb_sample_pattern):((idp + 1) * nb_sample_pattern)]=Y_simulated_M

        M_mar_test['fixed_nb_sample_pattern'] = M_mar_fixed_sample_pattern
        X_mar_test['fixed_nb_sample_pattern'] = X_mar_fixed_sample_pattern
        Yk['Test'][test_size] = Y_mar_fixed_sample_pattern

        X_mar['Test'] = X_mar_test
        M_mar['Test'] = M_mar_test

        return X_mar, M_mar, Yk

    def simulate_X_obs_given_M(self, coeffs, intercepts, m, n_samples, zoom=10):
        """
        Simulate MAR by a given coefficient of MAR
        :param coeffs: the coefficient of MAR
        :param intercepts: the coefficient(intercept) of MAR
        :param m: a certain mask to generate
        :param n_samples: the number of samples to generate
        :param zoom: the zoom factor
        :return: the simulated X and Y with the given mask m under the given coefficient of MAR
        """
        m = np.array(m)

        idxs_obs = np.where(self.var_missing == 0)[0]
        idxs_missing = np.where(self.var_missing == 1)[0]
        missing_indices = np.where(m == 1)[0]

        X_obs_samples = []
        Y_samples = []
        batch_size = zoom * n_samples  # Generate samples in batches, size can be adjusted for efficiency.

        while len(X_obs_samples) < n_samples:
            X_batch, Y_batch = self.generate_data(batch_size, data_type='test')
            P_M_given_X_batch = 1 / (1 + np.exp(-(X_batch[:, idxs_obs] @ coeffs + intercepts)))
            masks = np.random.uniform(0, 1, size=P_M_given_X_batch.shape) < P_M_given_X_batch
            masks = masks.astype(int)
            accepted_indices = (masks == m[idxs_missing]).all(axis=1)
            if np.sum(accepted_indices) == 0:
                continue  # Skip this batch to avoid indexing error
            # accepted_indices = np.random.uniform(0, 1, size=batch_size) < np.prod(P_M_given_X_batch, axis=1)
            X = X_batch[accepted_indices]
            X[:, missing_indices] = np.nan

            X_obs_samples.extend(X.tolist())
            Y_samples.extend(Y_batch[accepted_indices].tolist())
        return np.array(X_obs_samples[:n_samples]), np.array(Y_samples[:n_samples])

    def generate_MNAR(self, X, Yk, coeffs=None, intercepts=None, only_coeff=False):
        """
        Generate MNAR (self mask) is more difficult than MCAR, because we need to sample
        from unlimited data to create required missingness for test data, thus we need to
        generate X and Y together each time.
        :param X: The covariate of data, of a dict form, with train, cal, test as keys.
        :param Yk: The labels of data
        :param coeffs: If None, a new coefficient for MNAR will be created
        :param intercepts: If None, a new coefficient for MNAR will be created
        :param only_coeff: If true, only a new coefficient of MNAR will be returned.
        :return: the maksed covariates X, mask M and the label Y.
        """
        d = X['Train'].shape[1]
        prob_missing = self.missing_prob
        var_missing = self.var_missing

        X_mnar_train = copy.deepcopy(X['Train'])
        if only_coeff:
            M_mnar_train, coeffs, intercepts = utils.MNAR_mask_self_logistic(X_mnar_train, prob_missing, var_missing, return_vars=True)
            return coeffs, intercepts
        if coeffs is None or intercepts is None:
            M_mnar_train, coeffs, intercepts = utils.MNAR_mask_self_logistic(X_mnar_train, prob_missing, var_missing, return_vars=True)
        else:
            M_mnar_train = utils.MNAR_mask_self_logistic(X_mnar_train, prob_missing, var_missing, coeffs=coeffs, intercepts=intercepts)
        X_mnar_train[M_mnar_train] = np.nan

        X_mnar_cal = copy.deepcopy(X['Cal'])
        M_mnar_cal = utils.MNAR_mask_self_logistic(
            X_mnar_cal, prob_missing, var_missing,
            coeffs=coeffs, intercepts=intercepts)
        X_mnar_cal[M_mnar_cal] = np.nan

        mechanisms_test = ['iid', 'fixed_nb_sample_pattern']
        M_mnar = {'Train': M_mnar_train, 'Cal': M_mnar_cal}
        X_mnar = {'Train': X_mnar_train, 'Cal': X_mnar_cal}

        X_mnar_iid = copy.deepcopy(X['Test'][self.test_size_iid])
        M_mnar_iid = utils.MNAR_mask_self_logistic(
            X_mnar_iid, prob_missing, var_missing,
            coeffs=coeffs, intercepts=intercepts)
        X_mnar_iid[M_mnar_iid] = np.nan

        list_patterns = utils.create_patterns(d, var_missing)
        test_size = self.test_size_per_pattern * self.total_number_missing
        nb_sample_pattern = self.test_size_per_pattern

        M_mnar_fixed = np.full((test_size, d), False)
        X_mnar_fixed = np.empty(X['Test'][test_size].shape)
        Y_mnar_fixed = np.empty(Yk['Test'][test_size].shape)

        for i, pattern in enumerate(list_patterns):
            start, end = i * nb_sample_pattern, (i + 1) * nb_sample_pattern
            M_mnar_fixed[start:end, np.where(np.array(pattern) == 1)[0]] = True

            X_s, Y_s = self.simulate_X_obs_given_M_self_mask(
                coeffs, intercepts, m=pattern, n_samples=nb_sample_pattern)
            X_mnar_fixed[start:end, :] = X_s
            Y_mnar_fixed[start:end] = Y_s

        X_mnar['Test'] = {'iid': X_mnar_iid,
                          'fixed_nb_sample_pattern': X_mnar_fixed}
        M_mnar['Test'] = {'iid': M_mnar_iid,
                          'fixed_nb_sample_pattern': M_mnar_fixed}
        Yk['Test'][test_size] = Y_mnar_fixed

        return X_mnar, M_mnar, Yk

    def simulate_X_obs_given_M_self_mask(self, coeffs, intercepts,
                                         m, n_samples, zoom=10):
        """
        Simulate MNAR (self mask) by a given coefficient of MNAR
        :param coeffs: the coefficient of MNAR
        :param intercepts: the coefficient(intercept) of MNAR
        :param m: a certain mask to generate
        :param n_samples: the number of samples to generate
        :param zoom: the zoom factor
        :return: the simulated X and Y with the given mask m under the given coefficient of MNAR
        """
        m = np.asarray(m)
        idxs_missing = np.where(self.var_missing == 1)[0]

        X_list, Y_list = [], []
        batch = zoom * n_samples
        while len(X_list) < n_samples:
            X_batch, Y_batch = self.generate_data(batch, data_type='test')
            logits = X_batch * coeffs + intercepts
            ps = 1. / (1. + np.exp(-logits))[:, idxs_missing]  # (batch, |missing|)
            masks = (np.random.uniform(size=ps.shape) < ps).astype(int)

            ok = (masks == m[idxs_missing]).all(axis=1)
            if not ok.any():
                continue
            X_ok = X_batch[ok]
            X_ok[:, np.where(m == 1)[0]] = np.nan

            X_list.extend(X_ok.tolist())
            Y_list.extend(Y_batch[ok].tolist())

        return (np.array(X_list[:n_samples]),
                np.array(Y_list[:n_samples]))

import itertools
from scipy.stats import multivariate_normal
from collections import defaultdict

class Simple_synthetic(Missing_wrapper):
    def __init__(self, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern,
                 mechanism, missing_prob, var_missing, var_dependent,missing_function=None, seed=5,
                 mu=np.array([0, 0, 0]),
                 Sigma=np.array([[1, 0.2, 0.1], [0.2, 1, 0.1], [0.1, 0.1, 1]]),
                 beta=np.array([-0.5, 1.2, 0.3]),
                 noise_std=1,
                 mean_shift=-0.5, mean_scale=1.2, var_perturb=0.1):
        name = 'simple_linear_synthetic'
        super().__init__(name, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern,
                         mechanism, missing_prob, var_missing, var_dependent, missing_function, seed,
                         )
        assert mechanism == 'MCAR'
        self.d = len(mu)
        self.mu = mu
        self.Sigma = Sigma
        self.beta = beta
        self.noise_std = noise_std
        np.random.seed(seed)
        self.X, self.Y = self.generate_data(1000)
        self.patterns = list(itertools.product([0, 1], repeat=self.d))
        self.generate_mcar(missing_prob=missing_prob)
        self.all_imputation(mean_shift=mean_shift, mean_scale=mean_scale, var_perturb=var_perturb)
        self.X_imputed = self.impute_with_perturbation(self.X, self.Y)
        print('Lr range estimation:', self.estimate_lr_range())

    def generate_data(self, n, data_type=None):
        X = np.random.multivariate_normal(self.mu, self.Sigma, size=n)
        Y_reg = X.dot(self.beta)
        eps = np.random.normal(loc=0, scale=self.noise_std, size=n)
        Y = Y_reg + eps
        return X, Y

    def get_conditional_mis(self, x_obs, y, obs_idx):
        mis_idx = list(set(range(self.d)) - set(obs_idx))

        mu_obs = self.mu[obs_idx]
        mu_mis = self.mu[mis_idx]

        Sigma_oo = self.Sigma[np.ix_(obs_idx, obs_idx)]
        Sigma_om = self.Sigma[np.ix_(obs_idx, mis_idx)]
        Sigma_mo = self.Sigma[np.ix_(mis_idx, obs_idx)]
        Sigma_mm = self.Sigma[np.ix_(mis_idx, mis_idx)]

        beta_obs = self.beta[obs_idx]
        beta_mis = self.beta[mis_idx]

        mu_Y = self.beta @ self.mu
        Sigma_Y = self.beta @ self.Sigma @ self.beta + self.noise_std ** 2

        Sigma_oY = Sigma_oo @ beta_obs + Sigma_om @ beta_mis
        Sigma_mY = Sigma_mo @ beta_obs + Sigma_mm @ beta_mis

        mu_obsY = np.concatenate([mu_obs, [mu_Y]])
        Sigma_obsY = np.block([
            [Sigma_oo, Sigma_oY[:, None]],
            [Sigma_oY[None, :], Sigma_Y]
        ])

        Sigma_m_obsY = np.hstack([Sigma_mo, Sigma_mY[:, None]])

        obsY_concat = np.concatenate([x_obs, [y]])
        diff = obsY_concat - mu_obsY

        cond_mean = mu_mis + Sigma_m_obsY @ np.linalg.solve(Sigma_obsY, diff)
        cond_cov = Sigma_mm - Sigma_m_obsY @ np.linalg.solve(Sigma_obsY, Sigma_m_obsY.T)

        return cond_mean, cond_cov

    def generate_mcar(self, missing_prob=0.2):
        self.missing_prob = missing_prob
        mask = np.random.rand(*self.X.shape) < self.missing_prob
        self.X_missing = self.X.copy()
        self.X_missing[mask] = np.nan
        self.M = mask

    def all_imputation(self, mean_shift=-0.2, mean_scale=1.2, var_perturb=0.1):
        self.mean_shift = mean_shift
        self.mean_scale = mean_scale
        self.var_perturb = var_perturb

        patterns = list(itertools.product([0, 1], repeat=self.d))
        self.imputation_map = {}
        for pattern in patterns: # missing pattern
            obs_idx = [i for i, val in enumerate(pattern) if val == 0]
            mis_idx = [i for i in range(self.d) if i not in obs_idx]

            mu_obs = self.mu[obs_idx]
            mu_mis = self.mu[mis_idx]

            Sigma_oo = self.Sigma[np.ix_(obs_idx, obs_idx)]
            Sigma_om = self.Sigma[np.ix_(obs_idx, mis_idx)]
            Sigma_mo = self.Sigma[np.ix_(mis_idx, obs_idx)]
            Sigma_mm = self.Sigma[np.ix_(mis_idx, mis_idx)]

            beta_obs = self.beta[obs_idx]
            beta_mis = self.beta[mis_idx]

            mu_Y = self.beta @ self.mu
            Sigma_Y = self.beta @ self.Sigma @ self.beta + self.noise_std ** 2

            Sigma_oY = Sigma_oo @ beta_obs + Sigma_om @ beta_mis
            Sigma_mY = Sigma_mo @ beta_obs + Sigma_mm @ beta_mis

            cond_mean_mis = mu_mis
            cond_cov_mis = Sigma_mm

            perturbed_mean_mis = mean_scale * cond_mean_mis + mean_shift
            perturbed_cov_mis = cond_cov_mis + np.eye(len(mis_idx)) * var_perturb

            final_mean = np.zeros(self.d + 1)
            final_mean[obs_idx] = mu_obs
            final_mean[mis_idx] = perturbed_mean_mis
            final_mean[-1] = mu_Y

            final_cov = np.zeros((self.d + 1, self.d + 1))
            final_cov[np.ix_(obs_idx, obs_idx)] = Sigma_oo
            final_cov[np.ix_(obs_idx, mis_idx)] = Sigma_om
            final_cov[np.ix_(mis_idx, obs_idx)] = Sigma_mo
            final_cov[np.ix_(mis_idx, mis_idx)] = perturbed_cov_mis
            final_cov[-1, -1] = Sigma_Y

            final_cov[np.ix_(obs_idx, [-1])] = Sigma_oY[:, None]
            final_cov[np.ix_([-1], obs_idx)] = Sigma_oY[None, :]
            final_cov[np.ix_(mis_idx, [-1])] = Sigma_mY[:, None]
            final_cov[np.ix_([-1], mis_idx)] = Sigma_mY[None, :]

            self.imputation_map[pattern] = (final_mean, final_cov)

        return self.imputation_map

    def impute_with_perturbation(self, X, Y):
        X_imputed = X
        for i in range(len(X)):
            x_row = X[i]
            y_val = Y[i]
            obs_idx = [j for j in range(self.d) if not np.isnan(x_row[j])]
            mis_idx = [j for j in range(self.d) if np.isnan(x_row[j])]

            if not mis_idx:
                continue

            x_obs = x_row[obs_idx]
            cond_mean, cond_cov = self.get_conditional_mis(x_obs, y_val, obs_idx)

            perturbation_mean = cond_mean * self.mean_scale + self.mean_shift
            perturbation_cov = cond_cov + np.eye(len(cond_cov)) * self.var_perturb

            imputed_values = np.random.multivariate_normal(perturbation_mean, perturbation_cov)

            X_imputed[i, mis_idx] = imputed_values
        return X_imputed

    def density_under_imputation_distribution(self, x, y):
        xy = np.concatenate([x, [y]])
        density = 0
        patterns = list(itertools.product([0, 1], repeat=self.d))
        mcar_prob = (self.missing_prob ** np.sum(patterns, axis=1)) * ((1-self.missing_prob) ** (self.d - np.sum(patterns, axis=1)))

        for pattern, prob in zip(patterns, mcar_prob):
            mean, cov = self.imputation_map[pattern]
            density += prob * multivariate_normal.pdf(xy, mean=mean, cov=cov)

        return density

    def get_joint_distribution(self):
        mu_Y = self.beta @ self.mu
        Sigma_Y = self.beta @ self.Sigma @ self.beta + self.noise_std ** 2
        Sigma_XY = self.Sigma @ self.beta
        Sigma_YX = Sigma_XY.T

        mu_joint = np.concatenate([self.mu, [mu_Y]])
        Sigma_joint = np.block([
            [self.Sigma, Sigma_XY[:, None]],
            [Sigma_YX[None, :], np.array([[Sigma_Y]])]
        ])
        return mu_joint, Sigma_joint

    def get_marginal_distribution(self, obs_idx):
        mu_joint, Sigma_joint = self.get_joint_distribution()
        mu_marginal = mu_joint[obs_idx]
        Sigma_marginal = Sigma_joint[np.ix_(obs_idx, obs_idx)]
        return mu_marginal, Sigma_marginal

    def get_imputed_marginal_distribution(self, obs_idx):
        imputed_mu_sigma = dict()
        for key in self.imputation_map:
            mu_joint, Sigma_joint = self.imputation_map[key]
            mu_marginal = mu_joint[obs_idx]
            Sigma_marginal = Sigma_joint[np.ix_(obs_idx, obs_idx)]
            imputed_mu_sigma[key] = (mu_marginal, Sigma_marginal)
        return imputed_mu_sigma

    def density_under_joint_distribution(self, x, y):
        mu_joint, Sigma_joint = self.get_joint_distribution()
        xy = np.concatenate([x, [y]])
        return multivariate_normal.pdf(xy, mean=mu_joint, cov=Sigma_joint)

    def marginal_density_under_imputation_distribution(self, x_obs, y, obs_idx):
        patterns = list(itertools.product([0, 1], repeat=self.d))
        density = 0
        mcar_prob = (self.missing_prob ** np.sum(patterns, axis=1)) * ((1-self.missing_prob) ** (self.d - np.sum(patterns, axis=1)))
        obs_y_concat = np.concatenate([x_obs, [y]])

        for pattern, prob in zip(patterns, mcar_prob):
            mean, cov = self.imputation_map[pattern]
            mean_obs_y = mean[np.ix_(obs_idx + [-1])]
            cov_obs_y = cov[np.ix_(obs_idx + [-1], obs_idx + [-1])]

            density += prob * multivariate_normal.pdf(obs_y_concat, mean=mean_obs_y, cov=cov_obs_y)

        return density

    def marginal_density_under_joint_distribution(self, x_obs, y, obs_idx, get_marginal_params=False):
        mu_joint, Sigma_joint = self.get_joint_distribution()
        idx = obs_idx + [-1]
        mu_marginal = mu_joint[idx]
        Sigma_marginal = Sigma_joint[np.ix_(idx, idx)]
        obs_y_concat = np.concatenate([x_obs, [y]])
        if get_marginal_params:
            return mu_marginal, Sigma_marginal
        return multivariate_normal.pdf(obs_y_concat, mean=mu_marginal, cov=Sigma_marginal)

    def likelihood_ratio_estimation(self, x_obs, y, obs_idx):
        return self.marginal_density_under_joint_distribution(x_obs, y, obs_idx) / self.marginal_density_under_imputation_distribution(x_obs, y, obs_idx)

    def estimate_lr_range(self):
        self.lr_range = defaultdict(default_min_max)
        for pattern in self.patterns:
            obs_idx = [i for i, val in enumerate(pattern) if val == 0]
            for i in range(len(self.X_imputed)):
                x,y = self.X_imputed[i], self.Y[i]
                lr = self.likelihood_ratio_estimation(x[obs_idx],y,obs_idx)
                self.lr_range[pattern][0] = min(lr, self.lr_range[pattern][0])
                self.lr_range[pattern][1] = max(lr, self.lr_range[pattern][1])
        return self.lr_range

def default_min_max():
    return [float('inf'), -float('inf')]

class Synthetic_linear_data(Missing_wrapper):

    def __init__(self, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern,
                 mechanism, missing_prob, var_missing, var_dependent,missing_function=None, seed=0,
                 dim=10,
                 x_mean=1,
                 beta=np.array([1, 2, -1, 3, -0.5, -1, 0.3, 1.7, 0.4, -0.3]),
                 rho=0.8,
                 ):
        """
        A synthetic data generator with linear response model.
        This class inherits from Missing_wrapper and generates multivariate Gaussian covariates
        with a linear outcome function and Gaussian noise.
        :param n_rep: Number of replications.
        :param train_size: Number of training samples per replication.
        :param cal_size: Number of calibration samples per replication.
        :param test_size_iid: Number of IID test samples.
        :param test_size_per_pattern: Number of test samples per missing pattern.
        :param mechanism: Missingness mechanism ('MCAR', 'MAR', or 'MNAR').
        :param missing_prob: Overall missingness probability.
        :param var_missing: Binary indicator list for which covariates are subject to missingness.
        :param var_dependent: Variables that missingness may depend on (used in MAR/MNAR).
        :param missing_function: Custom missingness function (unused here).
        :param seed: Random seed for reproducibility.
        :param dim: Number of features (dimensions).
        :param x_mean: Mean of each feature in the multivariate normal distribution.
        :param beta: Coefficient vector for the linear response.
        :param rho: Correlation coefficient for the multivariate normal distribution.
        """
        name = 'synthetic_linear'
        super().__init__(name, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern,
                         mechanism, missing_prob, var_missing, var_dependent, missing_function, seed,
                         )
        self.d = dim
        self.x_mean = x_mean
        self.beta = beta
        self.rho = rho
        self.dim = dim
        self.unique_name = self.unique_name + '_dim' + str(dim) + '_rho' + str(rho)

    def generate_data(self, n, data_type):
        """
        Generate synthetic covariates and labels.
        :param n: Number of samples to generate.
        :param data_type: Type of data being generated (e.g., 'train', 'cal', 'test'), unused here.
        :return:
            X (np.ndarray): Covariates of shape (n, d), sampled from a multivariate Gaussian.
            Y (np.ndarray): Response variable, generated from a linear model with Gaussian noise.
        """
        mean = np.full(self.d, self.x_mean)
        cov = np.full((self.d, self.d), self.rho) + (1 - self.rho) * np.eye(self.d)
        X = np.random.multivariate_normal(mean, cov, size=n)
        Y_reg = X.dot(self.beta)
        eps = np.random.normal(loc=0, scale=1, size=(n))
        Y = Y_reg + eps
        return X, Y

    def generate_MCAR(self, X):
        """
        :param X: The covariate of data, of a dict form, with train, cal, test as keys.
        :return: Add MCAR mask on the covariates, return the masked covariates and the mask itself.
        """
        d = X['Train'].shape[1]

        prob_missing = self.missing_prob
        var_missing =  self.var_missing
        nb_var_missing = int(np.sum(var_missing))

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

        mechanisms_test = ['iid','fixed_nb_sample_pattern']

        M_mcar = {'Train': M_mcar_train, 'Cal': M_mcar_cal}
        M_mcar_test = dict.fromkeys(mechanisms_test)

        X_mcar = {'Train': X_mcar_train, 'Cal': X_mcar_cal}
        X_mcar_test = dict.fromkeys(mechanisms_test)

        M_mcar_iid = np.full((self.test_size_iid, d), False)
        M_mcar_iid[:,np.where(np.array(var_missing) == 1)[0]] = (np.random.uniform(low=0,high=1,size=(self.test_size_iid,nb_var_missing)) <= (prob_missing))
        M_mcar_test['iid'] = M_mcar_iid
        X_mcar_iid = copy.deepcopy(X['Test'][self.test_size_iid])
        X_mcar_iid[M_mcar_iid] = np.nan
        X_mcar_test['iid'] = X_mcar_iid

        list_patterns = utils.create_patterns(d, var_missing)
        test_size = self.test_size_per_pattern*self.total_number_missing
        nb_sample_pattern = self.test_size_per_pattern
        M_mcar_fixed_sample_pattern = np.full((test_size, d), False)
        X_mcar_fixed_sample_pattern = copy.deepcopy(X['Test'][test_size])
        for idp, pattern in enumerate(list_patterns):
            M_mcar_fixed_sample_pattern[(idp*nb_sample_pattern):((idp+1)*nb_sample_pattern),np.where(np.array(pattern) == 1)[0]] = 1
        X_mcar_fixed_sample_pattern[M_mcar_fixed_sample_pattern] = np.nan
        M_mcar_test['fixed_nb_sample_pattern'] = M_mcar_fixed_sample_pattern
        X_mcar_test['fixed_nb_sample_pattern'] = X_mcar_fixed_sample_pattern

        X_mcar['Test'] = X_mcar_test
        M_mcar['Test'] = M_mcar_test

        return X_mcar, M_mcar

class Real_data(Missing_wrapper):

    def __init__(self, name, n_rep, mechanism, missing_prob, var_missing, var_dependent, cal_size=None, test_size_per_pattern=None, missing_function=None, seed=0, ):
        """
        Wrapper for real-world datasets with missing value simulation.
        Automatically configures dataset-specific sizes and loads raw data. We design a MAR and MNAR for it
        :param name:  Dataset name (must be one of the predefined datasets).
        :param n_rep: Number of repetitions for experiments.
        :param mechanism: Type of missingness ('MCAR', 'MAR', or 'MNAR').
        :param missing_prob: Probability of missingness.
        :param var_missing: Indicator of which features can be missing.
        :param var_dependent: Variables the missingness mechanism may depend on.
        :param cal_size: Calibration set size (for some datasets).
        :param test_size_per_pattern: Test set size per missing pattern.
        :param missing_function: Placeholder for custom missingness logic.
        :param seed: Random seed for reproducibility.
        """
        if name == 'california':
            train_size, cal_size, test_size_iid, test_size_per_pattern = 10000, 1000, 1000, 100
        elif name == 'climate_model_crashes':
            train_size, cal_size, test_size_iid, test_size_per_pattern = 340, 100, 100, 100
        elif name == 'concrete_compression':
            train_size, cal_size, test_size_iid, test_size_per_pattern = 630, cal_size, 200, test_size_per_pattern
        elif name =='meps_19':
            train_size, cal_size, test_size_iid, test_size_per_pattern = 1000, 500, 100, 100
        else:
            raise ValueError('Dataset not found')
        if name == 'meps_19':
            dataset_base_path = "./data/cqr_datasets/"
            df, target, var_missing = GetDataset(name, dataset_base_path)
        else:
            super().__init__(name, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern,
                             mechanism, missing_prob, var_missing, var_dependent, missing_function, seed,
                             )
        if name == 'meps_19':
            self.d = df.shape[1] - 1
            data_features = df.loc[:, df.columns != target]
            response = df.loc[:, target]
            self.raw_data_X = np.array(data_features)
            self.raw_data_Y = np.array(response)
        else:
            self.raw_data_X, self.raw_data_Y = uci_dataset_loader(name)
        self.fixed_MAR_MNAR_coeff = True


    def generate_data(self, n, data_type):
        """
        Generate a batch of data (train, calibration, or test).
        :param n: Number of samples to generate.
        :param data_type: One of 'train', 'cal', or 'test'.
        :return:
            X (np.ndarray): Feature matrix of shape (n, d).
            Y (np.ndarray): Labels/targets of shape (n,).
        """
        if data_type == 'train':
            train_indice = np.random.choice(self.train_indice, n, replace=False)
            X = self.raw_data_X[train_indice]
            Y = self.raw_data_Y[train_indice]
        elif data_type == 'cal':
            cal_indice = np.random.choice(self.cal_indice, n, replace=False)
            X = self.raw_data_X[cal_indice]
            Y = self.raw_data_Y[cal_indice]
        elif data_type == 'test':
            available_indices = self.test_indice
            available_size = len(available_indices)

            if n <= available_size:
                test_indice = np.random.choice(available_indices, n, replace=False)
                X = self.raw_data_X[test_indice]
                Y = self.raw_data_Y[test_indice]
                return X, Y

            X_to_shuffle = self.raw_data_X[available_indices]
            Y_to_shuffle = self.raw_data_Y[available_indices]

            n_shuffle = X_to_shuffle.shape[0]
            nb_exact_shuffle = n // n_shuffle
            nb_rest = n % n_shuffle

            X_test_created = np.empty((n, X_to_shuffle.shape[1]))
            Y_test_created = np.empty((n,))

            for k in range(nb_exact_shuffle):
                ido = np.random.permutation(n_shuffle)
                X_test_created[k * n_shuffle:(k + 1) * n_shuffle] = X_to_shuffle[ido]
                Y_test_created[k * n_shuffle:(k + 1) * n_shuffle] = Y_to_shuffle[ido]

            if nb_rest > 0:
                ido = np.random.choice(n_shuffle, nb_rest, replace=False)
                X_test_created[nb_exact_shuffle * n_shuffle:] = X_to_shuffle[ido]
                Y_test_created[nb_exact_shuffle * n_shuffle:] = Y_to_shuffle[ido]

            return X_test_created, Y_test_created
        else:
            raise ValueError('data_type not recognized')
        return X, Y

class Real_data_MCAR(Semi_synthetic_data):
    def __init__(self, n_rep=10,prob_missing=0.2,dataset_name='concrete_compression',imputation='iterative_ridge',
                 seed=None,cal_size=None,mechanism='MCAR',var_missing_new=None):
        super().__init__(n_rep,prob_missing,dataset_name,imputation,seed,cal_size,mechanism,var_missing_new)
        self.name = dataset_name
        self.unique_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            self.name, str(n_rep), str(self.train_size), str(cal_size), str(self.test_size_iid),
            str(100), mechanism, str(prob_missing * 100), ''.join(['1'] * self.d),
            '', False
        )
    def generate_multiple_data(self):
        return

class Real_data_MNAR(Missing_wrapper):

    def __init__(self, name, n_rep, mechanism, missing_prob, var_missing, var_dependent, cal_size=None, test_size_per_pattern=None, missing_function=None, seed=0, ):
        """
        Wrapper for real-world datasets with missing value simulation.
        Automatically configures dataset-specific sizes and loads raw data. We design a MAR and MNAR for it
        :param name:  Dataset name (must be one of the predefined datasets).
        :param n_rep: Number of repetitions for experiments.
        :param mechanism: Type of missingness ('MCAR', 'MAR', or 'MNAR').
        :param missing_prob: Probability of missingness.
        :param var_missing: Indicator of which features can be missing.
        :param var_dependent: Variables the missingness mechanism may depend on.
        :param cal_size: Calibration set size (for some datasets).
        :param test_size_per_pattern: Test set size per missing pattern.
        :param missing_function: Placeholder for custom missingness logic.
        :param seed: Random seed for reproducibility.
        """
        if name == 'california':
            train_size, cal_size, test_size_iid, test_size_per_pattern = 10000, 1000, 1000, 100
        elif name == 'climate_model_crashes':
            train_size, cal_size, test_size_iid, test_size_per_pattern = 340, 100, 100, 100
        elif name == 'concrete_compression':
            train_size, cal_size, test_size_iid, test_size_per_pattern = 630, cal_size, 200, test_size_per_pattern
        elif name =='meps_19':
            train_size, cal_size, test_size_iid, test_size_per_pattern = 1000, 500, 100, 100
        else:
            raise ValueError('Dataset not found')
        if name == 'meps_19':
            dataset_base_path = "./data/cqr_datasets/"
            df, target, var_missing = GetDataset(name, dataset_base_path)
        else:
            super().__init__(name, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern,
                             mechanism, missing_prob, var_missing, var_dependent, missing_function, seed,
                             )
        if name == 'meps_19':
            self.d = df.shape[1] - 1
            data_features = df.loc[:, df.columns != target]
            response = df.loc[:, target]
            self.raw_data_X = np.array(data_features)
            self.raw_data_Y = np.array(response)
        else:
            self.raw_data_X, self.raw_data_Y = uci_dataset_loader(name)


    def generate_data(self, n, data_type):
        """
        Generate a batch of data (train, calibration, or test).
        :param n: Number of samples to generate.
        :param data_type: One of 'train', 'cal', or 'test'.
        :return:
            X (np.ndarray): Feature matrix of shape (n, d).
            Y (np.ndarray): Labels/targets of shape (n,).
        """
        if data_type == 'train':
            train_indice = np.random.choice(self.train_indice, n, replace=False)
            X = self.raw_data_X[train_indice]
            Y = self.raw_data_Y[train_indice]
        elif data_type == 'cal':
            cal_indice = np.random.choice(self.cal_indice, n, replace=False)
            X = self.raw_data_X[cal_indice]
            Y = self.raw_data_Y[cal_indice]
        elif data_type == 'test':
            available_indices = self.test_indice
            available_size = len(available_indices)

            if n <= available_size:
                test_indice = np.random.choice(available_indices, n, replace=False)
                X = self.raw_data_X[test_indice]
                Y = self.raw_data_Y[test_indice]
                return X, Y

            X_to_shuffle = self.raw_data_X[available_indices]
            Y_to_shuffle = self.raw_data_Y[available_indices]

            n_shuffle = X_to_shuffle.shape[0]
            nb_exact_shuffle = n // n_shuffle
            nb_rest = n % n_shuffle

            X_test_created = np.empty((n, X_to_shuffle.shape[1]))
            Y_test_created = np.empty((n,))

            for k in range(nb_exact_shuffle):
                ido = np.random.permutation(n_shuffle)
                X_test_created[k * n_shuffle:(k + 1) * n_shuffle] = X_to_shuffle[ido]
                Y_test_created[k * n_shuffle:(k + 1) * n_shuffle] = Y_to_shuffle[ido]

            if nb_rest > 0:
                ido = np.random.choice(n_shuffle, nb_rest, replace=False)
                X_test_created[nb_exact_shuffle * n_shuffle:] = X_to_shuffle[ido]
                Y_test_created[nb_exact_shuffle * n_shuffle:] = Y_to_shuffle[ido]

            return X_test_created, Y_test_created
        else:
            raise ValueError('data_type not recognized')
        return X, Y
if __name__ == '__main__':
    dataloader = Synthetic_linear_data(
        n_rep=10, train_size=1000, cal_size=200, test_size_iid=1000, test_size_per_pattern=100,
        mechanism='MNAR', missing_prob=0.5, var_missing=np.array([1,1,1,1,1,0,0,0,0,0]), var_dependent=np.array([]))

    X, X_missing, M, Y = dataloader.generate_multiple_data()

    # print(X['Train'][1])
    # print(dataloader.M['Train'][0].sum(axis=1))
    print(dataloader.M['Train'][0][dataloader.X['Train'][0]>0.5].sum(axis=0))
    print(dataloader.M['Train'][0][dataloader.X['Train'][0]<-1].sum(axis=0))

