from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from hyperimpute.plugins.imputers import Imputers, ImputerPlugin
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import QuantileRegressor
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import quantile_forest as qf
import utils


class RegressionDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = torch.from_numpy(X_data).float()
        self.y_data = torch.from_numpy(y_data).float()

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)

class NNet(nn.Module):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self, quantiles, num_features, num_hidden=64, dropout=0.1, no_crossing=False):
        """ Initialization
        Parameters
        ----------
        quantiles : numpy array of quantile levels (q), each in the range (0,1)
        num_features : integer, input signal dimension (p)
        num_hidden : integer, hidden layer dimension
        dropout : float, dropout rate
        no_crossing: boolean, whether to explicitly prevent quantile crossovers
        """
        super(NNet, self).__init__()

        self.no_crossing = no_crossing

        self.num_quantiles = len(quantiles)

        # Construct base network
        self.base_model = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_hidden, self.num_quantiles),
        )
        self.init_weights()

    def init_weights(self):
        """ Initialize the network parameters
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        x = self.base_model(x)
        if self.no_crossing:
            y,_ = torch.sort(x,1)
        else:
            y = x
        return y

class AllQuantileLoss(nn.Module):
    """ Pinball loss function
    """
    def __init__(self, quantiles):
        """ Initialize
        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ Compute the pinball loss
        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)
        Returns
        -------
        loss : cost function value
        """
        #assert not target.requires_grad
        #assert preds.size(0) == target.size(0)

        errors = target.unsqueeze(1)-preds
        Q = self.quantiles.unsqueeze(0)
        loss = torch.max((Q-1.0)*errors, Q*errors).mean()

        return loss

def fit_basemodel(X_train, Y_train, target='Mean', basemodel='Linear', alpha=0.1, params_basemodel={}):
    """
    The function for trianing a regression model for X_train and Y_train
    :return: The trained regression model, in our paper, this is $f$
    """
    assert target in ['Mean', 'Quantiles'], 'regression must be Mean or Quantiles.'
    assert basemodel in ['Linear', 'RF', 'NNet', 'XGBoost', 'histgbdt', 'NNet_missing'], 'regression must be Linear, RF or NNet.'

    cores = params_basemodel['cores']
    # print('cores:', cores)

    if basemodel == 'RF':
        n_estimators = params_basemodel['n_estimators']
        min_samples_leaf = params_basemodel['min_samples_leaf']
        max_features = params_basemodel['max_features']

    if target == 'Mean':
        if basemodel == 'Linear':
            trained_model = LinearRegression(n_jobs=cores).fit(X_train,Y_train)
        elif basemodel == 'RF':
            trained_model = RandomForestRegressor(n_jobs=cores,n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                                  max_features=max_features, random_state=1).fit(X_train,Y_train)
        elif basemodel == 'NNet':
            n_train = len(Y_train)
            n_features = X_train.shape[1]
            epochs = 2000
            lr = 0.0005
            batch_size = n_train
            dropout = 0.1
            grid_quantiles = [0.5]
            trained_model = Nnet_mse(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                                 dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=0,
                                 verbose=False).fit(X_train, Y_train)
    elif target == 'Quantiles':
        a_low = alpha/2
        a_high = 1-alpha/2
        if basemodel == 'Linear':
            trained_model = {'q_low': QuantileRegressor(quantile=a_low, solver='highs', alpha=0).fit(X_train,Y_train),
                             'q_high': QuantileRegressor(quantile=a_high, solver='highs', alpha=0).fit(X_train,Y_train)}
        elif basemodel == 'RF':
            trained_model = qf.RandomForestQuantileRegressor(random_state=1, min_samples_leaf=min_samples_leaf,
                                                             n_estimators=n_estimators, max_features=max_features).fit(X_train,Y_train)
        elif basemodel == 'XGBoost':
            trained_model = {'q_low': GradientBoostingRegressor(loss="quantile", alpha=a_low, n_estimators=25).fit(X_train,Y_train),
                             'q_high': GradientBoostingRegressor(loss="quantile", alpha=a_high, n_estimators=25).fit(X_train,Y_train)}
        elif basemodel == 'histgbdt':
            trained_model = {'q_low': HistGradientBoostingRegressor(loss="quantile", quantile=a_low).fit(X_train,Y_train),
                             'q_high': HistGradientBoostingRegressor(loss="quantile", quantile=a_high).fit(X_train,Y_train)}

        elif basemodel == 'NNet':
            n_train = len(Y_train)
            n_features = X_train.shape[1]
            epochs = 2000
            lr = 0.0005
            batch_size = n_train
            dropout = 0.1

            grid_quantiles = [alpha/2, 1-alpha/2]
            trained_model = QNet(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                                 dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=0,
                                 verbose=False).fit(X_train, Y_train)
        elif basemodel == 'NNet_missing':
            n_train = len(Y_train)
            n_features = X_train.shape[1]
            epochs = 2000
            lr = 0.0005
            batch_size = n_train
            dropout = 0.1
            grid_quantiles = [alpha / 2, 1 - alpha / 2]
            trained_model = QNet_missing(grid_quantiles, n_features, no_crossing=True, batch_size=batch_size,
                                 dropout=dropout, num_epochs=epochs, learning_rate=lr, calibrate=0,
                                 verbose=False,).fit(X_train, Y_train)

    return trained_model

class Split_imputer:

    def __init__(self, imputation):
        """
        This is not used in our experiments
        """
        imputers = Imputers()
        self.imputer = imputers.get(imputation)
        self.train_data = None
        self.trained = False

    def fit(self, x):
        self.train_data = x
        self.trained = True

    def transform(self, x):
        if not self.trained:
            raise ValueError("The imputer must be trained with train() before calling transform().")

        combined_data = np.vstack([self.train_data, x])
        combined_data = combined_data.astype(float)

        imputed_combined_data = self.imputer.fit_transform(combined_data)

        self.train_imputed_data = imputed_combined_data[:self.train_data.shape[0]].to_numpy()
        self.test_imputed_data = imputed_combined_data[self.train_data.shape[0]:].to_numpy()
        return self.test_imputed_data

class Data_for_mask():

    def __init__(self, X, M, numerical_feature=None, batch_size=128, test=0.2, contain_y=False, dataset=None):
        """
        A PyTorch-compatible dataset wrapper to create batches for training missingness models.
        Supports artificial generation of missing masks for learning likelihood ratio models.
        :param X: Input data matrix (with or without labels).
        :param M:  Original binary missing mask matrix (True for missing).
        :param numerical_feature: Indices of numerical features (default: all).
        :param batch_size: Size of each training batch.
        :param test: Proportion of data used as a test set.
        :param contain_y: Whether the last column is the label.
        :param dataset: Optional dataset name (e.g., 'bike') for hardcoded feature handling.
        """
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.contain_y = contain_y
        if contain_y:
            self.d_x = X.shape[1] - 1
        else:
            self.d_x = X.shape[1]
        if 'bike' in dataset:
            self.d_x = 4
        if numerical_feature is None:
            self.numerical_feature = list(range(self.d))
        else:
            self.numerical_feature = numerical_feature
        self.batch_size = batch_size
        self.test_n = int(self.N * test)
        self.M_original = M.copy()
        self.X = X.copy()
        self.M_num = np.array([[True,]*i + [False,]*(self.d_x-i) for i in range(self.d_x)])

        self.scaler = StandardScaler()
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)

        self.X_test = self.X[:self.test_n]
        self.X = self.X[self.test_n:]

        self.X_test_torch = torch.tensor(self.X_test, dtype=torch.float32)
        self.X_torch = torch.tensor(self.X, dtype=torch.float32)
        self.N_train = self.X.shape[0]
        return


    def __len__(self):
        return len(self.X_torch)

    def __iter__(self):
        """
        Initialize a new epoch: generate new masks and corresponding labels.
        Labels are 1 if the sampled mask is a subset of the original mask.
        """
        self.start_idx = 0
        self.indices = torch.arange(len(self.X_torch))
        # weights = np.linspace(1, 4, self.d_x)
        # weights = weights / weights.sum()
        # indices = np.random.choice(self.d_x, size=self.N, replace=True,  p=weights)
        # Randomly sample a mask index for each row
        indices = np.random.choice(self.d_x, size=self.N, replace=True)
        self.M = self.M_num[indices]
        for row in self.M:
            np.random.shuffle(row)
        assert self.M.shape == self.M_original.shape
        # Label = 1 if sampled mask is a subset of the true mask
        row_check = (self.M & self.M_original).sum(axis=1) == self.M_original.sum(axis=1)
        self.Y = row_check.astype(int)
        # If y is appended to X, extend mask
        if self.contain_y:
            self.M = np.hstack((self.M, np.zeros((self.M.shape[0], 1))))
        # Split test/train for current epoch
        self.M_test = self.M[:self.test_n]
        self.Y_test = self.Y[:self.test_n]

        self.M = self.M[self.test_n:]
        self.Y = self.Y[self.test_n:]
        # Convert to tensors for training
        self.M_torch = torch.tensor(self.M, dtype=torch.bool)
        self.Y_torch = torch.tensor(self.Y, dtype=torch.float32).view(-1, 1)
        self.X_nan_torch = self.X_torch.masked_fill(self.M_torch, float('nan'))

        self.M_test_torch = torch.tensor(self.M_test, dtype=torch.bool)
        self.Y_test_torch = torch.tensor(self.Y_test, dtype=torch.float32).view(-1, 1)
        self.X_nan_test_torch = self.X_test_torch.masked_fill(self.M_test_torch, float('nan'))

        return self

    def __next__(self):
        """
        Returns the next training batch.
        """
        if self.start_idx < self.N_train:
            end_idx = min(self.start_idx + self.batch_size, self.N_train)
            batch_indices = self.indices[self.start_idx:end_idx]
            self.start_idx = end_idx
            return (self.X_nan_torch[batch_indices],
                    self.Y_torch[batch_indices])
        else:
            raise StopIteration

class Miss_proba_simple_model:

    def __init__(self, data, model='xgboost', epoch=300, **hyperparam):
        """
        Trains a likelihood ratio model for missingness (Algorithm 2 in the paper).
        Supports multiple classifiers including XGBoost, HistGBDT, and logistic regression.
        :param data: Masked training dataset.
        :param model: Classifier type ('xgboost', 'histgbdt', 'logistic').
        :param epoch: Number of data epochs to sample from.
        :param hyperparam: Additional hyperparameters for the classifier.
        """
        self.data = data
        self.model_type = model
        self.hyperparam = hyperparam
        self.imputer = utils.get_impute('iterative_ridge')
        self.epoch = epoch
        # Apply the quantile normalization trick as proposed by Tibshirani et al. (2020) for robusteness
        # “Conformal Prediction Under Covariate Shift”
        self.robust_normalization = 0.9
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.fetch_data(epoch=self.epoch)

        # Initialize classifier
        if model == 'xgboost':
            self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", **hyperparam)
        elif model == 'histgbdt':
            self.model = HistGradientBoostingClassifier(**hyperparam)
        elif model == 'logistic':
            self.model = LogisticRegression(**hyperparam)
        else:
            raise ValueError(f"Unsupported model type: {model}")

    def fetch_data(self, epoch=300):
        """
        Collect training data from multiple sampling epochs.
        Final epoch is reserved as test set.
        """
        all_X, all_Y = [], []
        all_X_test, all_Y_test = [], []
        for i in range(epoch):
            for x_batch, y_batch in self.data:
                x_numpy = x_batch.numpy()
                y_numpy = y_batch.numpy()
                if i == epoch - 1:  # Use the last epoch as test data
                    all_X_test.append(x_numpy)
                    all_Y_test.append(y_numpy)
                else:
                    all_X.append(x_numpy)
                    all_Y.append(y_numpy)

        X_train = np.vstack(all_X)
        Y_train = np.vstack(all_Y).flatten()  # Flatten Y to 1D array
        X_test = np.vstack(all_X_test)
        Y_test = np.vstack(all_Y_test).flatten()

        return X_train, Y_train, X_test, Y_test

    def train(self):
        """
        Trains the model on masked inputs and computes binary cross-entropy loss.
        """
        if self.model_type == 'logistic':
            train_mask = np.isnan(self.X_train).astype(int)
            x_train_imputed = self.imputer.fit_transform(self.X_train)
            self.X_train = np.hstack((x_train_imputed, train_mask))

            test_mask = np.isnan(self.X_test).astype(int)
            x_test_imputed = self.imputer.transform(self.X_test)
            self.X_test = np.hstack((x_test_imputed, test_mask))

        self.model.fit(self.X_train, self.Y_train)
        y_pred_test = self.model.predict_proba(self.X_test)[:, 1]
        try:
            self.loss = log_loss(self.Y_test, y_pred_test)
        except:
            self.loss = 0
        # print(f"Training completed.")

    def normalize_eval(self, x):
        """
        Evaluate model and return normalized scores clipped to [0, 1].
        :param x: Input batch with possible missing values.
        :return: np.ndarray: Normalized likelihood scores.
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = self.data.scaler.transform(x)
        if self.model_type == 'logistic':
            mask = np.isnan(x).astype(int)
            x_imputed = self.imputer.transform(x)
            x = np.hstack((x_imputed, mask))
        proba = self.model.predict_proba(x)[:, 1]
        normalize_proba = np.clip(proba / np.quantile(proba, self.robust_normalization),0,1)
        return normalize_proba

    def eval(self, x):
        """
        Evaluate model and return approximate odds ratio of missingness.
        :param x: x (np.ndarray or torch.Tensor): Input with missing entries.
        :return: np.ndarray: Likelihood ratio estimate (P(M|X)/1-P(M|X)).
        """
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = self.data.scaler.transform(x)
        if self.model_type == 'logistic':
            mask = np.isnan(x).astype(int)
            x_imputed = self.imputer.transform(x)
            x = np.hstack((x_imputed, mask))
        proba = self.model.predict_proba(x)[:, 1]
        proba = np.clip(proba, 0, self.robust_normalization)
        return proba/(1-proba)


def predict_basemodel(fitted_basemodel, X_test, target='Mean', basemodel='Linear', alpha=0.1):
    """
    Generate predictions using a fitted base regression model.

    Supports both point prediction (mean) and quantile prediction (for conformal intervals).
    Depending on the specified base model and prediction type, it routes to the appropriate
    prediction interface.
    :param fitted_basemodel: The trained regression model or dictionary of models.
                             For quantile prediction, expected to be a dict with keys 'q_low' and 'q_high'.
    :param X_test: ndarray of test covariates.
    :param target: 'Mean' or 'Quantiles'. Determines whether to return point or interval prediction.
    :param basemodel: One of ['Linear', 'RF', 'NNet', 'XGBoost', 'histgbdt', 'NNet_missing'].
                      Determines the interface used for prediction.
    :param alpha: Miscoverage level for quantile estimation (used when target='Quantiles').
    :return:
        - If target == 'Mean': returns a 1D ndarray of predictions.
        - If target == 'Quantiles': returns a dict with keys:
            'y_inf': lower alpha/2 quantile prediction,
            'y_sup': upper (1-alpha/2) quantile prediction.
    """
    assert target in ['Mean', 'Quantiles'], 'regression must be Mean or Quantiles.'
    assert basemodel in ['Linear', 'RF', 'NNet', 'XGBoost','histgbdt','NNet_missing'], 'regression must be Linear, RF or NNet.'

    if target == 'Mean':
        predictions = fitted_basemodel.predict(X_test)
    elif target == 'Quantiles':
        a_low = alpha/2
        a_high = 1-alpha/2
        if basemodel == 'Linear':
            predictions = {'y_inf': fitted_basemodel['q_low'].predict(X_test),
                           'y_sup': fitted_basemodel['q_high'].predict(X_test)}
        elif basemodel == 'RF':
            both_pred = fitted_basemodel.predict(X_test, quantiles=[a_low, a_high])
            predictions = {'y_inf': both_pred[:, 0],
                           'y_sup': both_pred[:, 1]}
        elif basemodel == 'XGBoost' or basemodel == 'histgbdt':
            predictions = {'y_inf': fitted_basemodel['q_low'].predict(X_test),
                           'y_sup': fitted_basemodel['q_high'].predict(X_test)}
        elif basemodel == 'NNet' or basemodel =='NNet_missing':
            both_pred = fitted_basemodel.predict(X_test)
            predictions = {'y_inf': both_pred[:, 0],
                           'y_sup': both_pred[:, 1]}
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return predictions

class Data_for_mask_lr():

    def __init__(self, X, M, numerical_feature=None, batch_size=128, test=0.2, contain_y=True, dataset=None, one_M=None):
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.contain_y = contain_y
        if contain_y:
            self.d_x = X.shape[1] - 1
        else:
            self.d_x = X.shape[1]

        if numerical_feature is None:
            self.numerical_feature = list(range(self.d))
        else:
            self.numerical_feature = numerical_feature
        self.batch_size = batch_size
        self.test_n = int(self.N * test)
        self.M_original = M.copy()
        self.X = X.copy()
        self.M_num = np.array([[True,]*i + [False,]*(self.d_x-i) for i in range(self.d_x)])

        self.scaler = StandardScaler()
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)

        self.X_test = self.X[:self.test_n]
        self.X = self.X[self.test_n:]

        self.X_test_torch = torch.tensor(self.X_test, dtype=torch.float32)
        self.X_torch = torch.tensor(self.X, dtype=torch.float32)
        self.N_train = self.X.shape[0]

        self.one_M = one_M

        return


    def __len__(self):
        return len(self.X_torch)

    def __iter__(self):
        self.start_idx = 0
        self.indices = torch.arange(len(self.X_torch))
        if self.one_M is None:
            indices = np.random.choice(self.d_x, size=self.N, replace=True)
            self.M = self.M_num[indices]
            for row in self.M:
                np.random.shuffle(row)
        else:
            self.M = np.tile(self.one_M, (self.N, 1))

        assert self.M.shape == self.M_original.shape
        row_check = (self.M & self.M_original).sum(axis=1) == self.M_original.sum(axis=1)
        self.Y = row_check.astype(int)

        if self.contain_y:
            self.M = np.hstack((self.M, np.zeros((self.M.shape[0], 1))))

        self.M_test = self.M[:self.test_n]
        self.Y_test = self.Y[:self.test_n]

        self.M = self.M[self.test_n:]
        self.Y = self.Y[self.test_n:]

        self.M_torch = torch.tensor(self.M, dtype=torch.bool)
        self.Y_torch = torch.tensor(self.Y, dtype=torch.float32).view(-1, 1)
        self.X_nan_torch = self.X_torch.masked_fill(self.M_torch, float('nan'))

        self.M_test_torch = torch.tensor(self.M_test, dtype=torch.bool)
        self.Y_test_torch = torch.tensor(self.Y_test, dtype=torch.float32).view(-1, 1)
        self.X_nan_test_torch = self.X_test_torch.masked_fill(self.M_test_torch, float('nan'))

        return self

    def __next__(self):
        if self.start_idx < self.N_train:
            end_idx = min(self.start_idx + self.batch_size, self.N_train)
            batch_indices = self.indices[self.start_idx:end_idx]
            self.start_idx = end_idx
            return (self.X_nan_torch[batch_indices],
                    self.Y_torch[batch_indices])
        else:
            raise StopIteration