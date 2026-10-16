from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.cluster import MONET


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((15, 10))
    X = pd.DataFrame(X, index=list(ascii_lowercase)[:len(X)], columns= [f"feature{i}" for i in range(X.shape[1])])
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_default_params(sample_data):
    model = MONET()
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels[~np.isnan(labels)])) == model.n_clusters_
        assert np.nanmin(labels) == 0
        assert np.nanmax(labels) == (model.n_clusters_ - 1)
        assert model.total_weight_ > 0

def test_custom_parameters(sample_data):
    for Xs in sample_data:
        n_samples = len(Xs[0])
        try:
            init_modules = {'module1': list(Xs[0].index[:10]),
                            'module2': list(Xs[0].index[10:]),
                            }
            model = MONET(random_state=42, percentile_remove_edge=20, init_modules=init_modules, verbose=True)
        except AttributeError:
            model = MONET(random_state=42, percentile_remove_edge=10, verbose=True, min_mod_size=1)
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels[~np.isnan(labels)])) == model.n_clusters_
        assert np.nanmin(labels) == 0
        assert np.nanmax(labels) == (model.n_clusters_ - 1)
        assert model.total_weight_ > 0

def test_invalid_params(sample_data):
    estimator = MONET
    with pytest.raises(ValueError, match="Invalid similarity_mode."):
        estimator(similarity_mode='invalid')

def test_fit_predict(sample_data):
    model = MONET(random_state=42, verbose=True)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels[~np.isnan(labels)])) == model.n_clusters_
        assert np.nanmin(labels) == 0
        assert np.nanmax(labels) == (model.n_clusters_ - 1)
        assert model.total_weight_ > 0

def test_missing_values_handling(sample_data):
    model = MONET(random_state=42, verbose=True)
    for Xs in sample_data:
        Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
        labels = model.fit_predict(Xs)
        assert labels is not None
        assert len(np.unique(labels[~np.isnan(labels)])) == model.n_clusters_
        assert np.nanmin(labels) == 0
        assert np.nanmax(labels) == (model.n_clusters_ - 1)
        assert model.total_weight_ > 0


if __name__ == "__main__":
    pytest.main()