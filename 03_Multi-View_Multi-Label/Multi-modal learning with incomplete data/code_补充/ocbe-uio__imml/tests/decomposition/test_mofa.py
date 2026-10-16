import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.decomposition import MOFA


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_default_params(sample_data):
    transformer = MOFA()
    for Xs in sample_data:
        transformer.fit(Xs)
        assert hasattr(transformer, 'mofa_')
        assert hasattr(transformer, 'weights_')

def test_init(sample_data):
    n_components = 5
    transformer = MOFA(n_components=n_components, random_state=42)
    assert transformer.n_components == n_components
    assert transformer.random_state == 42

def test_fit(sample_data):
    n_components = 5
    transformer = MOFA(n_components=n_components, random_state=42)
    for Xs in sample_data:
        transformer.fit(Xs)
        assert hasattr(transformer, 'mofa_')
        assert hasattr(transformer, 'weights_')

def test_transform(sample_data):
    n_components = 5
    transformer = MOFA(n_components=n_components, random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        transformer.fit(Xs)
        transformed_X = transformer.transform(Xs)
        assert len(transformed_X) == len(Xs)
        assert transformed_X[0].shape == (n_samples, n_components)
        assert len(transformer.weights_) == len(Xs)
        assert transformer.weights_[0].shape == (Xs[0].shape[1], n_components)

def test_fit_transform(sample_data):
    n_components = 5
    transformer = MOFA(n_components=n_components, random_state=42)
    for Xs in sample_data:
        n_samples = len(Xs[0])
        transformed_X = transformer.fit_transform(Xs)
        assert not np.isnan(transformed_X).any().any()
        assert transformed_X.shape == (n_samples, n_components)
        assert len(transformer.weights_) == len(Xs)
        assert transformer.weights_[0].shape == (Xs[0].shape[1], n_components)

def test_missing_values_handling(sample_data):
    n_components = 5
    transformer = MOFA(n_components=n_components, random_state=42)
    for Xs in sample_data:
        Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
        n_samples = len(Xs[0])
        transformed_X = transformer.fit_transform(Xs)
        assert not np.isnan(transformed_X).any().any()
        assert transformed_X.shape == (n_samples, n_components)
        assert len(transformer.weights_) == len(Xs)
        assert transformer.weights_[0].shape == (Xs[0].shape[1], n_components)

def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid n_components"):
        MOFA(n_components=0)
    with pytest.raises(ValueError, match="Invalid n_components"):
        MOFA(n_components="invalid")


if __name__ == "__main__":
    pytest.main()