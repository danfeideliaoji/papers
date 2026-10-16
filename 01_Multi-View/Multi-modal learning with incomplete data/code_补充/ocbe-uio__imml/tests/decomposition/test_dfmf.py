from string import ascii_lowercase
import pytest
import numpy as np
import pandas as pd

from imml.ampute import Amputer
from imml.decomposition import DFMF


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((50, 270))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :100], X.iloc[:, 100:190], X.iloc[:, 190:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy


def test_init(sample_data):
    n_components = 5
    dfmf = DFMF(n_components=n_components, max_iter=200, init_type='random', n_run=3, random_state=42)
    assert dfmf.n_components == n_components
    assert dfmf.max_iter == 200
    assert dfmf.init_type == ['random', 'random']
    assert dfmf.n_run == 3
    assert dfmf.random_state == 42

def test_fit(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_components = 5
    dfmf = DFMF(n_components=n_components, max_iter=10, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        dfmf.fit(Xs)
        assert hasattr(dfmf, 'fuser_')
        assert hasattr(dfmf, 'transformer_')
        assert hasattr(dfmf, 'ts_')
        assert len(dfmf.ts_) == len(Xs)

def test_transform(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_components = 5
    dfmf = DFMF(n_components=n_components, max_iter=10, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        n_samples = len(Xs[0])
        dfmf.fit(Xs)
        transformed_X = dfmf.transform(Xs)
        assert transformed_X.shape == (n_samples, n_components)

def test_missing_values_handling(sample_data):
    Xs_pandas, Xs_numpy = sample_data
    n_components = 5
    dfmf = DFMF(n_components=n_components, max_iter=10, random_state=42)
    for Xs in [Xs_pandas, Xs_numpy]:
        Xs = Amputer(p= 0.3, random_state=42).fit_transform(Xs)
        n_samples = len(Xs[0])
        transformed_X = dfmf.fit_transform(Xs)
        assert not np.isnan(transformed_X).any().any()
        assert transformed_X.shape == (n_samples, n_components)

def test_dfmf_invalid_params():
    with pytest.raises(ValueError, match="Invalid n_components"):
        DFMF(n_components=0)


if __name__ == "__main__":
    pytest.main()