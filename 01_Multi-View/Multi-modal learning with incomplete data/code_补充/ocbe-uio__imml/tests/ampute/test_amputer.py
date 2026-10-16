import pytest
import numpy as np
import pandas as pd
from imml.ampute import Amputer
from imml.explore import get_pct_incom_samples, get_pct_com_samples

@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_default_params(sample_data):
    amputer = Amputer(random_state=42)
    for Xs in sample_data:
        transformed_Xs = amputer.fit_transform(Xs)
        assert get_pct_incom_samples(transformed_Xs) == round(amputer.p * 100)
        assert get_pct_com_samples(transformed_Xs) == round((1 - amputer.p) * 100)
        assert amputer.n_mods == len(Xs)

def test_invalid_params(sample_data):
    with pytest.raises(ValueError):
        Amputer(mechanism='Invalid mechanism')
    with pytest.raises(ValueError):
        Amputer(p=-1)

def test_fit(sample_data):
    p = 0.2
    amputer = Amputer(p=p, random_state=42)
    for Xs in sample_data:
        amputer.fit(Xs)
        assert amputer.n_mods == len(Xs)

def test_tranform(sample_data):
    p = 0.2
    amputer = Amputer(p=p, random_state=42)
    for Xs in sample_data:
        amputer.fit(Xs)
        transformed_Xs = amputer.transform(Xs)
        assert get_pct_incom_samples(transformed_Xs) == round(amputer.p * 100)
        assert get_pct_com_samples(transformed_Xs) == round((1 - amputer.p) * 100)
        assert amputer.n_mods == len(Xs)

def test_extreme_p(sample_data):
    for p in [0, 0.9]:
        for mechanism in ["mem", "mcar", "pm", "mnar"]:
            amputer = Amputer(p=p, random_state=42, mechanism=mechanism)
            for Xs in sample_data:
                Xs = Xs[:2]
                transformed_Xs = amputer.fit_transform(Xs)
                assert get_pct_incom_samples(transformed_Xs) == round(amputer.p*100)
                assert get_pct_com_samples(transformed_Xs) == round((1 - amputer.p)*100)
                assert amputer.n_mods == len(Xs)

def test_param_mechanism(sample_data):
    p = 0.2
    for mechanism in ["mem", "mcar", "pm", "mnar"]:
        amputer = Amputer(p=p, mechanism=mechanism, random_state=42)
        for Xs in sample_data:
            transformed_Xs = amputer.fit_transform(Xs)
            assert len(transformed_Xs) == len(Xs)
            assert get_pct_incom_samples(transformed_Xs) == round(amputer.p*100)
            assert get_pct_com_samples(transformed_Xs) == round((1 - amputer.p)*100)
            assert sum([np.isnan(transformed_X).sum().sum() > 0
                       for transformed_X, X in zip(transformed_Xs, Xs)])
            for transformed_X, X in zip(transformed_Xs, Xs):
                assert transformed_X.shape == X.shape
    X = np.random.default_rng(0).random((8, 5))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :2], X.iloc[:, 2:4], X.iloc[:, 4:]
    Amputer(p=p, mechanism="pm", random_state=0).fit_transform([X1, X2, X3])

if __name__ == "__main__":
    pytest.main()