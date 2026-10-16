import numpy as np
import pytest
import pandas as pd
from imml.preprocessing import (
    DropMod, ConcatenateMods, SingleMod, AddMissingMods, SortData,
    concatenate_mods, drop_mod, single_mod, add_missing_mods, sort_data
)
from imml.explore import get_samples

@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    return Xs_pandas, Xs_numpy

def test_drop_mod_transformer(sample_data):
    idx = 1
    transformer = DropMod(X_idx=idx)
    for Xs in sample_data:
        transformed_Xs = transformer.fit_transform(Xs)
        assert len(transformed_Xs) == (len(Xs) - 1)
        for X,transformed_X in zip(Xs[:idx] + Xs[idx+1:], transformed_Xs[:idx] + transformed_Xs[idx+1:]):
            assert np.equal(X.shape, transformed_X.shape).all()

def test_concatenate_mods_transformer(sample_data):
    transformer = ConcatenateMods()
    for Xs in sample_data:
        transformed_Xs = transformer.fit_transform(Xs)
        assert transformed_Xs.shape == (len(Xs[0]), sum([X.shape[1] for X in Xs]))

def test_single_mod_transformer(sample_data):
    idx = 1
    transformer = SingleMod(X_idx=idx)
    for Xs in sample_data:
        transformed_X = transformer.fit_transform(Xs)
        assert np.equal(transformed_X, Xs[idx]).all().all()

def test_add_missing_mods_transformer(sample_data):
    for Xs in sample_data:
        samples = get_samples(Xs=Xs)
        transformer = AddMissingMods(samples=samples)
        transformed_Xs = transformer.fit_transform(Xs)
        for transformed_X in transformed_Xs:
            assert len(transformed_X) == len(samples)

def test_sort_data_transformer(sample_data):
    transformer = SortData()
    for Xs in sample_data:
        transformed_Xs = transformer.fit_transform(Xs)
        for transformed_X in transformed_Xs:
            if isinstance(transformed_X, pd.DataFrame):
                assert all(transformed_X.index == list(range(len(Xs[0]))))

def test_concatenate_mods_function(sample_data):
    for Xs in sample_data:
        transformed_Xs = concatenate_mods(Xs)
        assert transformed_Xs.shape == (len(Xs[0]), sum([X.shape[1] for X in Xs]))

def test_drop_mod_function(sample_data):
    idx = 1
    for Xs in sample_data:
        transformed_Xs = drop_mod(Xs, X_idx=idx)
        assert len(transformed_Xs) == (len(Xs) - 1)
        for X,transformed_X in zip(Xs[:idx] + Xs[idx+1:], transformed_Xs[:idx] + transformed_Xs[idx+1:]):
            assert np.equal(X.shape, transformed_X.shape).all()

def test_single_mod_function(sample_data):
    idx = 1
    for Xs in sample_data:
        transformed_X = single_mod(Xs, X_idx=idx)
        assert np.equal(transformed_X, Xs[idx]).all().all()

def test_add_missing_mods_function(sample_data):
    for Xs in sample_data:
        samples = get_samples(Xs=Xs)
        transformed_Xs = add_missing_mods(Xs, samples=samples)
        for transformed_X in transformed_Xs:
            assert len(transformed_X) == len(samples)

def test_sort_data_function(sample_data):
    for Xs in sample_data:
        transformed_Xs = sort_data(Xs)
        for transformed_X in transformed_Xs:
            if isinstance(transformed_X, pd.DataFrame):
                assert all(transformed_X.index == list(range(len(Xs[0]))))

def test_invalid_drop_mod_index(sample_data):
    with pytest.raises(ValueError, match="X_idx out of range. Should be between 0 and n_mods - 1"):
        drop_mod(sample_data[0], X_idx=10)

def test_invalid_single_mod_index(sample_data):
    with pytest.raises(ValueError, match="X_idx out of range. Should be between 0 and n_mods - 1"):
        single_mod(sample_data[0], X_idx=10)

if __name__ == "__main__":
    pytest.main()