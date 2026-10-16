import pytest
import numpy as np
from imml.preprocessing import NormalizerNaN


@pytest.fixture
def sample_data():
    X = np.array([
        [1, 2, np.nan],
        [4, 5, 6],
        [7, np.nan, 9],
    ])
    return X


def test_normalizer_nan_l1(sample_data):
    transformer = NormalizerNaN(norm='l1')
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)

    expected_transformed_X = np.array([
        [0.33333333, 0.66666667, np.nan],
        [0.26666667, 0.33333333, 0.4],
        [0.4375, np.nan, 0.5625],
    ])
    assert np.allclose(transformed_X, expected_transformed_X, equal_nan=True)

def test_normalizer_nan_l2(sample_data):
    transformer = NormalizerNaN(norm='l2')
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)

    expected_transformed_X = np.array([
        [0.44721, 0.89443, np.nan],
        [0.45584, 0.56980, 0.68376],
        [0.61394, np.nan, 0.78935],
    ])

    assert np.allclose(transformed_X, expected_transformed_X, equal_nan=True)


def test_normalizer_nan_max(sample_data):
    transformer = NormalizerNaN(norm='max')
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)

    expected_transformed_X = np.array([
        [0.5, 1.0, np.nan],
        [0.66666667, 0.83333333, 1.0],
        [0.77777778, np.nan, 1.0],
    ])

    assert np.allclose(transformed_X, expected_transformed_X, equal_nan=True)


def test_invalid_norm():
    with pytest.raises(ValueError):
        NormalizerNaN(norm='invalid')


def test_transform(sample_data):
    transformer = NormalizerNaN(norm='l2')
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)
    assert transformed_X.shape == sample_data.shape


@pytest.mark.parametrize("norm, expected_shape", [
    ('l1', (3, 3)),
    ('l2', (3, 3)),
    ('max', (3, 3)),
])
def test_parameterized_norms(norm, expected_shape, sample_data):
    transformer = NormalizerNaN(norm=norm)
    transformer.fit(sample_data)
    transformed_X = transformer.transform(sample_data)

    assert transformed_X.shape == expected_shape

def test_nan_handling():
    X = np.array([
        [np.nan, 2],
        [3, np.nan],
        [np.nan, np.nan]
    ])

    transformer = NormalizerNaN(norm='l1')
    transformer.fit(X)
    transformed_X = transformer.transform(X)

    expected_transformed_X = np.array([
        [np.nan, 1.0],
        [1.0, np.nan],
        [np.nan, np.nan]
    ])

    assert np.allclose(transformed_X, expected_transformed_X, equal_nan=True)


if __name__ == "__main__":
    pytest.main()