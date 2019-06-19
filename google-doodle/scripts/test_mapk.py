from average_precision import mapk
import numpy as np
import pytest

@pytest.fixture
def labels():

    return np.array([[1],[2],[3],[3]])

@pytest.fixture
def preds():
    return np.array([[0.1, 0.9, 0.2, 0.3],
                  [0.1 , 0.2, 0.7, 0.4],
                  [0.7 , 0.2, 0.7, 0.9],
                  [0.1 , 0.8, 0.7, 0.9]])

def test_mapk(labels, preds):

    x = mapk(labels, preds, k =3)
    assert x == 1
