"""Test the Surrogate abstract base class and its implementations."""

import numpy as np
import pytest
from soogo.model import Surrogate
from soogo.model.rbf import RbfModel
from soogo.model.gp import GaussianProcess


def test_surrogate_is_abstract():
    """Test that Surrogate is an abstract class and cannot be instantiated."""
    with pytest.raises(TypeError):
        Surrogate()


def test_rbf_model_is_surrogate():
    """Test that RbfModel is a subclass of Surrogate."""
    rbf = RbfModel()
    assert isinstance(rbf, Surrogate)


def test_gaussian_process_is_surrogate():
    """Test that GaussianProcess is a subclass of Surrogate."""
    gp = GaussianProcess()
    assert isinstance(gp, Surrogate)


def test_surrogate_interface_methods():
    """Test that both surrogate implementations have the required methods."""
    # Test RBF Model
    rbf = RbfModel()

    # Create some sample data
    x_train = np.array([[0, 0], [1, 1], [2, 0]])
    y_train = np.array([0, 2, 1])

    # Test all required methods exist and work
    rbf.update(x_train, y_train)
    assert rbf.ntrain == 3
    assert rbf.X.shape == (3, 2)
    assert rbf.Y.shape == (3,)
    assert rbf.min_design_space_size(2) > 0
    assert rbf.check_initial_design(x_train)
    assert isinstance(rbf.iindex, tuple)

    # Test prediction
    x_test = np.array([[0.5, 0.5]])
    y_pred, distances = rbf(x_test, return_dist=True)
    assert y_pred.shape == (1,)
    assert distances.shape == (1, 3)

    # Test Gaussian Process
    gp = GaussianProcess()

    # Test all required methods exist and work
    gp.update(x_train, y_train)
    assert gp.ntrain == 3
    assert gp.X.shape == (3, 2)
    assert gp.Y.shape == (3,)
    assert gp.min_design_space_size(2) >= 0
    assert gp.check_initial_design(x_train)
    assert isinstance(gp.iindex, tuple)

    # Test prediction
    y_pred, y_std = gp(x_test, return_std=True)
    assert y_pred.shape == (1,)
    assert y_std.shape == (1,)


if __name__ == "__main__":
    test_surrogate_is_abstract()
    test_rbf_model_is_surrogate()
    test_gaussian_process_is_surrogate()
    test_surrogate_interface_methods()
    print("All tests passed!")
