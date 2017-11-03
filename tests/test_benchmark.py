#!/usr/bin/env python
import inspect
import unittest
import numpy as np
import baldor as br
from handeye import benchmark


class TestModule(unittest.TestCase):
  def test_add_relative_noise(self):
    # No noise
    T0 = br.transform.random()
    T1 = benchmark.add_relative_noise(T0, 0, 0)
    np.testing.assert_allclose(T0, T1)
    # Rotation noise only
    T2 = br.transform.random()
    T3 = benchmark.add_relative_noise(T2, 0.123, 0)
    np.testing.assert_allclose(T2[:3,3], T3[:3,3])
    # Translation noise only
    T4 = br.transform.random()
    T5 = benchmark.add_relative_noise(T4, 0, 0.123)
    np.testing.assert_allclose(T5[:3,:3], T5[:3,:3])

  def test_compute_estimation_error(self):
    X = br.transform.random()
    angle = 0.123
    axis = br.vector.unit([1,2,3])
    Toffset = br.axis_angle.to_transform(axis, angle)
    X_hat = np.dot(X, Toffset)
    rot_error, trans_error = benchmark.compute_estimation_error(X, X_hat)
    np.testing.assert_allclose(rot_error, angle)
    np.testing.assert_allclose(trans_error, 0)

  def test_rmse(self):
    error = benchmark.rmse(range(10))
    np.testing.assert_allclose(error, 5.33853912602)
