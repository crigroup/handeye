#!/usr/bin/env python
import inspect
import unittest
import numpy as np
import baldor as br
from handeye import HandEyeCalibrator, solver
from handeye.benchmark import generate_noisy_samples


class TestHandEyeCalibrator(unittest.TestCase):
  def test_clean_synthetic_data(self):
    num_samples = 10
    # Get the list of available solvers
    solver_classes = []
    for name in dir(solver):
      obj = getattr(solver, name)
      if inspect.isclass(obj) and obj != solver.SolverBase:
        solver_classes.append(obj)
    calibrator = HandEyeCalibrator()
    # Ground-truth values
    X = tTo = br.transform.random(max_position=0.1)
    bTc = br.transform.random(max_position=1.2)
    # Samples without any noise
    samples = generate_noisy_samples(tTo, bTc, num_samples, 0, 0)
    for Q,P in samples:
      calibrator.assess_tcp_pose(Q)
      calibrator.add_sample(Q, P)
    for solver_class in solver_classes:
      print 'Testing: {0}'.format(solver_class)
      Xest = calibrator.solve(method=solver_class)
      # Given that we don't have any noise, Xest == X must be true
      np.testing.assert_allclose(Xest, X)
      # Reprojection error should be zero
      rot_rmse, trans_rmse = calibrator.compute_reprojection_error(Xest)
      np.testing.assert_allclose([rot_rmse,trans_rmse], [0,0], atol=1e-08)
    self.assertEqual(calibrator.get_num_samples(), num_samples)

  def test_noisy_synthetic_data(self):
    # Get the list of available solvers
    solver_classes = []
    for name in dir(solver):
      obj = getattr(solver, name)
      if inspect.isclass(obj) and obj != solver.SolverBase:
        solver_classes.append(obj)
    calibrator = HandEyeCalibrator()
    # Ground-truth values
    X = tTo = br.transform.random(max_position=0.1)
    bTc = br.transform.random(max_position=1.2)
    # Samples without any noise
    samples = generate_noisy_samples(tTo, bTc, 10)
    [calibrator.add_sample(Q, P) for Q,P in samples]
    for solver_class in solver_classes:
      print 'Testing: {0}'.format(solver_class)
      Xest = calibrator.solve(method=solver_class)
      # Given that we have noise, Xest != X
      self.assertFalse(br.transform.are_equal(Xest, X))
      # Reprojection error should be bigger than zero
      rot_rmse, trans_rmse = calibrator.compute_reprojection_error(Xest)
      self.assertTrue(rot_rmse > br._FLOAT_EPS)
      self.assertTrue(trans_rmse > br._FLOAT_EPS)

  def test_invalid_num_samples(self):
    calibrator = HandEyeCalibrator()
    for num_samples in range(calibrator.min_samples_required):
      calibrator.reset()
      # Ground-truth values
      X = tTo = br.transform.random(max_position=0.1)
      bTc = br.transform.random(max_position=1.2)
      # Samples without any noise
      samples = generate_noisy_samples(tTo, bTc, num_samples)
      [calibrator.add_sample(Q, P) for Q,P in samples]
      try:
        calibrator.solve()
        raises = False
      except Exception:
        raises = True
      self.assertTrue(raises)
