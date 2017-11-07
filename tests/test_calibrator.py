#!/usr/bin/env python
import inspect
import unittest
import numpy as np
import baldor as br
from handeye import HandEyeCalibrator, Setup, solver
from handeye.benchmark import add_relative_noise


class TestHandEyeCalibrator(unittest.TestCase):
  def __init__(self, *args):
    super(TestHandEyeCalibrator, self).__init__(*args)
    # Get the list of all available solvers
    self.solver_classes = []
    for name in dir(solver):
      obj = getattr(solver, name)
      if inspect.isclass(obj) and obj != solver.SolverBase:
        self.solver_classes.append(obj)
    # Generate synthetic data
    self.num_samples = 10
    # Fixed setup
    self.Xf = bTc = br.transform.random(max_position=1.2)
    cTb = br.transform.inverse(bTc)
    eTo = br.transform.random(max_position=0.1)
    self.fixed_setup_samples = []
    for _ in range(self.num_samples):
      Q = bTe = br.transform.random()
      bTo = np.dot(bTe, eTo)
      Pinv = cTo = np.dot(cTb, bTo)
      self.fixed_setup_samples.append((Q, Pinv))
    # Moving setup
    self.Xm = eTc = br.transform.random(max_position=0.1)
    oTb = br.transform.random(max_position=1.2)
    self.moving_setup_samples = []
    for _ in range(self.num_samples):
      Q = bTe = br.transform.random()
      bTc = np.dot(bTe, eTc)
      P = oTc = np.dot(oTb, bTc)
      self.moving_setup_samples.append((Q, br.transform.inverse(P)))

  def test_noisy_synthetic_data(self):
    calibrator = HandEyeCalibrator(setup=Setup.Fixed)
    # Add noise to the samples
    for Q,Pinv in self.fixed_setup_samples:
      Qnoisy = add_relative_noise(Q, rot_noise=0.1e-2, trans_noise=1e-2)
      Pinv_noisy = add_relative_noise(Pinv, rot_noise=0.1e-2, trans_noise=1e-2)
      calibrator.add_sample(Qnoisy, Pinv_noisy)
    for solver_class in self.solver_classes:
      print 'Testing: {0}'.format(solver_class)
      Xest = calibrator.solve(method=solver_class)
      # Given that we have noise, Xest != X
      self.assertFalse(br.transform.are_equal(Xest, self.Xf))
      # Reprojection error should be bigger than zero
      rot_rmse, trans_rmse = calibrator.compute_reprojection_error(Xest)
      self.assertTrue(rot_rmse > br._FLOAT_EPS)
      self.assertTrue(trans_rmse > br._FLOAT_EPS)

  def test_num_samples(self):
    # Test invalid number
    calibrator = HandEyeCalibrator(Setup.Fixed)
    for num_samples in range(calibrator.min_samples_required):
      calibrator.reset()
      # Ground-truth values
      X = tTo = br.transform.random(max_position=0.1)
      bTc = br.transform.random(max_position=1.2)
      # Samples without any noise
      samples = self.fixed_setup_samples[:num_samples]
      [calibrator.add_sample(Q, P) for Q,P in samples]
      try:
        calibrator.solve()
        raises = False
      except Exception:
        raises = True
      self.assertTrue(raises)

  def test_setup(self):
    # Setup specified with strings
    HandEyeCalibrator('Fixed')
    HandEyeCalibrator('FiXed')
    HandEyeCalibrator('fixed')
    HandEyeCalibrator('Moving')
    HandEyeCalibrator('moving')
    # Invalid setup string
    try:
      HandEyeCalibrator('invalid.setup.str')
      raises = False
    except KeyError:
      raises = True
    self.assertTrue(raises)
    # Setup specified with int
    HandEyeCalibrator(setup=1)
    HandEyeCalibrator(setup=2)
    # Invalid setup int
    try:
      HandEyeCalibrator(setup=666)
      raises = False
    except ValueError:
      raises = True
    self.assertTrue(raises)

  def test_setup_fixed(self):
    calibrator = HandEyeCalibrator(setup=Setup.Fixed)
    for Q,Pinv in self.fixed_setup_samples:
      calibrator.assess_tcp_pose(Q)
      calibrator.add_sample(Q, Pinv)
    for solver_class in self.solver_classes:
      print 'Testing fixed setup with solver: {0}'.format(solver_class)
      Xest = calibrator.solve(method=solver_class)
      # Given that we don't have any noise, Xest == self.Xf must be true
      np.testing.assert_allclose(Xest, self.Xf)
      # Reprojection error should be zero
      rot_rmse, trans_rmse = calibrator.compute_reprojection_error(Xest)
      np.testing.assert_allclose([rot_rmse,trans_rmse], [0,0], atol=1e-08)
    self.assertEqual(calibrator.get_num_samples(), self.num_samples)

  def test_setup_moving(self):
    calibrator = HandEyeCalibrator(setup=Setup.Moving)
    for Q,Pinv in self.moving_setup_samples:
      calibrator.assess_tcp_pose(Q)
      calibrator.add_sample(Q, Pinv)
    for solver_class in self.solver_classes:
      print 'Testing fixed setup with solver: {0}'.format(solver_class)
      Xest = calibrator.solve(method=solver_class)
      # Given that we don't have any noise, Xest == self.Xm must be true
      np.testing.assert_allclose(Xest, self.Xm)
      # Reprojection error should be zero
      rot_rmse, trans_rmse = calibrator.compute_reprojection_error(Xest)
      np.testing.assert_allclose([rot_rmse,trans_rmse], [0,0], atol=1e-08)
    self.assertEqual(calibrator.get_num_samples(), self.num_samples)
