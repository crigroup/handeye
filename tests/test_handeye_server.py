#!/usr/bin/env python
import sys
import copy
import rospy
# Testing
import rostest
import unittest
# Utils
import numpy as np
import baldor as br
import criutils as cu
# HandEye Calibration service
from handeye.benchmark import generate_noisy_samples
from handeye.srv import CalibrateHandEye, CalibrateHandEyeRequest

NODENAME = 'test_handeye_server'


class TestHandEyeServer(unittest.TestCase):
  def __init__(self, *args):
    super(TestHandEyeServer, self).__init__(*args)
    rospy.init_node(NODENAME)
    self.srv = rospy.ServiceProxy('handeye_calibration', CalibrateHandEye)
    self.srv.wait_for_service(timeout=2.0)
    # Generate a valid calibration request
    self.X = bTc = br.transform.random(max_position=1.2)
    cTb = br.transform.inverse(bTc)
    eTo = br.transform.random(max_position=0.1)
    samples = []
    for _ in range(10):
      Q = bTe = br.transform.random()
      bTo = np.dot(bTe, eTo)
      Pinv = cTo = np.dot(cTb, bTo)
      samples.append((Q, Pinv))
    self.req = CalibrateHandEyeRequest()
    self.req.setup = 'Moving'
    self.req.solver = 'ParkBryan1994'
    # Populate the request
    self.req.effector_wrt_world.poses = [cu.conversions.to_pose(sample[0])
                                                          for sample in samples]
    self.req.object_wrt_sensor.poses = [cu.conversions.to_pose(sample[1])
                                                          for sample in samples]

  def test_num_samples(self):
    request = copy.deepcopy(self.req)
    # Inconsistent number of samples
    request.object_wrt_sensor.poses.pop()
    res = self.srv.call(request)
    self.assertFalse(res.success)

  def test_setup(self):
    request = copy.deepcopy(self.req)
    # Valid setups
    request.setup = 'Fixed'
    res = self.srv.call(request)
    self.assertTrue(res.success)
    request.setup = 'Moving'
    res = self.srv.call(request)
    self.assertTrue(res.success)
    # Valid weird-case setup
    request.setup = 'MoViNg'
    res = self.srv.call(request)
    self.assertTrue(res.success)
    # Invalid setup
    request.setup = 'some.invalid.setup'
    res = self.srv.call(request)
    self.assertFalse(res.success)

  def test_solver(self):
    request = copy.deepcopy(self.req)
    # Full path solver
    request.solver = 'handeye.solver.ParkBryan1994'
    res = self.srv.call(request)
    self.assertTrue(res.success)
    # Solver available at handeye.solver
    request.solver = 'TsaiLenz1989'
    res = self.srv.call(request)
    self.assertTrue(res.success)
    # Invalid solver
    request.solver = 'some.invalid.solver'
    res = self.srv.call(request)
    self.assertFalse(res.success)


if __name__ == '__main__':
  try:
    rostest.run('rostest', NODENAME, TestHandEyeServer, sys.argv)
  except KeyboardInterrupt:
    pass
