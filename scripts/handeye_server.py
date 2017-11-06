#!/usr/bin/env python
import os
import rospy
import argparse
# Utils
import criutils as cu
# OO Inspection
import inspect
import importlib
# HandEye
import handeye
from handeye import HandEyeCalibrator
from handeye.srv import CalibrateHandEye, CalibrateHandEyeResponse


def calibration_cb(req):
  response = CalibrateHandEyeResponse()
  solver_class = get_class_from_str(req.solver)
  # Check the given solver can be imported
  if solver_class is None:
    rospy.logwarn('Failed to find HandEye solver: {}'.format(req.solver))
    return response
  # Check the given solver inherits from handeye.solver.SolverBase
  if handeye.solver.SolverBase not in inspect.getmro(solver_class):
    rospy.logwarn('The HandEye solver {} does not inherit from' +
                                'handeye.solver.SolverBase'.format(req.solver))
    return response
  # Check the number of poses are consistent
  num_object_poses = len(req.object_wrt_sensor.poses)
  num_effector_poses = len(req.effector_wrt_world.poses)
  if num_object_poses != num_effector_poses:
    rospy.logwarn('Number of poses must be equal: ' +
                      '{0} != {1}'.format(num_effector_poses, num_object_poses))
    return response
  # Check a valid setup has been specified
  try:
    setup = handeye.calibrator.Setup[req.setup.capitalize()]
  except KeyError:
    rospy.logwarn('Invalid req.setup was specified: {}'.format(req.setup))
    return response
  # Run the HandEye calibration
  try:
    calibrator = HandEyeCalibrator(setup)
  except NotImplementedError:
    rospy.logwarn('Only Moving setup is supported')
    return response
  for i in xrange(num_object_poses):
    Q = cu.conversions.from_pose(req.effector_wrt_world.poses[i])
    P = cu.conversions.from_pose(req.object_wrt_sensor.poses[i])
    calibrator.add_sample(Q, P)
  Xhat = calibrator.solve(method=solver_class)
  # Populate the response
  response.success = True
  rotation_rmse, translation_rmse = calibrator.compute_reprojection_error(Xhat)
  response.rotation_rmse = rotation_rmse
  response.translation_rmse = translation_rmse
  response.sensor_frame = cu.conversions.to_pose(Xhat)
  return response

def get_class_from_str(class_str):
  split = class_str.rsplit('.', 1)
  if len(split) == 1:
    module = handeye.solver
    classname = split[0]
  else:
    try:
       module = importlib.import_module(split[0])
       classname = split[1]
    except ImportError:
      return None
  if not hasattr(module, classname):
    return None
  return getattr(module, classname)


def parse_args():
  # Remove extra IPython notebook args
  clean_argv = rospy.myargv()[1:]
  if '-f' in clean_argv:
    clean_argv = clean_argv[2:]
  # Parse
  format_class = argparse.RawDescriptionHelpFormatter
  parser = argparse.ArgumentParser(formatter_class=format_class,
                  description='HandEye calibration server')
  parser.add_argument('--debug', action='store_true',
    help='If set, will show debugging messages')
  args = parser.parse_args(clean_argv)
  return args

if __name__ == '__main__':
  args = parse_args()
  log_level= rospy.DEBUG if args.debug else rospy.INFO
  node_name = os.path.splitext(os.path.basename(__file__))[0]
  rospy.init_node(node_name, log_level=log_level)
  # Advertise service
  handeye_srv = rospy.Service('handeye_calibration', CalibrateHandEye,
                                                                calibration_cb)
  if args.debug:
    import IPython
    IPython.embed(banner1='')
    exit()
  else:
    srv_name = handeye_srv.resolved_name
    rospy.loginfo('HandEye server is up and running: {}'.format(srv_name))
    rospy.spin()
