#!/usr/bin/env python
import enum
import itertools
import numpy as np
import baldor as br
# Local modules
from . import benchmark, solver


class HandEyeCalibrator(object):
  """
  This class is the entry point for a Hand-Eye calibration session.

  It provides the methods to collect the samples to populate the matrices
  :math:`A` and :math:`B` so that :math:`X` can be estimated in the calibration
  problem :math:`AX = XB`.
  With this class you can calibrate the position of a sensor (e.g. camera, laser, etc.) with respect to a robot.

  See Also
  --------
  :class:`Setup`

  Notes
  -----
  This implementation uses the same notation as :cite:`Shi2005`.
  """
  def __init__(self, setup=None):
    """
    Parameters
    ----------
    setup: :class:`handeye.calibrator.Setup`
      Fixed setup or moving sensor

    Raises
    ------
    KeyError
      If `setup` is an invalid keyword (str)
    ValueError
      If `setup` is an invalid id (int)
    """
    if setup is None:
      self.setup = Setup.Moving
    elif type(setup) == Setup:
      self.setup = setup
    elif type(setup) == str:
      try:
        setup = setup.capitalize()
        self.setup = Setup[setup]
      except KeyError:
        raise KeyError('Invalid setup: {}'.format(setup))
    elif type(setup) == int:
      self.setup = Setup(setup)
    else:
      raise TypeError('Invalid setup type: {}'.format(type(setup)))
    self.reset()
    self.min_samples_required = 4
    self.valid_cos = lambda (x): np.clip(x, -1., 1.)

  def add_sample(self, Q, Pinv):
    """
    Add one sample to the `HandEyeCalibrator` instance

    Parameters
    ----------
    Q: array_like
      Homogeneous transformation of the end-effector w.r.t. the robot base
    Pinv: array_like
      Homogeneous transformation of the calibration pattern (often called
      `object`) w.r.t. the camera frame
    """
    self.Q.append(Q)
    self.Qinv.append(br.transform.inverse(Q))
    self.P.append(br.transform.inverse(Pinv))
    self.Pinv.append(Pinv)
    self.num_samples = len(self.Q)
    return self.num_samples

  def assess_tcp_pose(self, Q, alpha=0.5, beta=0.5, d=float('inf')):
    """
    Assess the golden rules presented in :cite:`Shi2005`.

    Parameters
    ----------
    Q: array_like
      Homogeneous transformation of the end-effector w.r.t. the robot base
    alpha: float, optional
      Minimum threshold for the angle between to consecutive motions. That is
      the angle :math:`<(\pmb{k}_{a,i}, \pmb{k}_{a,i+1})`
    beta: float, optional
      Minimal threshold for every :math:`\psi_{i}` that is the rotation angle
      of each relative motion :math:`\pmb{A}_{i}` and :math:`\pmb{B}_{i}`
    d: float, optional
      Maximum trheshold for the norm of the translation :math:`\pmb{t}_{a,i}`

    Returns
    -------
    is_pose_ok: bool
      `True` if the pose `Q` complies with the golden rules. `False` otherwise.

    Notes
    -----
    As a rotation matrix :math:`\pmb{R}` can be expressed as a rotation around
    a rotation axis :math:`\pmb{k}` by an angle :math:`\psi`, the relations
    between :math:`\psi`, :math:`\pmb{k}` and :math:`\pmb{R}` are given by
    Rodrigues theorem. Moreover, :math:`\pmb{R}_a` and :math:`\pmb{R}_b` have
    the same angle of rotation. We can rewrite :math:`\pmb{R}_a` and
    :math:`\pmb{R}_b` as :math:`\mbox{Rot}(\pmb{k}_a, \psi)` and
    :math:`\mbox{Rot}(\pmb{k}_b, \psi)` respectively.
    """
    metrics = self.compute_golden_rules_metrics(Q)
    num_metrics = len(metrics)
    if num_metrics == 0:
      is_pose_ok = True
    elif num_metrics == 2:
      theta_Ap,t_Ap = metrics
      is_pose_ok = (theta_Ap >= beta) and (t_Ap <= d)
    elif num_metrics == 3:
      angle_Ap_App,theta_App,t_App = metrics
      is_pose_ok = ((angle_Ap_App >= alpha) and (theta_App >= beta)
                                                      and (t_App <= d))
    return is_pose_ok

  def compute_golden_rules_metrics(self, Q=None, Ap=None, App=None):
    """
    Compute the golden rules as presented in :cite:`Shi2005`.

    Parameters
    ----------
    Q: array_like, optional
      Homogeneous transformation of the end-effector w.r.t. the robot base
    Ap: array_like, optional
      The penultimate motion homogeneous transformation :math:`\pmb{A}_{n-2}`
    App: array_like, optional
      The last motion motion homogeneous transformation :math:`\pmb{A}_{n-1}`
    """
    if Q is not None:
      if self.num_samples == 1:
        if self.setup == Setup.Moving:
          Ap = np.dot(self.Qinv[0], Q)
        elif self.setup == Setup.Fixed:
          Ap = np.dot(self.Q[0], br.transform.inverse(Q))
      elif self.num_samples >= 2:
        if self.setup == Setup.Moving:
          Ap = np.dot(self.Qinv[-2], self.Q[-1])
          App = np.dot(self.Qinv[-1], Q)
        elif self.setup == Setup.Fixed:
          Ap = np.dot(self.Q[-2], self.Qinv[-1])
          App = np.dot(self.Q[-1], br.transform.inverse(Q))
    res = tuple()
    if (Ap is not None) and (App is None):
      theta_Ap = abs(br.transform.to_axis_angle(Ap)[1])
      t_Ap = np.linalg.norm(Ap[:3,3])
      res = (theta_Ap, t_Ap)
    elif (Ap is not None) and (App is not None):
      theta_App = abs(br.transform.to_axis_angle(App)[1])
      t_App = np.linalg.norm(Ap[:3,3])
      k_Ap = br.transform.to_axis_angle(Ap)[0]
      k_App = br.transform.to_axis_angle(App)[0]
      angle_Ap_App = np.arccos(self.valid_cos(np.dot(k_Ap, k_App)))
      res = (angle_Ap_App, theta_App, t_App)
    return res

  def compute_motion_matrices(self):
    """
    Compute the (relative) motion matrices :math:`A` and :math:`B`

    Returns
    -------
    A: list
      List of homogeneous transformations with the relative motion of the
      end-effector
    B: list
      List of homogeneous transformations with the relative motion of the
      calibration pattern (often called `object`)
    """
    A = []
    B = []
    for i in range(self.num_samples-1):
      if self.setup == Setup.Moving:
        A.append( np.dot(self.Qinv[i], self.Q[i+1]) )
      elif self.setup == Setup.Fixed:
        A.append( np.dot(self.Q[i], self.Qinv[i+1]) )
      # B is the same for both setups
      B.append( np.dot(self.Pinv[i], self.P[i+1]) )
    return A, B

  def compute_reprojection_error(self, X):
    """
    Compute the reprojection error given an estimate for :math:`X`.

    Parameters
    ----------
    X: array_like
      The estimate of :math:`X` (homogeneous transformation)

    Returns
    -------
    rotation_rmse: float
      The reprojection error of the rotation component
    translation_rmse: float
      The reprojection error of the translation component

    Notes
    -----
    Here we use the rotation and translation metrics proposed in Section
    III-A and III-B of :cite:`Strobl2006`. The reprojection error is the RMS
    error of these metrics.
    """
    rot_errors = []
    trans_errors = []
    A,B = self.compute_motion_matrices()
    for Ai, Bi in itertools.izip(A, B):
      lT = np.dot(Ai, X)
      rT = np.dot(X, Bi)
      # Metric for rotation error
      Rerror = np.eye(4)
      Rerror[:3,:3] = np.dot(lT[:3,:3].T, rT[:3,:3])
      rot_errors.append( br.transform.to_axis_angle(Rerror)[1] )
      # Metric for translation error
      tt_error = lT[:3,3] - rT[:3,3]
      bt_error = (br.transform.inverse(lT)[:3,3] -
                                                br.transform.inverse(rT)[:3,3])
      terror = (np.linalg.norm(tt_error) + np.linalg.norm(bt_error)) / 2.
      trans_errors.append(terror)
    rotation_rmse = benchmark.rmse(rot_errors)
    translation_rmse = benchmark.rmse(trans_errors)
    return (rotation_rmse, translation_rmse)

  def get_num_samples(self):
    """
    Return the number of samples collected

    Returns
    -------
    num_samples: int
      The number of samples
    """
    return self.num_samples

  def reset(self):
    """
    Reset/initialize the `HandEyeCalibrator` instance
    """
    self.Q = []
    self.Qinv = []
    self.P = []
    self.Pinv = []
    self.num_samples = 0

  def solve(self, method=solver.ParkBryan1994):
    """
    Solve the Hand-Eye calibration using the collected samples.

    Returns
    -------
    Xhat: array_like
      The estimate of :math:`X` (homogeneous transformation)
    """
    if self.num_samples < self.min_samples_required:
      raise Exception('Not enough samples: {}'.format(self.num_samples))
    A,B = self.compute_motion_matrices()
    method_instance = method()
    Xhat = method_instance(A, B)
    return Xhat


class Setup(enum.Enum):
  """
  The :class:`HandEyeCalibrator` currently supports two types of setups: either
  your sensor is fixed with respect to the robot origin, or your sensor is
  mounted on the robot hand and is moving with the robot.
  """

  Fixed = 1
  """
  To calibrate a fixed sensor you need to grab a calibration pattern with the
  robot, move the pattern with the robot and observe the pattern in these
  different poses with the sensor. The exact position of the pattern on the
  robot hand does not need to be known, it will be calculated during
  calibration, along with the sensor's pose relative to the robot origin.
  """

  Moving = 2
  """
  Similarly you can calibrate a moving sensor on the robot hand. For this setup
  a calibration plate needs to be placed inside the robot's workspace, so that
  the sensor can observe the calibration plate. Then you move the robot around
  and observe the pattern from different positions and angles.
  Again, the position of the calibration pattern does not need to be known and
  will be calculated during calibration, together with the sensor's mounting
  position with respect to the robot hand.
  """
