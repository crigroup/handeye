#!/usr/bin/env python
import numpy as np
import baldor as br


def add_relative_noise(T, rot_noise, trans_noise, eps=1e-7):
  """
  Add relative noise to a homogeneous transformation

  Parameters
  ----------
  T: array_like
    The input homogeneous transformation
  rot_noise: float
    Magnitude of the `white` noise to be apply to the rotation component
  trans_noise: float
    Magnitude of the `white` noise to be apply to the translation component
  eps: float, optional
    Epsilon for a magnitude to be considered bigger than zero

  Returns
  -------
  Tnoisy: array_like
    The `noisy` homogeneous transformation
  """
  Tnoisy = np.array(T)
  # Add rotation noise
  if rot_noise > eps:
    rot_sigma = rot_noise*br.transform.to_axis_angle(T)[1]
    twist = np.random.uniform(size=3, low=-rot_sigma/2., high=rot_sigma/2)
    angle = br.vector.norm(twist)
    axis = br.vector.unit(twist)
    Radd = br.axis_angle.to_transform(axis, angle)[:3,:3]
    Tnoisy[:3,:3] = np.dot(Radd, T[:3,:3])
  # Add translation noise
  if trans_noise > eps:
    trans_sigma = np.linalg.norm(T[:3,3])*trans_noise
    Tnoisy[:3,3] += np.random.uniform(size=3, low=-trans_sigma/2.,
                                                    high=trans_sigma/2.)
  return Tnoisy

def compute_estimation_error(X, X_hat):
  """
  Compute the estimation error between two homogeneous transformations

  Parameters
  ----------
  X: array_like
    The first homogeneous transformation (ground truth)
  Xhat: array_like
    The second homogeneous transformation (estimated)

  Returns
  -------
  rot_error: float
    The error of the rotation component
  trans_error: float
    The error of the translation component
  """
  Rerror = np.eye(4)
  Rerror[:3,:3] = np.dot(X[:3,:3].T, X_hat[:3,:3])
  rot_error = abs(br.transform.to_axis_angle(Rerror)[1])
  trans_error = np.linalg.norm(X[:3,3]-X_hat[:3,3])
  return rot_error, trans_error

def generate_noisy_samples(tTo, bTc, num_samples, rot_noise=0.1e-2,
                                                            trans_noise=1e-2):
  """
  Generate noisy samples.

  A sample is a tuple of `Q` and `P` homogeneous transformations. `Q` is the
  transform of the end-effector, `P` is the transform of the calibration
  pattern.

  Parameters
  ----------
  tTo: array_like
    Homogeneous transformation of the calibration pattern (`o`) with respect to
    the end-effector (`t`)
  bTc: array_like
    Homogeneous transformation of the sensor (`c`) with respect to robot's
    origin (`b`)
  num_samples: int
    Number of samples to be generated
  rot_noise: float
    Magnitude of the `white` noise to be apply to the rotation component
  trans_noise: float
    Magnitude of the `white` noise to be apply to the translation component

  Returns
  -------
  samples: list
    List of noisy samples (tuple of `Q` and `P`)
  """
  samples = []
  cTb = br.transform.inverse(bTc)  # origin (b) wrt camera (c)
  for _ in range(num_samples):
    bTt = br.transform.random()   # gripper (t) wrt origin (b)
    bTo = np.dot(bTt, tTo)        # pattern (o) wrt origin (b)
    cTo = np.dot(cTb, bTo)        # pattern (o) wrt camera (c)
    Qnoisy = add_relative_noise(bTt, rot_noise, trans_noise)
    Pnoisy = add_relative_noise(cTo, rot_noise, trans_noise)
    samples.append((Qnoisy, Pnoisy))
  return samples

def rmse(a, axis=None):
  """
  Compute the RMS error along the specified axis.

  Parameters
  ----------
  a: array_like
    Array containing numbers whose RMSE is desired. If `a` is not an
    array, a conversion is attempted.
  axis : None or int or tuple of ints, optional
    Axis or axes along which the means are computed. The default is to
    compute the mean of the flattened array.

  Returns
  -------
  error: float
    The RMS error along the specified axis.
  """
  return np.sqrt(np.mean(np.square(a), axis=axis))
