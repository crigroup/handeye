#!/usr/bin/env python
import itertools
import numpy as np
import scipy.linalg
import baldor as br
from baldor.vector import skew


class SolverBase(object):
  """
  Base class for the solvers. It encodes the requirements and structure of
  Hand-Eye calibration solvers.
  """
  def __repr__(self):
    return self.__class__.__name__

  def __str__(self):
    return self.__repr__()

  def __call__(self, A, B):
    """
    A solver class must implement this method so that the class can be
    `callable`.

    Parameters
    ----------
    A: list
      List of homogeneous transformations with the relative motion of the
      end-effector
    B: list
      List of homogeneous transformations with the relative motion of the
      calibration pattern (often called `object`)

    Returns
    -------
    Xhat: array_like
      The estimate of :math:`X` (homogeneous transformation)
    """
    raise NotImplementedError('A solver must be callable')

  @staticmethod
  def estimate_translation(A, B, Rx):
    """
    Estimate the translation component of :math:`\hat{X}` in :math:`AX=XB`. This
    requires the estimation of the rotation component :math:`\hat{R}_x`

    Parameters
    ----------
    A: list
      List of homogeneous transformations with the relative motion of the
      end-effector
    B: list
      List of homogeneous transformations with the relative motion of the
      calibration pattern (often called `object`)
    Rx: array_like
      Estimate of the rotation component (rotation matrix) of :math:`\hat{X}`

    Returns
    -------
    tx: array_like
      The estimated translation component (XYZ value) of :math:`\hat{X}`
    """
    C = []
    d = []
    for Ai,Bi in itertools.izip(A, B):
      ta = Ai[:3,3]
      tb = Bi[:3,3]
      C.append(Ai[:3,:3]-np.eye(3))
      d.append(np.dot(Rx,tb)-ta)
    C = np.array(C)
    C.shape = (-1,3)
    d = np.array(d).flatten()
    tx, residuals, rank, s = np.linalg.lstsq(C, d, rcond=-1)
    return tx.flatten()


class Daniilidis1999(SolverBase):
  """
  Hand-Eye calibration solver that uses dual quaternions.

  Implementation based on: :cite:`Daniilidis1999`.
  """
  def __call__(self, A, B):
    # Step 1. Populate T using eq.(31) and eq.(33)
    S = np.zeros((6,8))
    T = np.zeros((6*len(A),8))
    for i,(Ai,Bi) in enumerate(zip(A, B)):
      qr_Ai, qt_Ai = br.transform.to_dual_quaternion(Ai)
      qr_Bi, qt_Bi = br.transform.to_dual_quaternion(Bi)
      a = qr_Ai[1:].reshape(3,1)
      b = qr_Bi[1:].reshape(3,1)
      a_ = qt_Ai[1:].reshape(3,1)
      b_ = qt_Bi[1:].reshape(3,1)
      S[:3,:] = np.hstack( (a-b, skew(a+b), np.zeros((3,4))) )
      S[3:,:] = np.hstack( (a_-b_, skew(a_+b_), a-b, skew(a+b)) )
      row = 6*i
      T[row:row+6,:] = S
    # Step 2. Compute the SVD of T
    U,S,V = np.linalg.svd(T)
    # Step 3. Compute the coefficients of eq.(35)
    v7 = V.T[:,6]
    v8 = V.T[:,7]
    u1 = v7[:4].reshape(4,1)
    v1 = v7[4:].reshape(4,1)
    u2 = v8[:4].reshape(4,1)
    v2 = v8[4:].reshape(4,1)
    coeff = np.zeros(3)
    coeff[0] = np.dot(u1.T, v1)
    coeff[1] = np.dot(u1.T, v2) + np.dot(u2.T, v1)
    coeff[2] = np.dot(u2.T, v2)
    s_list = np.roots(coeff)
    # Step 4. Choose the good s and compute lambdas
    vals = []
    for s in s_list:
      val = (s**2)*np.dot(u1.T,u1) + 2*s*np.dot(u1.T,u2) + np.dot(u2.T,u2)
      vals.append(float(val))
    idx = np.argmax(vals)
    max_val = vals[idx]
    s = s_list[idx]
    lbda2 = np.sqrt(1/max_val)
    lbda1 = s*lbda2
    # Step 5. Get the result and re-arrange for our notation
    sol = lbda1*v7 + lbda2*v8
    qr = sol[:4]
    qr /= np.linalg.norm(qr)            # Normalize just in case
    qt = sol[4:]
    X = br.quaternion.dual_to_transform(qr, qt)
    return X


class ParkBryan1994(SolverBase):
  """
  Hand-Eye calibration solver that uses axis-angle representation.

  Implementation based on: :cite:`Park1994`.
  """
  def __call__(self, A, B):
    M = np.zeros((3,3))
    for Ai,Bi in itertools.izip(A, B):
      # Transform the matrices to their axis-angle representation
      axis,angle,_ = br.transform.to_axis_angle(Ai)
      alpha = angle*axis
      axis,angle,_ = br.transform.to_axis_angle(Bi)
      beta = angle*axis
      # Compute M
      M += np.dot(beta.reshape(3,1), alpha.reshape(1,3))
    # Estimate Rx
    Rx = np.dot(np.linalg.inv(scipy.linalg.sqrtm(np.dot(M.T, M))), M.T)
    # Estimate tx
    tx = self.estimate_translation(A, B, Rx)
    # Return X
    X = np.eye(4)
    X[:3,:3] = Rx
    X[:3,3] = tx
    return X

class TsaiLenz1989(SolverBase):
  """
  Hand-Eye calibration solver that uses a modified version of the angle-axis
  representation.

  Implementation based on: :cite:`Tsai1989`.
  """
  def __call__(self, A, B):
    norm = np.linalg.norm
    C = []
    d = []
    for Ai,Bi in itertools.izip(A, B):
      # Transform the matrices to their axis-angle representation
      r_gij, theta_gij, _ = br.transform.to_axis_angle(Ai)
      r_cij, theta_cij, _ = br.transform.to_axis_angle(Bi)
      # Tsai uses a modified version of the angle-axis representation
      Pgij = 2*np.sin(theta_gij/2.)*r_gij
      Pcij = 2*np.sin(theta_cij/2.)*r_cij
      # Use C and d to avoid overlapping with the input A-B
      C.append(skew(Pgij+Pcij))
      d.append(Pcij-Pgij)
    # Estimate Rx
    C = np.array(C)
    C.shape = (-1,3)
    d = np.array(d).flatten()
    Pcg_, residuals, rank, s = np.linalg.lstsq(C, d, rcond=-1)
    Pcg = 2*Pcg_ / np.sqrt(1 + norm(Pcg_)**2)
    R1 = (1 - norm(Pcg)**2/2.) * np.eye(3)
    R2 = (np.dot(Pcg.reshape(3,1),Pcg.reshape(1,3)) +
                        np.sqrt(4-norm(Pcg)**2) * skew(Pcg)) / 2.
    Rx = R1 + R2
    # Estimate tx
    tx = self.estimate_translation(A, B, Rx)
    # Return X
    X = np.eye(4)
    X[:3,:3] = Rx
    X[:3,3] = tx
    return X
