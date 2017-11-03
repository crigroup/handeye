#!/usr/bin/env python
import inspect
import unittest
import numpy as np
import baldor as br
from handeye import solver


class TestModule(unittest.TestCase):
  def test_SolverBase(self):
    # Test class name
    base = solver.SolverBase()
    self.assertEqual(str(base), 'SolverBase')
    # Test NotImplemented
    try:
      base(None, None)
      raises = False
    except NotImplementedError:
      raises = True
    self.assertTrue(raises)
