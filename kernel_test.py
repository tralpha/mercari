import unittest
import numpy as np
from kernel import Evaluator


class TestUM(unittest.TestCase):
  def setUp(self):
    pass

  def test_rmsle_same(self):
    self.assertEqual(
        Evaluator(np.array([1, 2, 3])).rmsle(np.array([1, 2, 3])), 0)

  def test_rmsle_diff(self):
    self.assertEqual(
        Evaluator(np.array([1.1, 2.2, 3])).rmsle(np.array([1, 2, 3.1])),
        0.048837915417030378)

  def test_rmsle_zeros(self):
    self.assertEqual(Evaluator(np.zeros(5)).rmsle(np.zeros(5)), 0.0)

  def test_rmsle_loop(self):
    self.assertEqual(
        Evaluator(np.array([1, 2, 3])).rmsle_loop(np.array([1, 2, 3])), 0)
    self.assertEqual(
        Evaluator(np.array([1, 2, 3])).rmsle_loop(np.array([1, 2, 3.1])),
        0.014256286526046115)
    self.assertEqual(Evaluator(np.zeros(5)).rmsle_loop(np.zeros(5)), 0.0)


if __name__ == '__main__':
  unittest.main()
