import unittest

from .learning_rate_schedule import *

class LearningRateScheduleTests(unittest.TestCase):

    def testConstant(self):
        schedule = LearningRateSchedule(0.1)
        self.assertEqual(schedule(0), 0.1)
        self.assertEqual(schedule(1), 0.1)
        self.assertEqual(schedule(20), 0.1)

    def testDecay(self):
        schedule = LearningRateSchedule(0.2, 0.1, [10, 20, 30])
        self.assertAlmostEqual(schedule(1), 0.2)
        self.assertAlmostEqual(schedule(9), 0.2)
        self.assertAlmostEqual(schedule(10), 0.02)
        self.assertAlmostEqual(schedule(19), 0.02)
        self.assertAlmostEqual(schedule(20), 0.002)
        self.assertAlmostEqual(schedule(29), 0.002)
        self.assertAlmostEqual(schedule(30), 0.0002)
        self.assertAlmostEqual(schedule(99), 0.0002)

    def testDecayUnsorted(self):
        schedule = LearningRateSchedule(0.2, 0.1, [30, 10, 20,])
        self.assertAlmostEqual(schedule(1), 0.2)
        self.assertAlmostEqual(schedule(9), 0.2)
        self.assertAlmostEqual(schedule(10), 0.02)
        self.assertAlmostEqual(schedule(19), 0.02)
        self.assertAlmostEqual(schedule(20), 0.002)
        self.assertAlmostEqual(schedule(29), 0.002)
        self.assertAlmostEqual(schedule(30), 0.0002)
        self.assertAlmostEqual(schedule(99), 0.0002)

if __name__ == "__main__":
    unittest.main()
