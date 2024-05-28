from ml.enviroment import ml_model
import unittest

class test_enviroment(unittest.TestCase):
    def test_reward(self):
        reward = ml_model.reward(self, 0.5, 0.6, False)
        self.assertEqual(reward, 0)

        reward = ml_model.reward(self, 20, 1, True)
        self.assertAlmostEqual(reward, 0.7375)

        reward = ml_model.reward(self, 2, 1, True)
        self.assertAlmostEqual(reward, 0.5125)

        reward = ml_model.reward(self, 1, 2, True)
        self.assertAlmostEqual(reward, 0.0975)

        reward = ml_model.reward(self, 2, 1, True)
        self.assertAlmostEqual(reward, 0.5125)