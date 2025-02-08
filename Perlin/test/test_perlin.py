import unittest

from setup import *
from Perlin.perlin import Perlin


class Test_Perlin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.perlin = Perlin()

    def test_reproducibility(self):
        noise1 = list(self.perlin.noise(length=10000, step=0.01, octave=1, count=1, seed=5))
        noise2 = list(self.perlin.noise(length=10000, step=0.01, octave=1, count=1, seed=5))

        self.assertListEqual(noise1, noise2, "Two noises with the same seed and parameters are different!")


if __name__ == "__main__":
    unittest.main(exit=False)
