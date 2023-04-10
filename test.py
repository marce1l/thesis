import unittest
import time
from perlin_demo2 import *

class TestPseudoRandomGenerator(unittest.TestCase):
    
    # test if it generates the same value with the same seed
    def test_generated_values_with_same_seed(self):
        seed = time.time()
        
        prg1 = PseudoRandomGenerator(seed)
        value1 = prg1.generate()
        
        prg2 = PseudoRandomGenerator(seed)
        value2 = prg2.generate()
        
        self.assertEqual(value1, value2, "Generated numbers does not match with same seed")

try:
    unittest.main()
except SystemExit:
    pass