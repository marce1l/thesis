import unittest
import time
# from perlin_1d import *
from perlin_2d import *

class TestPseudoRandomGenerator(unittest.TestCase):
    
    # test if it generates the same value with the same seed
    def test_generated_values_with_same_seed(self):
        seed = time.time()
        
        prg1 = PseudoRandomGenerator(seed)
        value1 = prg1.generate()
        
        prg2 = PseudoRandomGenerator(seed)
        value2 = prg2.generate()
        
        self.assertEqual(value1, value2, "Generated numbers does not match with same seed")


######################################
#           RATE LIMITER             #
######################################

def rate_limit_value(signal, previous_output, limiter, Ts):
    if abs((signal - previous_output) / Ts) > limiter:
        exp_sign = np.sign(signal - previous_output)
        value = previous_output + exp_sign * limiter * Ts
    else:
        value = signal
    return value

def test_step():
        t = np.arange(0,1.5,0.001)
        signal = np.append(3 * np.ones(500), -3*np.ones(500))
        signal = np.pad(signal, (0, len(t)- len(signal)), "constant")
        ratelimited = [signal[0]]
        
        for i in range(1,len(signal)):
            ratelimited.append(rate_limit_value(signal[i], ratelimited[-1], 10, 0.01))
            
        plt.plot(t, signal, label = "Wheel angle [deg]", color = "blue", alpha = 0.8)
        plt.plot(t, ratelimited, label = "Rate limited wheel angle [deg]", color = "firebrick")
        plt.legend()
        plt.show()

test_step()


# try:
    # unittest.main()
# except SystemExit:
    # pass