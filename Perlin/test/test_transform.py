import unittest

from setup import *
from Perlin.transform import *
from Perlin.perlin import Perlin
from Perlin.robustness_generator import VehicleSpeedRange


class Test_Transform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        perlin = Perlin()
        cls.noise = perlin.noise(length=10000, step=0.01, octave=2, count=1)

    def test_scaling_noise(self):
        min_value = -120
        max_value = 180

        scaled_noise = offset_to_range(scale_up_noise(self.noise), min_value, max_value)

        self.assertLessEqual(max(scaled_noise), max_value, "The maximum value of the scaled noise (%.1f) is higher than required! (%.1f)" % (max(scaled_noise), max_value))
        self.assertGreaterEqual(min(scaled_noise), min_value, "The minimum value of the scaled noise (%.1f) is lower than required! (%.1f)" % (min(scaled_noise), min_value))

    def test_transition_between_signals(self):
        signal_length = 10000
        test_length = 360
        acc_rate = 10.0
        deacc_rate = 30.0

        Time = np.linspace(0, test_length, signal_length)
        dt = test_length/signal_length

        # accelerating case
        vsp_ranges = [VehicleSpeedRange(40, 40, 0.3), VehicleSpeedRange(100, 100, 0.3), VehicleSpeedRange(150, 150, 0.4)]
        signals = [np.linspace(r.low_lim, r.high_lim, signal_length) for r in vsp_ranges]

        trans_signal = transition_between_signals(signals, vsp_ranges, dt, acc_rate, deacc_rate)
        trans_signal_deriv = numderive(trans_signal, Time)

        self.assertEqual(len(trans_signal), signal_length, "The length of the final signal (%d) differs from the length of the input signals length! (%d)" % (len(trans_signal), signal_length))
        self.assertLess(max(abs(trans_signal_deriv)), acc_rate, "The derivative of the final signal is higher than expected")

        # deaccelerating case
        vsp_ranges = [VehicleSpeedRange(120, 120, 0.1), VehicleSpeedRange(55, 55, 0.2), VehicleSpeedRange(0, 0, 0.6), VehicleSpeedRange(10, 10, 0.1)]
        signals = [np.linspace(r.low_lim, r.high_lim, signal_length) for r in vsp_ranges]

        trans_signal = transition_between_signals(signals, vsp_ranges, dt, acc_rate, deacc_rate)
        trans_signal_deriv = numderive(trans_signal, Time)

        self.assertEqual(len(trans_signal), signal_length, "The length of the final signal (%d) differs from the length of the input signals length! (%d)" % (len(trans_signal), signal_length))
        self.assertLess(max(abs(trans_signal_deriv)), deacc_rate, "The derivative of the final signal is higher than expected")

    def test_get_fade_length(self):
        dt         = 360/10000
        acc_rate   = 10
        deacc_rate = 30

        self.assertTrue(get_fade_length(0, acc_rate, deacc_rate, dt) % 2 == 0, "The calculated fade_length is not even!")
        self.assertTrue(get_fade_length(100, acc_rate, deacc_rate, dt) % 2 == 0, "The calculated fade_length is not even!")
        self.assertTrue(get_fade_length(-100, acc_rate, deacc_rate, dt) % 2 == 0, "The calculated fade_length is not even!")

    def test_get_max_stwa(self):
        min_stwa = 10
        endpos = 400
        endpos_till_vsp = 30

        vsp = np.linspace(0, 300, 10000)
        max_stwa = get_max_stwa(vsp, min_stwa, endpos, endpos_till_vsp)

        last_endpos_idx = 0
        for i in range(len(max_stwa)):
            if max_stwa[i+1] != endpos:
                last_endpos_idx = i
                break

        self.assertGreaterEqual(min(max_stwa), min_stwa, "The minimum value of the max_stwa (%.2f) is lower than the min_stwa (%d) parameter!" % (min(max_stwa), min_stwa))
        self.assertLessEqual(max(max_stwa), endpos, "The maximum value of the max_stwa (%.2f) is higher than the endpos (%d) parameter!" % (max(max_stwa), endpos))
        self.assertAlmostEqual(vsp[last_endpos_idx], endpos_till_vsp, 1, "The maximum vehicle speed where the steering wheel angle can reach the endpos (%.2f) is not in the allowed range (%d rounded)" % (vsp[last_endpos_idx], endpos_till_vsp))


if __name__ == "__main__":
    unittest.main(exit=False)
