import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
# from perlin_1d import *
from perlin_2d import *


def plot_noise(noise, label="", color="blue"):
    Time = np.linspace(0, len(noise), len(noise))
    plt.plot(Time, noise, color=color)
    plt.title(label)
    plt.show()

def noise_distribution(noise, label, bins=10, color="blue"):
    plt.figure()
    plt.ylabel("gyakoriság")
    plt.xlabel("Járműsebesség [km/h]")
    plt.hist(noise, bins=bins, density=True, color=color, ec='black')
    plt.title(label)
    plt.show()

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
#               TEMP                 #
######################################

mode = {
        "city": [
            [0, 30, 0.3],
            [0, 90, 0.7]
            ],
        "highway": [
            [70, 90, 0.3],
            [100, 125, 0.7]
            ]
    }

perlin_2D = Perlin_2D()
transform = Transform()

# 1.
start = time.time()

noise = perlin_2D.noise(100, 0.1, 1, skip_factor=1, seed=1010)
end = time.time()
print(end - start)
print(len(noise))
vsp1 = noise[0]
stwa = transform.scale_noise(noise[-1], 600)

# noise = perlin_2D.noise(500, 0.01, 2, skip_factor=1, seed=23652)
# vsp2 = noise[0]

# # 2.
# scaled_vsp1 = transform.scale_to_range(vsp1, (mode["highway"][0][0], mode["highway"][0][1]))
# scaled_vsp2 = transform.scale_to_range(vsp2, (mode["highway"][1][0], mode["highway"][1][1]))

# distr_vsp = transform.transition_from_signal_to_another(scaled_vsp1, scaled_vsp2, 100, mode["highway"][0][2])

# # 3.
# stwa = transform.limit_stwa_by_vsp(stwa, distr_vsp)

# # 4.    
# limited_vsp  = transform.rate_limit_signal(distr_vsp, 5, 1)
# limited_stwa = transform.rate_limit_signal(stwa, 20, 1)

# # 5. (extra)
# ramped_vsp  = transform.ramp_from_and_to_zero(limited_vsp, 5, 1)
# ramped_stwa = transform.ramp_from_and_to_zero(limited_stwa, 20, 1)

# extended_vsp  = transform.extend_signal(ramped_vsp, 10000)
# extended_stwa = transform.extend_signal(ramped_stwa, 10000)

Time = np.linspace(0, 5*60, 500)
plot_noise(stwa)
plot_noise(numderive(stwa, Time))
plt.imshow(noise, cmap='gray')
plt.title("Perlin-zaj")
plt.show()

# speed = np.linspace(0, 200, 100)
# angle = 0.5+np.exp((1-speed/50)*2)*100

# plt.plot(speed, angle+50, color="blue")
# plt.xlabel("Járműsebesség [km/h]")
# plt.ylabel("Kormányszög [deg]")
# plt.title("Kormányszög limitálás járműsebesség alapján")
# plt.show()

# plot_noise(vsp1, "")
# noise_distribution(distr_vsp*2, "Módosított Perlin-zaj eloszlása", 11)
# bins=9
# seed=1010
# noise=vsp1*4

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


# try:
    # unittest.main()
# except SystemExit:
    # pass