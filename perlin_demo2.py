import math
import random
import time
import matplotlib.pyplot as plt
import scipy.stats as stats

# pseudo-random number generator
class PseudoRandomGenerator():

    # linear congruential generator parameters
    M = 4294967296.0        # .0 necessary
    A = 1664525
    C = 1

    def __init__(self, seed):
        self.seed = seed
        self.Z    = math.floor(seed * self.M)
    
    def generate(self):
        self.Z = (self.A * self.Z + self.C) % self.M
        return self.Z / self.M - 0.5


class PerlinNoise():
    
    def Noise(self):
        pass
    
    def CombineOctaves(self, octave):
        combinedOctaves = []
        
        for i in range(0, len(octave[0])):
            total = 0
            for j in range(0, len(octave)):
                total += octave[j][i]
            combinedOctaves.append(total)
        return combinedOctaves
    
    # octave generator      rename ?
    def generateOctave(self, amplitude, wavelength, width, octaves, divisor=2):
        octave = []
        
        for i in range(0, octaves):
            octave.append(self.Perlin(amplitude, wavelength, width))
            amplitude = amplitude / divisor
            wavelength = wavelength / divisor
        return octave
    
    # 1D Perlin line generator
    def Perlin(self, amplitude, wavelength, width):
        prg = PseudoRandomGenerator(time.time())
        a = prg.generate()
        b = prg.generate()
        x = 0
        pos = []
        
        while(x < width):
            if (x % wavelength == 0):
                a = b
                b = prg.generate()
                pos.append(a * amplitude)
            else:
                pos.append(self.Interpolate(a, b, (x % wavelength) / wavelength) * amplitude)
            x += 1
        return pos

    # cosine interpolation
    def Interpolate(self, y1, y2, x):
        xv = (1 - math.cos(x * math.pi)) * 0.5
        return y1 + (y2 - y1) * xv


def plot_noise(noise):
    Time = [i for i in range(0, len(noise[0]))]
    
    bins = 30
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle('Distribution')
    n, x, _ = ax1.hist(noise[0], bins=bins, density=True, color='blue', ec='black')
    ax1.plot(x, stats.gaussian_kde(noise[0])(x), color='black', lw=2.5)
    n, x, _ = ax2.hist(noise[1], bins=bins, density=True, color='orange', ec='black')
    ax2.plot(x, stats.gaussian_kde(noise[1])(x), color='black', lw=2.5)
    n, x, _ = ax3.hist(noise[2], bins=bins, density=True, color='green', ec='black')
    ax3.plot(x, stats.gaussian_kde(noise[2])(x), color='black', lw=2.5)
    n, x, _ = ax4.hist(noise[3], bins=bins, density=True, color='red', ec='black')
    ax4.plot(x, stats.gaussian_kde(noise[3])(x), color='black', lw=2.5)


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 6))
    fig.suptitle('Perlin noise')
    ax1.plot(Time, noise[0], color='blue')
    ax2.plot(Time, noise[1], color='orange')
    ax3.plot(Time, noise[2], color='green')
    ax4.plot(Time, noise[3], color='red')
    
    plt.show()

def rate_limit_value(signal, previous_output, limiter, Ts):
    if abs((signal - previous_output) / Ts) > limiter:
        exp_sign = np.sign(signal - previous_output)
        value = previous_output + exp_sign * limiter * Ts
    else:
        value = signal
    return value

def main():
    pn = PerlinNoise()

    noise = [pn.CombineOctaves(pn.generateOctave(amplitude=1, wavelength=512, width=10000, octaves=12)) for i in range(4)]
    plot_noise(noise)

if __name__ == "__main__":
    main()