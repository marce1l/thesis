import math
import time
import matplotlib.pyplot as plt

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


class Perlin_1D():
    
    def noise(self, amplitude, wavelength, width, octaves, divisor=2):
        return self.combine_octaves(self.generate_octave(amplitude, wavelength, width, octaves, divisor))
    
    def combine_octaves(self, octave):
        combinedOctaves = []
        
        for i in range(0, len(octave[0])):
            total = 0
            for j in range(0, len(octave)):
                total += octave[j][i]
            combinedOctaves.append(total)
        return combinedOctaves
    
    # octave generator      rename ?
    def generate_octave(self, amplitude, wavelength, width, octaves, divisor):
        octave = []
        
        for i in range(0, octaves):
            octave.append(self.perlin(amplitude, wavelength, width))
            amplitude = amplitude / divisor
            wavelength = wavelength / divisor
        return octave
    
    # 1D Perlin line generator
    def perlin(self, amplitude, wavelength, width):
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
                pos.append(self.interpolate(a, b, (x % wavelength) / wavelength) * amplitude)
            x += 1
        return pos

    # cosine interpolation
    def interpolate(self, y1, y2, x):
        xv = (1 - math.cos(x * math.pi)) * 0.5
        return y1 + (y2 - y1) * xv

    def plot_noise(self, noise):
        Time = [i for i in range(0, len(noise))]
        plt.plot(Time, noise)
        
        plt.show()


# not integrated yet
def rate_limit_value(signal, previous_output, limiter, Ts):
    if abs((signal - previous_output) / Ts) > limiter:
        exp_sign = np.sign(signal - previous_output)
        value = previous_output + exp_sign * limiter * Ts
    else:
        value = signal
    return value

def main():
    perlin_1D = Perlin_1D()

    noise = perlin_1D.noise(amplitude=1.0, wavelength=512.0, width=10000, octaves=12)
    perlin_1D.plot_noise(noise)


if __name__ == "__main__":
    main()