import numpy as np
import matplotlib.pyplot as plt
import time
import datetime


class Perlin_2D():
    # noise = [[]]
    
    # gradient vectors
    gradX = [[]]
    gradY = [[]]


    def perlin(self, x, y):
        # grid top left index
        xi = int(x)
        yi = int(y)
        
        # top left distance vector
        xd = x - xi
        yd = y - yi
        
        # top left, top right, bottom left, bottom right dot product
        p00 = xd     * self.gradX[yi][xi]     + yd     * self.gradY[yi][xi]
        p01 = (xd-1) * self.gradX[yi][xi+1]   + yd     * self.gradY[yi][xi+1]
        p10 = xd     * self.gradX[yi+1][xi]   + (yd-1) * self.gradY[yi+1][xi]
        p11 = (xd-1) * self.gradX[yi+1][xi+1] + (yd-1) * self.gradY[yi+1][xi+1]

        # fade for smooth linear interpolation
        xf = self.fade(xd)
        yf = self.fade(yd)
        
        a = self.interpolate(p00, p10, yf)
        b = self.interpolate(p01, p11, yf)
        return self.interpolate(a, b, xf)

    def fade(self, f):
        return 6 * f**5 - 15 * f**4 + 10 * f**3

    # linear interpolation
    def interpolate(self, a, b, x):
        return a + x * (b - a)

    def create_gradientVectors(self, seed):
        self.set_seed(seed)

        gradX = np.random.rand(256,256) * 2 - 1
        gradY = np.random.rand(256,256) * 2 - 1

        # convert to unit vectors
        c = np.sqrt(gradX ** 2 + gradY ** 2)
        gradX = gradX / c
        gradY = gradY / c
        
        return (gradX, gradY)

    def set_seed(self, seed):
        # -1000000000 to make it work for a century (2**32 numpy limit)
        offset = -1000000000
        
        # Note: python 2 and python 3 time.time() differs in precision
        if seed is None:
            seed = int(time.time())+offset
            
        try:
            np.random.seed(seed)
        except TypeError as e:
            seed = int(time.time())+offset
            print("TypeError: %s" % e)
            print("Current time is used as seed")
        except ValueError as e:
            seed = int(time.time())+offset
            print("ValueError: %s" % e)
            print("Current time is used as seed")
        
        np.random.seed(seed)
        # self.store_seed(seed, offset)
    
    def store_seed(self, seed, offset):
        seed = seed+(-offset)
        print(datetime.datetime.fromtimestamp(seed))
        raise NotImplementedError
    
    def noise(self, amplitude, width, step, seed=None):
        self.gradX, self.gradY = self.create_gradientVectors(seed)
        noise = np.empty((width, width))
    
        for y in range(width):
            for x in range(width):
                noise[y][x] = self.perlin(x*step, y*step)*amplitude*2       # *2 because not sure if normal range is [-0.5, 0.5] or [-1, 1]
        return noise

    def generate_octave(self):
        raise NotImplementedError

    def combine_octaves(self):
        raise NotImplementedError

    def plot_noise(self, noise):
        Time = [i for i in range(len(noise))]
        
        plt.plot(Time, noise[0])
        plt.show()
        
        plt.imshow(noise, cmap='gray')
        plt.show()


def generate_noise(amplitude, step, width):
    for y in range(width):
        for x in range(width):
            noise[y][x] = perlin(x*step, y*step)
    return noise
    
def generateOctave(octaves, amplitude, width, step, divisor=2):
    octave = []
    
    for i in range(octaves):
        octave.append(generate_noise(amplitude, step, width))
        amplitude = amplitude / divisor
        step      = step      / divisor
    return octave

def combineOctave(octave):
    combinedNoise = np.empty(len(octave[0][0], len(octave[0][0])))
    
    for y in range(len(octave[0])):
        for x in range (len(octave[0][0])):            
            total = 0
            for i in range(len(octave)):
                total += octave[k][y][x]
            combinedNoise[y][x] = total
    return combinedNoise


def main():
    perlin_2D = Perlin_2D()
    perlin_2D.plot_noise(perlin_2D.noise(amplitude=1, width=1000, step=0.01, seed=142))


if __name__ == "__main__":
    main()


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

def test_step(self):
        t = np.arange(0,1.5,0.001)
        signal = np.append(3 * np.ones(500), -3*np.ones(500))
        signal = np.pad(signal, (0, len(t)- len(signal)), "constant")
        ratelimited = [signal[0]]
        
        for i in range(1,len(signal)):
            ratelimited.append(rate_limit_value(signal[i], ratelimited[-1], 10, 0.01))
            
        plt.plot(t, signal, label = "Wheel angle [deg]", color = REFERENCE_COLOR, alpha = 0.8)
        plt.plot(t, ratelimited, label = "Rate limited wheel angle [deg]", color = "firebrick")
        plot_settings()
        plt.show()