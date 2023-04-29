import numpy as np
import matplotlib.pyplot as plt
import time

# TODO:
#   -> Stitch together multiple noise's for longer signal (noise[0] + noise[1]....) ((easy))
#          have to find ideal step rate first
#      3.6m ms = 1h
#   -> Performane improvements (ditching for loops) IDK...

class Perlin_2D():

    # gradient vectors
    gradX = [[]]
    gradY = [[]]


    def __perlin(self, x, y):
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

        xf = self.__fade(xd)
        yf = self.__fade(yd)
        
        a = self.__interpolate(p00, p10, yf)
        b = self.__interpolate(p01, p11, yf)
        return self.__interpolate(a, b, xf)

    # for smooth linear interpolation
    def __fade(self, f):
        return 6 * f**5 - 15 * f**4 + 10 * f**3

    # linear interpolation
    def __interpolate(self, a, b, x):
        return a + x * (b - a)

    def __generate_noise(self, amplitude, width, step):
        noise = np.empty((width, width))
    
        for y in range(width):
            for x in range(width):
                noise[y][x] = self.__perlin(x*step, y*step)*amplitude*2       # *2 because not sure if normal range is [-0.5, 0.5] or [-1, 1]
                # if x != 0:
                    # noise[y][x] = rate_limit_value(noise[y][x], noise[y][x-1], amplitude*2*0.005, 1)
        return noise

    def __generate_octaves(self, amplitude, width, step, octaves, divisor=2):
        octave_list = []

        for i in range(octaves):
            octave_list.append(self.__generate_noise(amplitude, width, step))
            amplitude = amplitude / divisor
            step      = step      * divisor     # step should get closer to 0 with each octave

        return octave_list

    def __combine_octaves(self, octave_list):
        return np.add.reduce(octave_list)

    def __create_gradientVectors(self, size):
        gradX = np.random.rand(size,size) * 2 - 1
        gradY = np.random.rand(size,size) * 2 - 1
        
        # convert to unit vectors
        c = np.sqrt(gradX**2 + gradY**2)
        gradX /= c
        gradY /= c
        
        return (gradX, gradY)

    def _stitch_noise(self, noise, width, step, grid_size):
        if (width * step) * grid_size:
            pass
        for x in noise:
            combined_noise = np.append(combined_noise, x)
        return combined_noise

    def __setup(self, seed):
        self.__set_seed(seed)
        self.gradX, self.gradY = self.__create_gradientVectors(256)

    def __set_seed(self, seed):
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
        # self.__store_seed(seed, offset)
    
    def __store_seed(self, seed, offset):
        seed = seed+(-offset)
        raise NotImplementedError

    def noise(self, amplitude, width, step, octaves, seed=None):
        self.__setup(seed)
        
        return self.__combine_octaves(self.__generate_octaves(amplitude, width, step, octaves))

    def plot_noise(self, noise):
        Time = [i for i in range(len(noise))]
        
        plt.figure()
        plt.hist(noise[0], bins=10, density=True, color='blue', ec='black')
        plt.show()
        
        plt.figure()
        plt.plot(Time, noise[0])
        plt.show()
        
        plt.imshow(noise, cmap='gray')
        plt.show()


def rate_limit_value(signal, previous_output, limiter, Ts):
    if abs((signal - previous_output) / Ts) > limiter:
        # print("sig: %s prev: %s" % (signal, previous_output))
        exp_sign = np.sign(signal - previous_output)
        value = previous_output + exp_sign * limiter * Ts
    else:
        value = signal
    return value

def main():
    perlin_2D = Perlin_2D()
    
    amplitude = 600
    start = time.time()
    noise = perlin_2D.noise(amplitude=amplitude, width=2000, step=0.01, octaves=1, seed=142)
    end = time.time()
    print(end - start)
    perlin_2D.plot_noise(noise)
    
    
    # noise = noise[0]
    # Time = [i for i in range(len(noise))]
    
    # out = [noise[0]]
    # for i in range(1, len(noise)):
        # out.append(rate_limit_value(noise[i], out[-1], amplitude*0.01, 1))

    # plt.plot(Time, noise, label="normal")
    # plt.plot(Time, out, label="limited")
    # plt.legend()
    
    # plt.show()

if __name__ == "__main__":
    main()
