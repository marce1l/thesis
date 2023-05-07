import numpy as np
import matplotlib.pyplot as plt
import time

# TODO:
#   -> fix rate limit so it limits to km/s? or deg/s
#   -> send rate limit plot
#   -> make signal transition demo and send plot with distribution chart
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

    def __generate_noise(self, width, step):
        noise = np.empty((width, width))
    
        for y in range(width):
            for x in range(width):
                noise[y][x] = self.__perlin(x*step, y*step)
        return noise

    def __generate_octaves(self, amplitude, width, step, octaves, divisor=2):
        octave_list = []

        for i in range(octaves):
            octave_list.append(self.scale_noise(self.__generate_noise(width, step), amplitude))
            amplitude = amplitude / divisor
            step      = step      * divisor     # step should get closer to 0 with each octave

        return octave_list

    def __combine_octaves(self, octave_list):
        return np.add.reduce(octave_list)

    def scale_noise(self, noise, amplitude):
        return noise * amplitude*2       # *2 because not sure if normal range is [-0.5, 0.5] or [-1, 1]

    def scale_to_range(self, noise, given_range)
        '''Doesn't account for minus values'''
        num1, num2 = given_range
        
        if num1 > num2:
            difference = num1 - num2
            offset = num1 - difference
        else:
            difference = num2 - num1
            offset = num2 - difference
        
        return noise + offset

    def extend_signal(self, signal, count):
        extend_list = []
        
        for i in range(len(signal)):
            t = np.linspace(0, len(signal[i]), len(signal))
            Time = np.linspace(0, len(signal[i]), count)
            
            out = np.interp(Time, t, signal[i])
            extend_list.append(out.tolist())
            # Time = Time / len(signal) * length
        return extend_list

    def rate_limit_signal(self, signal, limit, step):
        #1. végig a tömbön másodpercenként
        #2. megnézni, hogy egy másodpercbe tartozó elemek összege nagyobb-e, mint a limit
        #3. ha nem skip
        #4. ha igen, leosztani egy másodperccel a túlment értéket és kivonni mindegyik elemből
        limited = []

        for y in range(len(signal)):
            row = [signal[y][0]]
            for x in range(1, len(signal[0])):
                row.append(rate_limit_value(signal[y][x], row[x-1], limit, 1))
            limited.append(row)
        return limited

    def transition_from_signal_to_another(self, signal1, signal2, speed, percent):
        combined = []
        fade = np.linspace(0, 1, speed)
        
        indx = len(signal1)*percent
        j = 0
        for i in range(len(signal1)):
            if i >= indx and i < indx+len(fade):
                combined.append(signal1[i]*(1-fade[j]) + signal2[i]*fade[j])
                j += 1
            elif i < indx:
                combined.append(signal1[i])
            else:
                combined.append(signal2[i])
        return combined

    def __create_gradientVectors(self, size):
        gradX = np.random.rand(size,size) * 2 - 1
        gradY = np.random.rand(size,size) * 2 - 1
        
        # convert to unit vectors
        c = np.sqrt(gradX**2 + gradY**2)
        gradX /= c
        gradY /= c
        
        return (gradX, gradY)

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
        Time = np.linspace(0, len(noise), len(noise[0]))

        # plt.figure()
        # plt.hist(noise, bins=10, density=True, color='blue', ec='black')
        # plt.show()
        
        plt.figure()
        plt.plot(Time, noise[0], 'o')
        plt.show()
        
        # plt.imshow(noise, cmap='gray')
        # plt.show()


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

    noise = perlin_2D.noise(amplitude=600, width=500, step=0.01, octaves=4, seed=999)
    noise1 = perlin_2D.noise(amplitude=300, width=500, step=0.01, octaves=4, seed=167)
    # perlin_2D.plot_noise(noise)
    
    noise2 = perlin_2D.extend_signal(noise, 500)
    noise22 = perlin_2D.extend_signal(noise1, 500)
    noise22 = np.array(noise22) + 300
    # perlin_2D.plot_noise(noise2)

    noise4 = perlin_2D.transition_from_signal_to_another(noise2[0], noise22[0], 100, 0.5)
    Time = np.linspace(0, len(noise[0]), len(noise[0]))
    plt.plot(Time, noise4, label="combined")
    plt.figure()
    plt.plot(Time, noise2[0], label="600")
    plt.plot(Time, noise22[0], label="300")
    plt.legend()
    plt.figure()
    plt.imshow(noise2, cmap='gray')
    
    plt.figure()
    plt.hist(noise2[0], bins=10, density=True, color='blue', ec='black')
    plt.figure()
    plt.hist(noise4, bins=10, density=True, color='blue', ec='black')
    
    plt.show()
    # noise3 = perlin_2D.rate_limit_signal(noise2, 20, int(len(noise2[0])/len(noise2)))
    # perlin_2D.plot_noise(noise3)
    

if __name__ == "__main__":
    main()

# x = np.linspace(0, 4*np.pi, 1000)
# sin = np.sin(x*1.3)
# cos = np.cos(x*0.6)

# fade = np.linspace(0, 1, 400)

# where = 300
# j = 0
# combined = []
# for i in range(len(sin)):
    # if i >= where and i < where+len(fade):
        # combined.append(sin[i]*(1-fade[j]) + cos[i]*fade[j])
        # j += 1
    # elif i < where:
        # combined.append(sin[i])
    # else:
        # combined.append(cos[i])


# plt.plot(x, sin, label="sin*1.3")
# plt.plot(x, cos, label="cos*0.6")
# # plt.axvline(x[where], linestyle="--", color="red")
# # plt.axvline(x[where+len(fade)], linestyle="--", color="red")
# plt.axvspan(where/1000*(4*np.pi), (where+len(fade))/1000*(4*np.pi), color='grey', label="transition", alpha=0.2, zorder=10)
# plt.legend()

# plt.figure()
# plt.plot(x, combined, label="sin --> cos")
# plt.axvspan(where/1000*(4*np.pi), (where+len(fade))/1000*(4*np.pi), color='grey', label="transition", alpha=0.2, zorder=10)
# plt.legend()
# plt.show()
