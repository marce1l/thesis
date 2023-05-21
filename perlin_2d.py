import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.io as matio

# TODO:
#   -> merge 1D and 2D perlin?
#   -> numpy.random.seed() is legacy, replacement?
#   -> random generated seed can't be stored in filename

# Steering wheel angle and vehicle speed relation
# https://www.researchgate.net/figure/Steering-wheel-angle-limit-as-a-function-of-vehicle-speed_fig3_224266032
# https://www.researchgate.net/figure/Vehicle-Velocity-versus-Steering-Angle-explains-the-most-important-interaction-behavior_fig4_268407527

# -1000000000 to make it work for a century (2**32 numpy limit)
seed_offset = -1000000000

class Perlin_2D():

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

        xf = self.fade(xd)
        yf = self.fade(yd)
        
        a = self.interpolate(p00, p10, yf)
        b = self.interpolate(p01, p11, yf)
        return self.interpolate(a, b, xf)

    # for smooth linear interpolation
    def fade(self, f):
        return 6 * f**5 - 15 * f**4 + 10 * f**3

    # linear interpolation
    def interpolate(self, a, b, x):
        return a + x * (b - a)

    def generate_noise(self, width, step):
        noise = np.empty((width, width))
    
        for y in range(width):
            for x in range(width):
                noise[y][x] = self.perlin(x*step, y*step)
        return noise

    def generate_octaves(self, width, step, octaves, divisor=2):
        amplitude = 1.0
        octave_list = []

        for i in range(octaves):
            octave_list.append(self.generate_noise(width, step)*amplitude)
            amplitude = amplitude / divisor
            step      = step      * divisor     # step should get closer to 0 with each octave

        return octave_list

    def combine_octaves(self, octave_list):
        return np.add.reduce(octave_list)

    def create_gradientVectors(self, size):
        gradX = np.random.rand(size,size) * 2 - 1
        gradY = np.random.rand(size,size) * 2 - 1
        
        # convert to unit vectors
        c = np.sqrt(gradX**2 + gradY**2)
        gradX /= c
        gradY /= c
        
        return (gradX, gradY)

    def setup(self, seed):
        self.set_seed(seed)
        self.gradX, self.gradY = self.create_gradientVectors(256)

    def set_seed(self, seed):
        # Note: python 2 and python 3 time.time() differs in precision
        if seed is None:
            seed = int(time.time())+seed_offset
            
        try:
            np.random.seed(seed)
        except TypeError as e:
            seed = int(time.time())+seed_offset
            print("TypeError: %s" % e)
            print("Current time is used as seed")
        except ValueError as e:
            seed = int(time.time())+seed_offset
            print("ValueError: %s" % e)
            print("Current time is used as seed")
        
        np.random.seed(seed)

    def noise(self, width, step, octaves, seed=None):
        self.setup(seed)
        return self.combine_octaves(self.generate_octaves(width, step, octaves))


class Transform():      # rename

    def rate_limit_value(self, signal, previous_output, limiter, Ts):
        if abs((signal - previous_output) / Ts) > limiter:
            exp_sign = np.sign(signal - previous_output)
            value = previous_output + exp_sign * limiter * Ts
        else:
            value = signal
        return value
    
    def rate_limit_signal(self, signal, limit, sampling_rate):
        limited = [signal[0]]

        for i in range(1, len(signal)):
            limited.append(self.rate_limit_value(signal[i], limited[i-1], limit, sampling_rate))        # Ts should be set according to noise sampling rate
        return limited

    def scale_noise(self, noise, amplitude):
        return noise * amplitude*2       # *2 because not sure if normal range is [-0.5, 0.5] or [-1, 1]

    def extend_signal(self, signal, count):
        t = np.linspace(0, len(signal), len(signal))
        Time = np.linspace(0, len(signal), count)
        
        out = np.interp(Time, t, signal)
        return out
    
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
    
    def scale_to_range(self, noise, given_range):               # something doesn't work (can get lower values than given_range minimum)
        '''Doesn't account for minus values (Doesn't have to)'''
        num1, num2 = given_range
        
        if num1 > num2:
            difference = num1 - num2
            offset = num2
        else:
            difference = num2 - num1
            offset = num1

        return noise*difference + offset

    def limit_stwa_by_vsp(self, stwa, vsp):
        limited = []
        
        for i in range(0, len(stwa)):
            angle = self.get_stwa_from_vsp(vsp[i])
            if abs(stwa[i]) > angle:
                if stwa[i] < 0:
                    limited.append(angle*-1)
                else:
                    limited.append(angle)
            else:
                limited.append(stwa[i])
        return limited

    def get_stwa_from_vsp(self, vsp):
        # speed = np.linspace(0, 4, 100)
        # angle = 0.5+np.exp((1-speed)*2)     # geogebra
        # cut off at Endpos?
        return 0.5+np.exp((1-vsp/50)*2)*100
    
    def ramp_from_and_to_zero(self, signal, rate_limit, sampling_rate):
        signal[0]  = 0
        ramp_start = self.rate_limit_signal(signal, rate_limit, sampling_rate)
        ramp_start[-1] = 0
        flipped = np.flip(ramp_start)
        
        ramp_end = self.rate_limit_signal(flipped, rate_limit, sampling_rate)
        return np.flip(ramp_end)


def noise(test_length, mode, Endpos, octaves=2, seed=None, filename="export",):
    perlin    = Perlin_2D()
    transform = Transform()
    
    # 1.
    noise = perlin.noise(500, 0.01, octaves, seed=seed)
    vsp1 = noise[0]
    vsp2 = noise[int(len(noise)/2)]
    stwa = transform.scale_noise(noise[-1], Endpos)
    
    # 2.
    scaled_vsp1 = transform.scale_to_range(vsp1, (mode["highway"][0][0], mode["highway"][0][1]))
    scaled_vsp2 = transform.scale_to_range(vsp2, (mode["highway"][1][0], mode["highway"][1][1]))
    
    distr_vsp = transform.transition_from_signal_to_another(scaled_vsp1, scaled_vsp2, 100, mode["highway"][0][2])
    
    # 3.
    limited_by_vsp_stwa = transform.limit_stwa_by_vsp(stwa, distr_vsp)
    
    # 4.    
    limited_vsp  = transform.rate_limit_signal(distr_vsp, 5, 1)
    limited_stwa = transform.rate_limit_signal(limited_by_vsp_stwa, 20, 1)
    
    # 5.
    ramped_vsp  = transform.ramp_from_and_to_zero(limited_vsp, 5, 1)
    ramped_stwa = transform.ramp_from_and_to_zero(limited_stwa, 20, 1)
    
    # 6.
    extended_vsp  = transform.extend_signal(ramped_vsp, 10000)
    extended_stwa = transform.extend_signal(ramped_stwa, 10000)
    
    Time = np.linspace(0, test_length*60, 10000)
    
    filename = filename + "_" + str(seed_to_store(seed)) + ".mat"
    # export_to_mat(extended_vsp, extended_stwa, Time, filename=filename)
    
    return (extended_vsp, extended_stwa, Time)

def seed_to_store(seed):
    if seed is None:
        return ""
        # return seed+(-seed_offset)
    else:
        return seed

def export_to_mat(vsp, stwa, Time, filename = ""):
    path = r'c:\Users\dud\Desktop' + os.sep + filename

    export_dict = {}
    export_dict["VSP"] = vsp
    export_dict["STWA"] = stwa
    export_dict["Time"] = Time

    matio.savemat(path, export_dict)
    print(path)

def main():
    mode = {
        "city": [
            [0, 30, 0.3],
            [0, 90, 0.7]
            ],
        "highway": [
            [60, 120, 0.6],
            [90, 130, 0.4]
            ]
    }

    vsp, stwa, Time = noise(5, mode, 600, 2, seed=1000)
    
    # 6. (plot)
    plt.figure()
    plt.plot(Time, vsp, label="vsp")
    plt.legend()
    
    plt.figure()
    plt.plot(Time, stwa, label="stwa")
    plt.legend()
    
    plt.show()
    

if __name__ == "__main__":
    main()


