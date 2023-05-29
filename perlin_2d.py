import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.io as matio

# Octaves and test length are directly proportional to step size in terms of stwa asp results (2*test_length --> 2*step size, +2 octave --> 2*step size) (step size / 2 in my implementation)

# TODO:
#   -> limit stwa after extending it because sampling frequency? (vsp should be fine)
#   -> merge 1D and 2D perlin?
#   -> numpy.random.seed() is legacy, replacement?

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

    def generate_noise(self, width, step, skip_factor):
        noise = np.empty((int(width/skip_factor), width))
    
        for y in range(width):
            if y%skip_factor == 0:
                for x in range(width):
                    noise[int(y/skip_factor)][x] = self.perlin(x*step, y*step)
        return noise

    def generate_octaves(self, width, step, octaves, skip_factor, divisor=2):
        amplitude = 1.0
        octave_list = []

        for i in range(octaves):
            octave_list.append(self.generate_noise(width, step, skip_factor)*amplitude)
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

    def noise(self, width, step, octaves, skip_factor, seed=None):
        self.gradX, self.gradY = self.create_gradientVectors(512)
        return self.combine_octaves(self.generate_octaves(width, step, octaves, skip_factor))


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
            limited.append(self.rate_limit_value(signal[i], limited[-1], limit, sampling_rate))
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
    
    def scale_to_range(self, noise, min_range, max_range):
        '''Doesn't account for minus values (Doesn't have to)'''
        difference = max_range - min_range
        offset = min_range

        max_value = abs(max(noise))
        min_value = abs(min(noise))

        if max_value > min_value:
            noise = noise * (1/max_value)
        else:
            noise = noise * (1/min_value)
        
        noise = ((noise+1)/2)*difference+offset
        
        max_value = max(noise)
        min_value = min(noise)
        
        if max_range-max_value < min_value-min_range:
            noise = noise - (min_value-min_range)/2
        else:
            noise = noise + (max_range-max_value)/2

        return noise

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
        # speed = np.linspace(0, 4, 100)*50
        # angle = np.exp((1-speed)*2)*100+10     # geogebra
        # cut off at Endpos?
        return np.exp((1-vsp/50)*2)*100+10
    
    def ramp_from_and_to_zero(self, signal, rate_limit, sampling_rate):
        ramp_start = [0]
        
        for i in range(1, len(signal)):
            if abs((signal[i] - ramp_start[-1]) / sampling_rate) < rate_limit:
                break
            ramp_start.append(self.rate_limit_value(signal[i], ramp_start[-1], rate_limit, sampling_rate))
        
        for i in range(len(ramp_start), len(signal)):
            ramp_start.append(signal[i])
        
        flipped = np.flip(ramp_start)

        ramp_end = [0]

        for i in range(1, len(flipped)):
            if abs((flipped[i] - ramp_end[-1]) / sampling_rate) < rate_limit:
                break
            ramp_end.append(self.rate_limit_value(flipped[i], ramp_end[-1], rate_limit, sampling_rate))

        for i in range(len(ramp_end), len(flipped)):
            ramp_end.append(flipped[i])

        return np.flip(ramp_end) 


def noise(test_length, mode_array, mode, Endpos, octaves=2, seed=None, filename="export"):
    perlin    = Perlin_2D()
    transform = Transform()
    
    perlin_length = 5000
    test_length   = test_length*60
    
    seed = set_seed(seed)
    
    # 1.
    noise = perlin.noise(perlin_length, 0.0125, octaves, skip_factor=500, seed=seed)
    vsp1 = noise[0]
    vsp2 = noise[1]
    stwa = transform.scale_noise(noise[2], Endpos)
    
    # 2.    # TODO: add check for max, min value
    scaled_vsp1 = transform.scale_to_range(vsp1, mode_array[mode][0][0], mode_array[mode][0][1])
    scaled_vsp2 = transform.scale_to_range(vsp2, mode_array[mode][1][0], mode_array[mode][1][1])
    
    distr_vsp = transform.transition_from_signal_to_another(scaled_vsp1, scaled_vsp2, 100, mode_array[mode][0][2])
    
    ramped_vsp = transform.ramp_from_and_to_zero(distr_vsp, 10, test_length/perlin_length)
    # 3.
    limited_by_vsp_stwa = transform.limit_stwa_by_vsp(stwa, distr_vsp)
    
    Time = np.linspace(0, test_length, len(ramped_vsp))
    ramped_stwa = transform.ramp_from_and_to_zero(limited_by_vsp_stwa, max(abs(numderive(limited_by_vsp_stwa, Time))), test_length/perlin_length)
    
    # 4.
    limited_vsp = transform.rate_limit_signal(ramped_vsp, 10, test_length/perlin_length)
    limited_stwa = transform.rate_limit_signal(ramped_stwa, 800, test_length/perlin_length)
    
    # 6.
    extended_vsp  = transform.extend_signal(ramped_vsp, 10000)
    extended_stwa = transform.extend_signal(ramped_stwa, 10000)
    
    Time = np.linspace(0, test_length, len(limited_vsp))
    
    filename = filename + mode + "_" + str(octaves) + "_" + str(Endpos) + "_" + str(mode_array[mode][0][0]) + str(mode_array[mode][0][1]) + "_" + str(mode_array[mode][0][2]) + "_" + str(mode_array[mode][1][0]) + str(mode_array[mode][1][1]) + "_" + str(seed) + ".mat"
    # export_to_mat(limited_vsp, limited_stwa, stwa, distr_vsp, Time, filename=filename)
    
    return limited_vsp, limited_stwa, stwa, Time

def set_seed(seed):
    # Note: python 2 and python 3 time.time() differs in precision
    if seed is None:
        # seed = int(time.time())+seed_offset
        seed = int(time.time())
        
    try:
        np.random.seed(seed)
    except TypeError as e:
        # seed = int(time.time())+seed_offset
        seed = int(time.time())
        print("TypeError: %s" % e)
        print("Current time is used as seed")
    except ValueError as e:
        # seed = int(time.time())+seed_offset
        seed = int(time.time())
        print("ValueError: %s" % e)
        print("Current time is used as seed")
    
    np.random.seed(seed)
    return seed

def export_to_mat(vsp, stwa, Time, filename = ""):
    path = r'c:\Users\dud\Desktop' + os.sep + filename

    export_dict = {}
    export_dict["VSP"] = vsp
    export_dict["STWA"] = stwa
    export_dict["raw_STWA"] = raw_stwa
    export_dict["raw_VSP"] = raw_vsp
    export_dict["Time"] = Time

    matio.savemat(path, export_dict)
    print(path)

def numderive(y_values, x_values):
    ''' Calculate differential signal as diff(y)/diff(x) '''
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    derivate = np.diff(y_values) / np.asarray( np.diff(x_values), float)
    derivate = np.append(derivate, derivate[-1])        # array size compensation due to diff
    return derivate 


def main():
    mode_array = {
        "parking": [    # 5000, 5m, 0.005, Endpos deg, 1 octave   ---> target 100 deg/s
            [0, 20, 0.8],
            [10, 35, 0.2]
        ],
        # "city": [
            # [15, 70, 0.7],
            # [0, 40, 0.3]
        # ],
        "city": [       # 5000, 5m, 0.025, 200 deg, 1 octave   ---> target 250-300 deg/s
            [50, 70, 0.7],
            [15, 40, 0.3]
        ],
        "rural": [      # 5000, 5m, 0.025, 150 deg, 1 octave   ---> target 100-150 deg/s
            [60, 90, 0.6],
            [40, 60, 0.4]
        ],
        "highway": [    # 5000, 5m, 0.005, 60 deg, 1 octave   ---> target 10-15 deg/s
            [60, 100, 0.2],
            [90, 130, 0.8]
        ]
    }

    vsp, stwa, raw_stwa, Time = noise(test_length=5, mode_array=mode_array, mode="rural", Endpos=150, octaves=3, filename="")
    
    # t = Transform()
    
    # print(t.get_stwa_from_vsp(120))
    # print(max(numderive(stwa, Time)))
    
    # speed = np.linspace(0, 4, 100)
    # angle = np.exp((1-speed)*2)*100+10    # geogebra
    # plt.figure()
    # plt.plot(speed*50, angle)
    # plt.show()

    # 6. (plot)
    plt.figure()
    plt.plot(Time, numderive(stwa, Time), label="stwa derivative")
    plt.legend()
    
    plt.figure()
    plt.plot(Time, stwa, label="stwa")
    # plt.plot(Time, raw_stwa)
    plt.legend()
    
    plt.show()
    

if __name__ == "__main__":
    main()


# def GetLookupDatabase(path):
    # if not os.path.exists(path):
        # raise Exception("%s does NOT exists!" % path)

    # vars = matio.loadmat(path)
    # for var, value in vars.items():
        # try:
            # if len(value) == 1:
                # vars[var] = value[0]
            # else:   vars[var] = value.flatten()   # flatten nested array structures
        # except: pass

    # return vars

# path = r"\\d1wficdc09.europe.prestagroup.com\HOME\Active\Budapest\students\marcell.mikes\Desktop\reference\parking_1_600_020_0.8_1035_1684926392.mat"
# database = GetLookupDatabase(path=path)
# print(database)

# Time = database["Time"]
# Angle = database["STWA"]
# Vsp = database["VSP"]   # plt.plot(Time, Angle)

# plt.figure()
# plt.plot(Time, Vsp)
# plt.show()