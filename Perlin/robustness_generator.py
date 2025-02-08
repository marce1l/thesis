import matplotlib.pyplot as plt
import numpy as np
import logging
import random
import time
import json
import os

from dataclasses import dataclass, asdict
from fractions import Fraction
from .perlin import Perlin
from typing import Union
from .transform import *

DEBUG_MODE = False

logging_level = logging.DEBUG if DEBUG_MODE else logging.CRITICAL
logger = logging.getLogger("perlin_robustness")
logger.setLevel(logging_level)

stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s", style="%", datefmt="%H:%M")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class FractionEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Fraction):
            return str(o)
        return super().default(o)

@dataclass
class VehicleSpeedRange:
    low_lim:    float                           # low limit of the vehicle speed range
    high_lim:   float                           # high limit of the vehicle speed range
    percent:    Union[Fraction, float]          # percent of the final vehicle speed

    def __post_init__(self):
        validate_type(self.low_lim,  "low_lim",  int, float)
        validate_type(self.high_lim, "high_lim", int, float)
        validate_type(self.percent,  "percent",  Fraction, float)

        validate_boundaries(self.low_lim,  "low_lim",  min_value=0,   max_value=300)
        validate_boundaries(self.high_lim, "high_lim", min_value=0,   max_value=300)
        validate_boundaries(float(self.percent),  "percent",  min_value=0.0, max_value=1.0)

        if self.low_lim > self.high_lim:
            raise ValueError("'low_lim' can not be higher than 'high_lim'!")

@dataclass
class InputParameters:
    seed:               int                     # seed of the generated noises
    test_length_sec:    int                     # final maneuver length (sec)
    global_asp_limit:   float                   # global limit of the derivative of steering wheel angle (deg/s)
    vsp_ranges:         list[VehicleSpeedRange] # vehicle speed ranges of the final vehicle speed
    endpos:             int                     # endpos of the current system (deg)
    octave:             int = 5                 # octave of the generated noises
    min_stwa:           int = 5                 # minimum amplitude of the maximum possible steering wheel angle (deg)
    endpos_till_vsp:    int = 15                # maximum vehicle speed where the steering wheel angle could reach the endpos (km/h)
    perlin_length:      int = 10000             # length of the generated noise
    acc_ramp_rate:      float = 10.0            # transition rate between vehicle speed ranges in case of acceleration ((km/h)/s)
    deacc_ramp_rate:    float = 30.0            # transition rate between vehicle speed ranges in case of deacceleration ((km/h)/s)

    def to_dict(self):
        return asdict(self)

    def to_json(self, savedir, fname):
        with open(os.path.join(savedir, fname), 'w') as f:
            json.dump(self.to_dict(), f, indent=4, cls=FractionEncoder)

    @classmethod
    def from_json(InputParameters, path):
        with open(path, 'r') as f:
            params = json.load(f)

        ranges = []
        for vsp_range in params["vsp_ranges"]:
            percent = vsp_range["percent"]

            if isinstance(percent, str):
                percent = Fraction(percent)

            ranges.append(VehicleSpeedRange(vsp_range["low_lim"], vsp_range["high_lim"], percent))

        params["vsp_ranges"] = ranges
        return InputParameters(**params)

@dataclass
class UserParameters:
    endpos:             int
    test_length_sec:    int
    vsp_ranges:         list[VehicleSpeedRange]
    seed:               int = int(time.time())
    global_asp_limit:   float = 700

    def __post_init__(self):
        validate_type(self.endpos,              "endpos",           int)
        validate_type(self.global_asp_limit,    "global_asp_limit", int, float)
        validate_type(self.test_length_sec,     "test_length_sec",  int)
        validate_type(self.vsp_ranges,          "vsp_ranges",       list)
        validate_type(self.seed,                "seed",             int)

        validate_boundaries(self.endpos,            "endpos",           min_value=0)
        validate_boundaries(self.global_asp_limit,  "global_asp_limit", min_value=0)
        validate_boundaries(self.test_length_sec,   "test_length_sec",  min_value=0)

        if sum(vsp_range.percent for vsp_range in self.vsp_ranges) != 1:
            raise Warning("The sum of vehicle speed percentage's shall add up to 1.0")

class RobustnessGeneratorModel:
    def __init__(self, input_params: InputParameters):
        self.params = input_params

    def generate(self):
        logger.info("creating perlin noises")
        vsp_noises, stwa_noise = self.create_noise()

        logger.info("creating vehicle speed")

        for i in range(50):
            try:
                vsp = self.create_vsp(vsp_noises)
            except TransitionOverlapError:
                random.shuffle(self.params.vsp_ranges)
            else:
                break
        else:
            vsp = self.create_vsp(vsp_noises)

        logger.info("creating steering wheel angle")
        stwa = self.create_stwa(stwa_noise, vsp)
        Time = np.linspace(0, self.params.test_length_sec, self.params.perlin_length)

        self.output_check(vsp, stwa, Time)

        return Time, vsp, stwa

    def create_noise(self):
        perlin = Perlin()

        # Step size equation for a signal is the following (test length * target max rate) / (number of elements * range of the signal * octave correction)
        # Octave has an effect on the derivative of the noise, thus based on experience a *1.4^octave correction was added
        stwa_step = (self.params.test_length_sec*self.params.global_asp_limit) / (self.params.perlin_length*self.params.endpos*2*1.4**self.params.octave)
        logger.debug("stwa step size: %s" % stwa_step)
        assert stwa_step < 1, "Stwa step size has to be lower than 1!"

        stwa_noise = perlin.noise(self.params.perlin_length, stwa_step, self.params.octave, 1, self.params.seed)

        vsp_noises = []
        for i in range(len(self.params.vsp_ranges)):
            if self.params.vsp_ranges[i].low_lim == self.params.vsp_ranges[i].high_lim:
                vsp_noises.append(np.ones((1, self.params.perlin_length))[0])
                continue

            vsp_step = (self.params.test_length_sec) / (self.params.perlin_length*1.4**self.params.octave) * 1/10
            logger.debug("vsp_ranges\t%2d  low: %3d  high: %3d  percent: %s\tstep_size: %s" % (i, self.params.vsp_ranges[i].low_lim, self.params.vsp_ranges[i].high_lim, self.params.vsp_ranges[i].percent, vsp_step))
            assert vsp_step < 1, "Vsp step size has to be lower than 1!"

            vsp_noises.append(perlin.noise(self.params.perlin_length, vsp_step, self.params.octave, 1, self.params.seed))

        return vsp_noises, stwa_noise

    def create_vsp(self, noises):
        scaled_vsps = [offset_to_range(scale_up_noise(noises[i]), self.params.vsp_ranges[i].low_lim, self.params.vsp_ranges[i].high_lim) for i in range(len(noises))]

        distr_vsp = transition_between_signals(scaled_vsps, self.params.vsp_ranges, self.params.test_length_sec/self.params.perlin_length, self.params.acc_ramp_rate, self.params.deacc_ramp_rate)

        return distr_vsp

    def create_stwa(self, noise, vsp):
        max_stwa = get_max_stwa(vsp, self.params.min_stwa, self.params.endpos, self.params.endpos_till_vsp)

        scaled_noise = scale_up_noise(noise)

        vsp_limited_stwa = scaled_noise * max_stwa

        return vsp_limited_stwa

    def output_check(self, vsp, stwa, time):
        max_vsp = max(abs(vsp))
        max_stwa = max(abs(stwa))

        max_vsp_deriv = max(abs(numderive(vsp, time)))
        max_stwa_deriv = max(abs(numderive(stwa, time)))

        logger.debug("output_check\tmax_vsp: %.2f  max_stwa: %.2f  max_vsp_deriv: %.2f  max_stwa_deriv: %.2f" % (max_vsp, max_stwa, max_vsp_deriv, max_stwa_deriv))

        if max_vsp > 300:
            raise Warning("Absolute maximum vsp %.2f is higher than 300" % max_vsp_deriv)

        if max_stwa > self.params.endpos:
            raise Warning("Absolute maximum stwa %.2f is higher than %d" % (max_vsp_deriv, self.params.endpos))

        if max_vsp_deriv > 50:
            raise Warning("Absolute maximum vsp derivative %.2f is higher than 50" % max_vsp_deriv)

        if max_stwa_deriv > 1000:
            raise Warning("Absolute maximum stwa derivative %.2f is higher than 1000" % max_stwa_deriv)

    def save_params(self, savedir, fname):
        self.params.to_json(savedir, fname)
        logger.info("input_params saved:\t%s" % os.path.join(savedir, fname))

    def get_params(self):
        return self.params

def RobustnessGenerator(user_params):
    logger.debug("user_params\tseed: %s  endpos: %3d  test_length_sec: %s  global_asp_limit: %4d" % (user_params.seed, user_params.endpos, user_params.test_length_sec, user_params.global_asp_limit))
    input_params = InputParameters(seed=user_params.seed, endpos=user_params.endpos, test_length_sec=user_params.test_length_sec, vsp_ranges=user_params.vsp_ranges, global_asp_limit=user_params.global_asp_limit)
    return RobustnessGeneratorModel(input_params)

def RobustnessGenerator_from_json(path):
    input_params = InputParameters.from_json(path)
    logger.info("input_params loaded:\t%s" % path)
    return RobustnessGeneratorModel(input_params)

def validate_type(arg, name, *types):
    if not isinstance(arg, types):
        types_str = " or ".join(t.__name__ for t in types)
        raise TypeError("Field '%s' must be of %s type" % (name, types_str))

def validate_boundaries(arg, name, min_value=None, max_value=None):
    tx = "value"

    if hasattr(arg, '__len__'):
        arg = len(arg)
        tx = "length"

    if min_value is not None and max_value is not None and min_value == max_value and arg != min_value:
        raise ValueError("Field '%s' %s must be %s!" % (name, tx, min_value))

    if min_value is not None and arg < min_value:
        raise ValueError("Field '%s' %s must be higher than %s" % (name, tx, min_value))

    if max_value is not None and arg > max_value:
        raise ValueError("Field '%s' %s must be lower than %s" % (name, tx, max_value))

def random_range_percents(count):
    nums = [random.randint(1, 10) for i in range(count)]
    return [Fraction(num, sum(nums)) for num in nums]

def random_ranges(min_vsp, max_vsp, count):
    assert int(count) == count and count >= 1
    assert min_vsp >= 0 and max_vsp >= 0
    assert min_vsp <= max_vsp

    ranges = []
    range_percents = random_range_percents(count)

    for percent in range_percents:
        num  = round(random.uniform(min_vsp, max_vsp), 2)

        if num > (min_vsp + max_vsp) / 2:
            high_lim = num
            low_lim = round(random.uniform(min_vsp, high_lim), 2)
        else:
            low_lim = num
            high_lim = round(random.uniform(low_lim, max_vsp), 2)

        ranges.append(VehicleSpeedRange(low_lim=low_lim, high_lim=high_lim, percent=percent))

    return ranges


class RobustnessVisualizer:
    def __init__(self, time, vsp, stwa, params):
        self.Time = time
        self.Vsp = vsp
        self.Stwa = stwa
        self.params = params

    def maneuver_plot(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, layout="tight")
        ax_titles = {ax1: "Vehicle speed [km/h]", ax2: "Steering wheel angle [deg]", ax3: "Vehicle speed acceleration [(km/h)/s]", ax4: "Steering wheel angle speed [deg/s]"}

        ax1.plot(self.Time, self.Vsp)
        ax2.plot(self.Time, self.Stwa)
        ax3.plot(self.Time, numderive(self.Vsp, self.Time))
        ax4.plot(self.Time, numderive(self.Stwa, self.Time))

        start_percent = 0
        end_percent = 0

        max_stwa = np.array(get_max_stwa(self.Vsp, self.params.min_stwa, self.params.endpos, self.params.endpos_till_vsp))

        for r in self.params.vsp_ranges:
            end_percent += len(self.Time) * r.percent

            win = [True if i >= start_percent and i <= end_percent else False for i in range(len(self.Time))]
            ax1.fill_between(self.Time, r.low_lim, r.high_lim, where=win, alpha=0.2, label="%3d - %3d km/h" % (r.low_lim, r.high_lim))
            ax1.legend(loc="upper right")

            ax2.fill_between(self.Time, -1*max_stwa, max_stwa, where=win, alpha=0.2, label="%3d - %3d km/h" % (r.low_lim, r.high_lim))
            ax2.legend(loc="upper right")

            start_percent = end_percent

        for ax in (ax1, ax2, ax3, ax4):
            ax.set_title(ax_titles[ax])
            ax.grid()

        plt.setp((ax3, ax4), xlabel="Time [s]")
        fig.show()

    def max_stwa_plot(self):
        perlin = Perlin()
        noise = perlin.noise(len(self.Time), 0.015, 5, 1, int(time.time()))

        vsp = np.linspace(0, 200, len(self.Time))
        max_stwa = np.array(get_max_stwa(vsp, self.params.min_stwa, self.params.endpos, self.params.endpos_till_vsp))

        scaled_noise = scale_up_noise(noise) * max_stwa

        plt.figure()
        plt.plot(vsp, max_stwa, color="blue", label="Maximum possible steering wheel angle [deg]")
        plt.plot(vsp, -max_stwa, color="blue")
        plt.plot(vsp, scaled_noise, color="green", label="Steering wheel angle [deg]")
        plt.xlabel("Vehicle speed [km/h]")
        plt.title("Maximum steering wheel angle limit defined by vehicle speed")
        plt.legend()
        plt.grid()
        plt.show()
