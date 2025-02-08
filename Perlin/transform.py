import numpy as np
import logging

logger = logging.getLogger("perlin_robustness")

class TransitionOverlapError(Exception):
    pass

def numderive(y_values, x_values):
    ''' Calculate differential signal as diff(y)/diff(x) '''
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    derivate = np.diff(y_values) / np.asarray( np.diff(x_values), float)
    derivate = np.append(derivate, derivate[-1])       # array size compensation due to diff
    return derivate

def transition_between_signals(signals, ranges, dt, acc_rate, deacc_rate):
    '''
    Transitions between signals by
        - calculating the furthest differences (always assuming the worst case) between the signals ranges
        - for each signal:
            1. calculate the transition fade length (index count) using the difference, aiming for acc_rate or deacc_rate
            2. create a sinus transition fade (0..1) with the calculated transition fade length
            3. create a mask with values ranging from 0 to 1 which determines where and what amount of the signal should be used
               (half of the fade length will come off the current and the other half from the next signal)
        - multiply signals by their masks and add them together

    TransitionOverlapError can occur if
        The (used part of the) currently processed signal's length is too small and the start of the transition overlaps with the previous transition's end.
        or
        The currently processed signal + transition's length is higher than the desired final signal's length
    '''
    percents = [r.percent for r in ranges]
    assert sum(percents) == 1.0

    diffs = calc_furthest_diffs(ranges)

    signal_length = len(signals[0])

    percent = 0
    prev_trans_end = 0

    # masks initialized with 0-s
    masks = [np.zeros(signal_length) for _ in range(len(signals))]

    for i in range(len(signals)-1):
        diff = diffs[i]

        # fade_length/2 operation should always result in an even number
        fade_length = get_fade_length(diff, acc_rate, deacc_rate, dt)
        assert fade_length % 2 == 0

        fade = (np.sin(np.linspace(3*np.pi/2, 5*np.pi/2, fade_length))+1)/2
        percent += percents[i]

        # signal end in index minus it's part of the transition
        sig_end = int(signal_length * percent) - int(fade_length/2)

        logger.debug("transition_between_signals\t%2d  fade_length: %4d\tprev_trans_end: %4d\tsig_end: %4d\tdiff: %s" % (i, fade_length, prev_trans_end, sig_end, diff))

        if prev_trans_end >= sig_end or (sig_end + fade_length) > signal_length:
            logger.error("TransitionOverlapError\t\tprev_trans_end >= sig_end: %s\t\t(sig_end + fade_length) > signal_length: %s" % (prev_trans_end >= sig_end, (sig_end + fade_length) > signal_length))
            raise TransitionOverlapError("Transitions between vehicle speeds overlap! Adjust the vsp_ranges or the test_length")

        # set the mask indexes after the previous and before current transition to 1
        masks[i][prev_trans_end:sig_end] = masks[i][prev_trans_end:sig_end] + 1

        # set mask indexes for the current signal's transition end to 1-fade (1..0)
        masks[i][sig_end:sig_end+fade_length] = masks[i][sig_end:sig_end+fade_length] + (1-fade)

        # set mask indexes for the next signal's transition start to (0..1)
        masks[i+1][sig_end:sig_end+fade_length] = masks[i+1][sig_end:sig_end+fade_length] + fade

        prev_trans_end = sig_end + fade_length

    # set mask indexes after the last transition to 1
    masks[-1][prev_trans_end:] = masks[-1][prev_trans_end:] + 1

    combined = sum(signals[i] * masks[i] for i in range(len(signals)))
    return combined

def get_fade_length(diff, acc_rate, deacc_rate, dt):
    '''
    Calculates the length (in index count) of the transition fade, aiming for the acc_rate or deacc_rate based on the difference's sign

    Must return an even number to avoid indexing errors
    '''
    if diff > 0:
        length = (abs(diff) / acc_rate) / dt
    elif diff < 0:
        length = (abs(diff) / deacc_rate) / dt
    else:
        length = 1

    # fade length compensation due to sinusoidal fade function
    length = length * 1.6

    # round up to the nearest even number to avoid indexing errors when dividing by 2
    return int(np.ceil(length / 2) * 2)

def calc_furthest_diffs(vsp_ranges):
    '''
    Calculates the furthest difference between vehicle speed ranges, meaning the highest difference between two vehicle speed's low and high limits
    '''
    diffs = []

    for i in range(len(vsp_ranges)-1):
        x = vsp_ranges[i+1].high_lim - vsp_ranges[i].low_lim
        y = vsp_ranges[i+1].low_lim - vsp_ranges[i].high_lim

        diff = x if abs(x) > abs(y) else y
        diffs.append(diff)

    return diffs

def scale_up_noise(noise):
    '''
    Scales up noise so it's values are closer to a -1..1 range
    '''
    max_value = abs(max(noise))
    min_value = abs(min(noise))

    if max_value > min_value:
        return noise * (1/max_value)
    else:
        return noise * (1/min_value)

def offset_to_range(noise, min_range, max_range):
    '''
    Offsets noise to a specified range
    '''
    assert min_range <= max_range

    if min_range == max_range:
        return noise * min_range

    difference = max_range - min_range
    offset = min_range

    noise = ((noise+1)/2)*difference+offset

    max_value = max(noise)
    min_value = min(noise)

    if max_range-max_value < min_value-min_range:
        return noise - (min_value-min_range)/2
    else:
        return noise + (max_range-max_value)/2

def get_max_stwa(vsp, min_stwa, endpos, endpos_till_vsp):
    '''
    Calculates the maximum steering wheel angle for each value in a given vehicle speed signal
    '''
    assert min_stwa >= 0
    assert endpos_till_vsp >= 0

    x_offset = get_vsp_offset(endpos, endpos_till_vsp, min_stwa)
    return [get_stwa_from_vsp(value, endpos, min_stwa, x_offset) for value in vsp]

def get_stwa_from_vsp(vsp, ep, yoffs, xoffs):
    '''
    The exponential function which calculates the maximum steering wheel angle for a vehicle speed value
    '''
    return min(np.exp(2-(vsp-xoffs)/25)*100+yoffs, ep)

def get_vsp_offset(ep, vsp_ep, yoffs):
    '''
    Calculates the vehicle speed offset, meaning the value with which the vehicle speed should be offseted with,
    to get a x offset (endpos_till_vsp) on the exponential function
    '''
    return vsp_ep-25*(2-np.log((ep-yoffs)/100))