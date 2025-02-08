import numpy as np
import time

class Perlin():
    _gradX = [[]]
    _gradY = [[]]

    def _perlin(self, x, y):
        # grid top left index
        xi = int(x)
        yi = int(y)

        # top left distance vector
        xd = x - xi
        yd = y - yi

        # top left, top right, bottom left, bottom right dot product
        p00 = xd     * self._gradX[yi][xi]     + yd     * self._gradY[yi][xi]
        p01 = (xd-1) * self._gradX[yi][xi+1]   + yd     * self._gradY[yi][xi+1]
        p10 = xd     * self._gradX[yi+1][xi]   + (yd-1) * self._gradY[yi+1][xi]
        p11 = (xd-1) * self._gradX[yi+1][xi+1] + (yd-1) * self._gradY[yi+1][xi+1]

        xf = self._fade(xd)
        yf = self._fade(yd)

        a = self._interpolate(p00, p10, yf)
        b = self._interpolate(p01, p11, yf)
        return self._interpolate(a, b, xf)

    # noise smoothing
    def _fade(self, f):
        return 6 * f**5 - 15 * f**4 + 10 * f**3

    def _interpolate(self, a, b, x):
        return a + x * (b - a)

    def _generate_noise(self, length, step, count):
        skip_factor = int(length/count)
        arr = np.empty((count, length))

        for y in range(0, length, skip_factor):
            for x in range(length):
                arr[int(y/skip_factor)][x] = self._perlin(x * step, y * step)

        return arr

    def _generate_octaves(self, length, step, octave, count, divisor=2):
        amplitude = 1.0
        octaves = []

        for _ in range(octave):
            octaves.append(self._generate_noise(length, step, count) * amplitude)
            amplitude /= divisor
            step      *= divisor

        # combine octaves
        combined = np.add.reduce(octaves)

        if combined.shape[0] == 1:
            return combined[0]

        return combined

    def _create_gradient_vectors(self, size, count, seed):
        rng = np.random.Generator(np.random.PCG64(seed=int(seed)))

        # don't generate the full y axis when only 2 is needed
        # can't use this approach if the noise count is higher than 1
        # because the noises generated from adjecent indexes would be too similar
        ysize = count+1 if count == 1 else size
        xsize = size

        gradX = rng.random((ysize, xsize)) * 2 - 1
        gradY = rng.random((ysize, xsize)) * 2 - 1

        # convert to unit vectors
        c = np.sqrt(gradX**2 + gradY**2)
        gradX /= c
        gradY /= c

        return (gradX, gradY)

    def _set_gradient_size(self, step, length, octave):
        return int(np.ceil(step*length*2**(octave-1))+1)

    def noise(self, length, step, octave, count, seed=time.time()):
        if step >= 1:
            raise ValueError("Step size (%s) has to be lower than 1" % step)

        gradient_size = self._set_gradient_size(step, length, octave)
        self._gradX, self._gradY = self._create_gradient_vectors(gradient_size, count, seed)

        return self._generate_octaves(length, step, octave, count)
