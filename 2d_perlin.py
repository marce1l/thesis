import math
import numpy as np
import matplotlib.pyplot as plt

gradX = [[np.random.rand() for i in range(256)] for i in range(256)]
gradY = [[np.random.rand() for i in range(256)] for i in range(256)]

step  = 0.01
width = 10000

def toUnitVector():
    for i in range(256):
        for j in range(256):
            c = math.sqrt(pow(gradX[i][j], 2) + pow(gradY[i][j], 2))
            gradX[i][j] = gradX[i][j] / c
            gradY[i][j] = gradY[i][j] / c

def perlin(x, y):
    xi = int(x)
    yi = int(y)
    # m = 1 / step
    
    # if x % m == 0:
        # xi = x / m
    # else:
        # xi = x / m
    # if y % m == 0:
        # yi = y / m
    # else:
        # yi = y % m
    
    P1 = (x * gradX[xi][yi]) + (y * gradY[xi][yi])
    P2 = (x * gradX[xi+1][yi]) + (y * gradY[xi+1][yi])
    P3 = (x * gradX[xi][yi-1]) + (y * gradY[xi][yi-1])
    P4 = (x * gradX[xi+1][yi-1]) + (y * gradY[xi+1][yi-1])

    points = [
            (xi, yi, P1),
            (xi+1, yi, P2),
            (xi, yi-1, P3),
            (xi+1, yi-1, P4)]
    
    return bilinear_interpolation(x, y, points)

def bilinear_interpolation(x, y, points):
    points = sorted(points)
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def makeNoise():
    toUnitVector()

    noise = np.empty((width, width))
    
    for x in range(width):
        for y in range(width):
            noise[x][y] = perlin(x*step, y*step)
    
    Time = [i for i in range(width)]
    
    plt.plot(Time, noise[0])
    plt.show()

makeNoise()