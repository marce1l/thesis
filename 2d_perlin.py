import math
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6187254)

gradX = [[np.random.rand() for i in range(256)] for i in range(256)]
gradY = [[np.random.rand() for i in range(256)] for i in range(256)]

step  = 0.01
width = 1000

def toUnitVector():
    for i in range(256):
        for j in range(256):
            c = math.sqrt(pow(gradX[i][j], 2) + pow(gradY[i][j], 2))
            gradX[i][j] = gradX[i][j] / c
            gradY[i][j] = gradY[i][j] / c

def perlin(x, y):
    xi = int(x)
    yi = int(y)
    
    # print("\nx: %s y: %s" % (x,y))
    # print("xi: %s yi: %s" % (xi,yi))
    
    xd = x - xi
    yd = y - yi
    
    # print("xd: %s yd: %s" % (xd,yd))
    
    p00 = xd     * gradX[xi][yi]     + yd     * gradY[xi][yi]
    p01 = (xd-1) * gradX[xi+1][yi]   + yd     * gradY[xi+1][yi]
    p10 = xd     * gradX[xi][yi+1]   + (yd-1) * gradY[xi][yi+1]
    p11 = (xd-1) * gradX[xi+1][yi+1] + (yd-1) * gradY[xi+1][yi+1]

    xf = fade(xd)
    yf = fade(yd)
    
    x1 = interpolation(p00, p10, yf)
    x2 = interpolation(p01, p11, yf)
    
    # print("p00: %s p01: %s p10: %s p11: %s" % (p00, p01, p10, p11))
    
    return interpolation(x1, x2, xf)

def interpolation(a, b, x):
    return a + x * (b - a)
    
def fade(f):
    return 6 * f**5 - 15 * f**4 + 10 * f**3

def makeNoise():
    toUnitVector()

    noise = np.empty((width, width))
    
    for y in range(width):
        for x in range(width):
            noise[y][x] = perlin(x*step, y*step)
    
    Time = [i for i in range(width)]
    
    plt.plot(Time, noise[0])
    plt.show()
    
    plt.imshow(noise, cmap='gray')
    plt.show()

makeNoise()