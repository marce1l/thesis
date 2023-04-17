import math
import numpy as np

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
    pass

def makeNoise():
    noise = [[]]

    for x in range(width*step)):
        for y in range(width*step):
            noise[x][y] = perlin(x, y)