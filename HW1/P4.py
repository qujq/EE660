import numpy as np
import math
import matplotlib.pyplot as plt

def plot1(x):
    y_value = []
    for s in x:
        y_value.append(math.log(1 + math.exp(-s)))
    y = np.array(y_value)
    plt.plot(x, y)
    plt.show()

def plot2(x):
    y_value = []
    for s in x:
        if s <= 0:
            y_value.append(-s)
        else:
             y_value.append(0)
    y = np.array(y_value)
    plt.plot(x, y)
    plt.show()

def plot3(x):
    y_value = []
    for s in x:
        y_value.append((1/2) * -2 * s)
    y = np.array(y_value)
    plt.plot(x, y)
    plt.show()


x = np.linspace(-10, 10, 100)
plot1(x)
x = np.linspace(-2, 2, 100)
plot1(x)

x = np.linspace(-10, 10, 100)
plot2(x)
x = np.linspace(-2, 2, 100)
plot2(x)

x = np.linspace(-10, 10, 100)
plot3(x)
x = np.linspace(-2, 2, 100)
plot3(x)
