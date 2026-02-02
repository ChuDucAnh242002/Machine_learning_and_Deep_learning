import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

def my_linfit(x, y, n):
    a_quotient = n * np.sum(x * y) - np.sum(x) * np.sum(y)
    a_divisor = n * np.sum(x ** 2) - np.sum(x) ** 2
    a = a_quotient / a_divisor
    b = (np.sum(y) - a * np.sum(x)) / n
    return a, b

def plot_line(a, b):
    xp = np.arange(-2, 5, 0.1)
    yp = a*xp + b
    plt.plot(xp, yp, 'r-')
    print(f"My fit: a ={a} and b={b}")

def on_click(event):
    if event.button == MouseButton.LEFT:
        x = event.xdata
        y = event.ydata
        plt.plot(x, y, 'kx')
        x_points.append(x)
        y_points.append(y)
    elif event.button == MouseButton.RIGHT:
        if (x_points == [] or y_points == []):
            return
        x_array = np.array(x_points)
        y_array = np.array(y_points)
        n = np.size(x_array)
        a, b = my_linfit(x_array, y_array, n)
        plot_line(a, b)
        plt.disconnect(binding_id)

plt.ion()
plt.axis([-2, 5, 0, 3])
x_points = []
y_points = []

binding_id = plt.connect('button_press_event', on_click)

plt.title("Linear model fit with N > 2 training points")
plt.show(block = True)
