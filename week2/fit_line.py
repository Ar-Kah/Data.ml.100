""" dataml100 toisen viikon tehävä"""
import numpy as np
import matplotlib.pyplot as plt


def my_linfit(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)

    # slope (a)
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    # intercept (b)
    b = (sum_y - a * sum_x) / n

    return a, b


def main():
    x = np.random.uniform(-2, 5, 10)
    y = np.random.uniform(0, 3, 10)
    a, b = my_linfit(x, y)
    plt.plot(x, y, 'kx')
    xp = np.arange(-2 ,5, 0.1)
    plt.plot(xp, a*xp+b, 'r-')
    print(f"My fit: a={a} and b = {b}")
    plt.show()


if __name__ == '__main__':
    main()
