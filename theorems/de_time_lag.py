import numpy as np
import matplotlib.pyplot as plt

N = 256
dt = 0.1
y0 = 0.0
y1 = np.sin(dt)


def fig(y: np.ndarray, title: str) -> None:
    x = dt * np.arange(N)
    plt.plot(x, y, marker='.', linestyle='', zorder=0)
    plt.plot(x, np.sin(x), linestyle='--', zorder=1)
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("y(t)")
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    y = np.empty(N)
    # forward
    y[0] = y0
    y[1] = y1
    for i in range(1, N - 1):
        y[i + 1] = 2 * y[i] - (1 + dt * dt) * y[i - 1]
    fig(y, "Forward Formula")
    # backward
    y[0] = y0
    y[1] = y1
    for i in range(1, N - 1):
        y[i + 1] = (2 * y[i] - y[i - 1]) / (1 + dt * dt)
    fig(y, "Backward Formula")
    # centered
    y[0] = y0
    y[1] = y1
    for i in range(1, N - 1):
        y[i + 1] = (2 - dt * dt) * y[i] - y[i - 1]
    fig(y, "Centered Formula")
