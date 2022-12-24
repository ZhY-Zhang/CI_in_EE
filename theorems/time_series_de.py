import numpy as np
import matplotlib.pyplot as plt

# u = Au
N = 128
A = np.array([[0, -1], [1, 0]])
dt = 0.1
dy0 = 1.
y0 = 0.
dy1 = 0.995
y1 = 0.100


def fig(u: np.ndarray, title: str) -> None:
    plt.plot(u[:, 1], u[:, 0], zorder=0)
    plt.scatter(u[:, 1], u[:, 0], c=range(N), zorder=1)
    plt.grid()
    plt.axis("equal")
    plt.xlabel("y(t)")
    plt.ylabel("derivative of y(t)")
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    u = np.empty([N, 2])
    # forward
    F = np.eye(2) + dt * A
    u[0] = [dy0, y0]
    for i in range(N - 1):
        u[i + 1] = np.matmul(F, u[i])
    fig(u, "Forward Equation Sprials Out")
    # backward
    B = np.linalg.inv(np.eye(2) - dt * A)
    u[0] = [dy0, y0]
    for i in range(N - 1):
        u[i + 1] = np.matmul(B, u[i])
    fig(u, "Backward Equation Sprials In")
    # centered
    C = 2 * dt * A
    u[0] = [dy0, y0]
    u[1] = [dy1, y1]
    for i in range(1, N - 1):
        u[i + 1] = u[i - 1] + np.matmul(C, u[i])
    fig(u, "Centered Equation Stays On The Circle")
