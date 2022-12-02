from typing import List

import numpy as np
import matplotlib.pyplot as plt

N = 65536


def root_node(size: int = 1) -> np.ndarray:
    x = np.random.normal(0.0, 1.0, size)
    return x


def child_node(xs: List[np.ndarray], gains: List[float], disturb: float) -> np.ndarray:
    y = disturb * np.random.normal(0.0, 1.0, xs[0].size)
    for x, g in zip(xs, gains):
        y += g * x
    return y


def extracter(x1: np.ndarray, x2: np.ndarray, c: np.ndarray, c_value: float, step: float = 0.025) -> None:
    ext = np.logical_and(c >= c_value - step, c <= c_value + step)
    y1 = np.extract(ext, x1)
    y2 = np.extract(ext, x2)
    return y1, y2


def overall_output(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    fig = plt.figure(figsize=(8, 8))
    # plot the 3-D joint distribution
    # perspect A
    ax1 = fig.add_subplot(221, projection='3d')
    ext = np.where(np.random.random(x.size) < 1024 / x.size, True, False)
    x1, y1, z1 = x[ext], y[ext], z[ext]
    ax1.scatter(x1, y1, z1, s=1)
    ax1.set_title("Joint Distribution", fontsize=10)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    plt.grid(True)
    # perspect B
    ax1 = fig.add_subplot(223, projection='3d')
    ax1.scatter(x1, y1, z1, s=1)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=45, azim=120)
    plt.grid(True)
    # plot the 2-D conditional distributions
    # Condition: X = 1.0
    ax2 = fig.add_subplot(322)
    y1, z1 = extracter(y, z, x, 1.0)
    ax2.scatter(y1, z1, s=2)
    ax2.set_title("Conditional distribution of Y and Z under X = 1.0", fontsize=10)
    ax2.set_xlabel("Y")
    ax2.set_ylabel("Z")
    ax2.axis('equal')
    plt.grid(True)
    # Condition: Y = 1.0
    ax3 = fig.add_subplot(324)
    x1, z1 = extracter(x, z, y, 1.0)
    ax3.scatter(x1, z1, s=2)
    ax3.set_title("Conditional distribution of X and Z under Y = 1.0", fontsize=10)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.axis('equal')
    plt.grid(True)
    # Condition: Z = 1.0
    ax4 = fig.add_subplot(326)
    x1, y1 = extracter(x, y, z, 1.0)
    ax4.scatter(x1, y1, s=2)
    ax4.set_title("Conditional distribution of X and Y under Z = 1.0", fontsize=10)
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.axis('equal')
    plt.grid(True)

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.45, hspace=0.35)
    plt.show()


if __name__ == '__main__':
    # Chain: X -> Y -> Z
    X = root_node(N)
    Y = child_node([X], [1.0], 0.1)
    Z = child_node([Y], [1.0], 0.1)
    # output results
    print("Chain: X -> Y -> Z")
    overall_output(X, Y, Z)

    # Fork: X <- Y -> Z
    Y = root_node(N)
    X = child_node([Y], [1.0], 0.1)
    Z = child_node([Y], [1.0], 0.1)
    # output results
    print("Fork: X <- Y -> Z")
    overall_output(X, Y, Z)

    # Collider: X -> Y <- Z
    X = root_node(N)
    Z = root_node(N)
    Y = child_node([X, Z], [1.0, 1.0], 0.1)
    # output results
    print("Collider: X -> Y <- Z")
    overall_output(X, Y, Z)
