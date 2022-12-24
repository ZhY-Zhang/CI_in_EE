from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
import dowhy.gcm as gcm

# parameters
# file
FILE_PATH = Path("D:\\work\\YuCai\\projects\\CI_in_EE\\stuff\\generator")
MERGE = True
# graph
DAG_TL = nx.DiGraph([('Y2', 'Y'), ('Y1', 'Y')])
NODES_TL = ["Y", "Y1", "Y2"]
DAG_DE = nx.DiGraph([('Y1', 'Y'), ('Y2', 'Y'), ('Z1', 'Z'), ('Z2', 'Z'), ('Y1', 'Z'), ('Z1', 'Y')])
NODES_DE = ["Y", "Y1", "Y2", "Z", "Z1", "Z2"]
# data
OPERATION: np.ndarray = np.array([[2, 0.5, 10]])
SAMPLE_PERIOD: float = 0.01
NOISE_SCALE: float = 0.02
AVERAGE_NUM: int = 8


def signal_generator(operation: np.ndarray, sample_period: float = SAMPLE_PERIOD, noise_scale: float = 0.05) -> pd.DataFrame:
    num = np.sum(operation[:, 2] // sample_period, dtype=int)
    a0 = np.empty(num)
    f0 = np.empty(num)
    y0 = np.empty(num)
    n0 = 0
    p0 = 0
    for a, f, t in operation:
        n = int(t // sample_period)
        p = 2 * np.pi * f * np.arange(n) * sample_period + p0
        a0[n0:n0 + n] = a
        f0[n0:n0 + n] = f
        y0[n0:n0 + n] = a * np.sin(p)
        n0 += n
        p0 += 2 * np.pi * t / f
    t0 = np.arange(num) * sample_period
    y0 += np.random.normal(0.0, noise_scale, num)
    df = pd.DataFrame({'A': a0, 'F': f0, 'Y': y0, 'T': t0})
    return df


def data_expander(df: pd.DataFrame) -> pd.DataFrame:
    df['Y1'] = df['Y'].shift(1)
    df['Y2'] = df['Y'].shift(2)
    df['Z'] = df['Y'].diff(2).shift(-1) / (2 * SAMPLE_PERIOD)
    df['Z1'] = df['Z'].shift(1)
    df['Z2'] = df['Z'].shift(2)
    df['A1_inv'] = 1 / df['A'].shift(1)
    return df


def plotter(title: str) -> None:
    num = np.sum(OPERATION[:, 2] // SAMPLE_PERIOD, dtype=int)
    time = np.arange(num) * SAMPLE_PERIOD
    for p in FILE_PATH.glob("[1234567890.]*.txt"):
        a = np.loadtxt(str(p))
        plt.plot(time, a, label="ns: {}".format(p.stem))
    ideal_data = signal_generator(OPERATION, noise_scale=0.0)
    plt.plot(time, ideal_data['Y'], label="ideal data", linewidth=2, linestyle='--', zorder=0)
    plt.title(title)
    plt.xlabel("time (s)")
    plt.ylabel("voltage (V)")
    plt.grid()
    plt.legend()
    plt.show()


def generated_generator() -> None:
    num = np.sum(OPERATION[:, 2] // SAMPLE_PERIOD, dtype=int)
    amplitude, frequency = OPERATION[0, 0], OPERATION[0, 1]
    # generate training data
    data = signal_generator(OPERATION, noise_scale=NOISE_SCALE)
    data = data_expander(data)
    data.dropna(axis=0, how='any', inplace=True)
    # train dowhy
    # TODO: change comment when changing modes
    """
    # Time-Lag Model
    causal_model = gcm.StructuralCausalModel(DAG_TL)
    gcm.auto.assign_causal_mechanisms(causal_model, data)
    gcm.fit(causal_model, data)
    # save the result
    result = np.empty(num)
    y1 = amplitude * np.sin(-2 * np.pi * frequency * SAMPLE_PERIOD)
    y2 = amplitude * np.sin(-4 * np.pi * frequency * SAMPLE_PERIOD)
    for i in range(num):
        s = gcm.interventional_samples(causal_model, {'Y1': lambda _: y1, 'Y2': lambda _: y2}, num_samples_to_draw=AVERAGE_NUM)
        y = np.mean(s['Y'])
        result[i] = y
        y2 = y1
        y1 = y
    np.savetxt(str(FILE_PATH.joinpath("{}.txt".format(NOISE_SCALE))), result)
    """
    # Derivative Model
    causal_model = gcm.StructuralCausalModel(DAG_DE)
    gcm.auto.assign_causal_mechanisms(causal_model, data)
    gcm.fit(causal_model, data)
    # save the result
    result = np.empty(num)
    y1 = amplitude * np.sin(-2 * np.pi * frequency * SAMPLE_PERIOD)
    y2 = amplitude * np.sin(-4 * np.pi * frequency * SAMPLE_PERIOD)
    z1 = amplitude * 2 * np.pi * frequency * np.cos(-2 * np.pi * frequency * SAMPLE_PERIOD)
    z2 = amplitude * 2 * np.pi * frequency * np.cos(-4 * np.pi * frequency * SAMPLE_PERIOD)
    for i in range(num):
        s = gcm.interventional_samples(causal_model, {
            'Y1': lambda _: y1,
            'Y2': lambda _: y2,
            'Z1': lambda _: z1,
            'Z2': lambda _: z2
        },
                                       num_samples_to_draw=AVERAGE_NUM)
        y, z = np.mean(s['Y']), np.mean(s['Z'])
        result[i] = y
        y2 = y1
        y1 = y
        z2 = z1
        z1 = z
    np.savetxt(str(FILE_PATH.joinpath("{}.txt".format(NOISE_SCALE))), result)


if __name__ == '__main__':
    if MERGE:
        # TODO: change title when changing modes
        # plotter("Training Results of the Time-Lag Model (Exact Initial Values)")
        plotter("Training Results of the Derivative Model (Exact Initial Values)")
    else:
        generated_generator()
