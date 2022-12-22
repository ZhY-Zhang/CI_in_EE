from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dowhy import CausalModel

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

# parameters
DAG_TL_FILE = Path("theorems/dag_time_lag.txt")
NODES_TL = ["A", "F", "Y", "Y1", "Y2"]
DAG_DE_FILE = Path("theorems/dag_derivative.txt")
NODES_DE = ["A", "F", "Y", "Y1", "Y2", "Z", "Z1", "Z2"]

DAG_FILE = DAG_DE_FILE
NODES = NODES_DE
TREATMENT = "F"
OUTCOME = "Y"


def signal_generator(operation: np.ndarray, sample_period: float = 0.01, noise_scale: float = 0.05) -> pd.DataFrame:
    N = np.sum(operation[:, 2] // sample_period, dtype=int)
    a0 = np.empty(N)
    f0 = np.empty(N)
    y0 = np.empty(N)
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
    t0 = np.arange(N) * sample_period
    y0 += np.random.normal(0.0, noise_scale, N)
    df = pd.DataFrame({'A': a0, 'F': f0, 'Y': y0, 'T': t0})
    return df


def data_expander(df: pd.DataFrame) -> pd.DataFrame:
    df['Y1'] = df['Y'].shift(1)
    df['Y2'] = df['Y'].shift(2)
    df['Z'] = df['Y'].diff(2).shift(-1) / 2
    df['Z1'] = df['Z'].shift(1)
    df['Z2'] = df['Z'].shift(2)
    return df


if __name__ == '__main__':
    # A. Load Causal Graph.
    with open(DAG_FILE) as f:
        s = f.readlines()
    s1 = ";".join(filter(lambda x: '>' in x, s))
    s1 = s1.replace("\n", "")
    GML_GRAPH = "digraph {{{}}}".format(s1)
    print("Loaded causal graph succesfully.")

    # B. Generate Data.
    df = signal_generator(np.array([[1, 1, 2.5], [1, 3, 2.4], [0.7, 5, 1], [2, 0.8, 4], [0.6, 2, 6]]))
    df = data_expander(df)
    df.dropna(axis=0, how='any', inplace=True)
    print(df)

    # C. DoWhy
    # I. Create a causal model from the data and given graph.
    model = CausalModel(data=df, treatment=TREATMENT, outcome=OUTCOME, graph=GML_GRAPH)
    model.view_model()

    # II. Identify causal effect and return target estimands.
    identified_estimand = model.identify_effect()
    print(identified_estimand)

    # III. Estimate the target estimand using a statistical method.
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    print(estimate)

    # IV. Refute the obtained estimate using multiple robustness checks.
    refute_results = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
    print(refute_results)
