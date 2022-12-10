from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solar_model.climate import climate_loader
from solar_model.solar_panel import IdealPanel

plt.rcParams['figure.figsize'] = (8.0, 12.0)

DATASET_PATH = Path("D:\\work\\YuCai\\projects\\data\\5031581_31.28_121.47_2020.csv")
LATITUDE = 31.28
FACTORS = ['Power', 'GHI', 'Temperature']

if __name__ == '__main__':
    climate_data = climate_loader(DATASET_PATH)
    panel = IdealPanel()
    result = panel.simulate(climate_data, LATITUDE, expand_frame=True)
    power = result['Power']

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Solar Irradiation and Output Power Per Week in Shanghai in 2020")
    # plot the solar irradiation
    climate_res = climate_data.resample('W').mean()
    l11 = ax1.plot(climate_res.index, climate_res["DHI"], label="DHI", linewidth=1)
    l12 = ax1.plot(climate_res.index, climate_res["DNI"], label="DNI", linewidth=1)
    l13 = ax1.plot(climate_res.index, climate_res["GHI"], label="GHI", linewidth=1)
    ax1.set_ylabel("Average Solar Irradiation Per Week (W/m²)")
    ax1.grid()
    ax1.legend()
    # plot the power
    power_res = power.resample('W').mean()
    l21 = ax2.plot(power_res.index, power_res.values, label="Output Power")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Average Output Power Per Week (W)")
    ax2.grid()
    ax2.legend()

    plt.show()
