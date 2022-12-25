from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from solar_model.climate import climate_loader
from solar_model.solar_panel import RealPanel, virtual_scene

plt.rcParams['figure.figsize'] = (8.0, 12.0)

DATASET_PATH = Path("D:\\work\\YuCai\\projects\\data\\5031581_31.28_121.47_2020.csv")
LATITUDE = 31.28
CLEAN_PLAN = [pd.Timestamp("2020-04-01, 10:30:00"), pd.Timestamp("2020-09-01, 13:50:00"), pd.Timestamp("2020-12-20, 09:45:00")]
REPAIR_PLAN = [pd.Timestamp("2020-05-20, 14:20:00"), pd.Timestamp("2020-11-10, 09:20:00")]

if __name__ == '__main__':
    climate_data = climate_loader(DATASET_PATH)
    climate_data = virtual_scene(climate_data, REPAIR_PLAN, CLEAN_PLAN)
    panel = RealPanel()
    result = panel.simulate(climate_data, LATITUDE, expand_frame=True)
    ideal_power = result['Power']
    real_power = result['Real Power']

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
    ideal_power_res = ideal_power.resample('W').mean()
    real_power_res = real_power.resample('W').mean()
    l21 = ax2.plot(ideal_power_res.index, ideal_power_res.values, label="Ideal Output Power")
    l22 = ax2.plot(real_power_res.index, real_power_res.values, label="Real Output Power")
    # plot settings
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Average Output Power Per Week (W)")
    ax2.grid()
    ax2.legend(loc='upper right')
    plt.show()
