from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = Path("D:\\work\\YuCai\\projects\\data\\5031581_31.28_121.47_2020.csv")


def climate_loader(dataset_path: Path) -> pd.DataFrame:
    data = pd.read_csv(dataset_path,
                       header=2,
                       parse_dates={"Datetime": ["Year", "Month", "Day", "Hour", "Minute"]},
                       date_parser=lambda x: datetime.strptime(x, "%Y %m %d %H %M"))
    data.set_index("Datetime", inplace=True)
    return data


if __name__ == '__main__':
    # load the NREL climate dataset
    d = climate_loader(DATASET_PATH)
    print(d.head())
    da = d.resample('W').mean()
    # plot the solar irradiation as an example
    plt.plot(da.index, da["DHI"], label="DHI", linewidth=1)
    plt.plot(da.index, da["DNI"], label="DNI", linewidth=1)
    plt.plot(da.index, da["GHI"], label="GHI", linewidth=1)
    plt.title("Average Solar Irradiation Per Week in Shanghai in 2020")
    plt.xlabel("Date")
    plt.ylabel("Average Solar Irradiarion Per Week (W/m²)")
    plt.legend()
    plt.grid()
    plt.show()
