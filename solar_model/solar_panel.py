from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class Battery:

    def __init__(self) -> None:
        # standard params
        self.size = 1.638
        self.U_oc_st = 40.01
        self.U_m_st = 32.68
        self.I_sc_st = 9.87
        self.I_m_st = 9.34
        self.eta_st = 18.3
        self.T_ref = 25.0
        self.S_ref = 1000.0
        # NOCT = Nominal Operating Cell Temperature
        self.NOCT = 44.0
        # modifying params
        self.a = 0.0025
        self.b = 0.0005
        self.c = 0.00288
        # actual params
        self.U_oc = self.U_oc_st
        self.U_m = self.U_m_st
        self.I_sc = self.I_sc_st
        self.I_m = self.I_m_st
        self.__eta = self.U_m * self.I_m / self.size / self.S_ref

    @property
    def eta(self) -> float:
        return self.__eta

    def update_env(self, irradiance: float, temperature: float) -> None:
        del_T = temperature - self.T_ref
        del_S = irradiance - self.S_ref
        gain_I = irradiance * (1 + self.a * del_T) / self.S_ref
        gain_U = (1 - self.c * del_T) * np.log(np.e + self.b * del_S)
        self.I_sc = self.I_sc_st * gain_I
        self.U_oc = self.U_oc_st * gain_U
        self.I_m = self.I_m_st * gain_I
        self.U_m = self.U_m_st * gain_U
        self.__eta = self.U_m * self.I_m / self.size / irradiance

    def plot_IU(self, u_range: np.ndarray = np.linspace(0, 40, 81), label="Solar") -> None:
        """
        Please use "plt.show()" to view the figure.
        """
        c2 = (self.U_m / self.U_oc - 1) / np.log(1 - self.I_m / self.I_sc)
        c1 = (1 - self.I_m / self.I_sc) * np.exp(-self.U_m / c2 / self.U_oc)
        i = self.I_sc * (1 - c1 * (np.exp(u_range / c2 / self.U_oc) - 1))
        flt = i >= 0
        plt.plot(u_range[flt], i[flt], label=label)
        plt.scatter(self.U_m, self.I_m)

    def plot_PU(self, u_range: np.ndarray = np.linspace(0, 40, 81), label="Solar") -> None:
        """
        Please use "plt.show()" to view the figure.
        """
        c2 = (self.U_m / self.U_oc - 1) / np.log(1 - self.I_m / self.I_sc)
        c1 = (1 - self.I_m / self.I_sc) * np.exp(-self.U_m / c2 / self.U_oc)
        i = self.I_sc * (1 - c1 * (np.exp(u_range / c2 / self.U_oc) - 1))
        p = u_range * i
        flt = p >= 0
        plt.plot(u_range[flt], p[flt], label=label)
        plt.scatter(self.U_m, self.U_m * self.I_m)


class IdealPanel:

    def __init__(self) -> None:
        # battery
        self.battery = Battery()
        # power calculation
        self.eta_MPPT = 1.0
        self.beta = 0.005
        self.rho_reflection = 1.0
        # structure
        self.area = 100.0
        self.ang_Z = 30.0      # deg, the altitude angle, 0 means parallel to the ground
        self.ang_P = 180.0     # deg, the azimuth angle, 0 means facing north

    def get_power(self, G_bh: float, G_dh: float, altitude: float, azimuth: float, temperature: float) -> float:
        G_st = self.get_irradiance(G_bh, G_dh, altitude, azimuth)
        eta_ref = self.battery.eta
        T_c_ref = self.battery.T_ref
        T_c = temperature + ((self.battery.NOCT - 20) / 800) * G_st
        eta = eta_ref * self.eta_MPPT * (1 - self.beta * (T_c - T_c_ref))
        power = eta * self.area * G_st
        return power

    def get_irradiance(self, G_bh: float, G_dh: float, altitude: float, azimuth: float) -> float:
        G_gh = G_bh + G_dh
        # NOTICE: degree to radian
        ang_Z = np.deg2rad(self.ang_Z)
        ang_P = np.deg2rad(self.ang_P)
        altitude = np.deg2rad(altitude)
        azimuth = np.deg2rad(azimuth)
        cos_theta = np.sin(ang_Z) * np.cos(altitude) * np.cos(ang_P - azimuth) + np.sin(altitude) * np.cos(ang_Z)
        G_b = G_bh * cos_theta
        G_d = G_dh * (1 + np.cos(ang_Z / 2)) * 2 / 3
        G_r = G_gh * self.rho_reflection * (1 - np.cos(ang_Z / 2)) / 2
        G_st = G_b + G_d + G_r
        return G_st

    def simulate(self, data: pd.DataFrame, latitude: float, expand_frame: bool = False) -> pd.DataFrame:
        temperature = data['Temperature']
        G_bh = data['DNI']
        G_dh = data['DHI']
        # calculate the Declination angle
        n = data['Day of Year']
        dec = 23.45 * np.sin(2 * np.pi * (284 + n) / 365)
        # calculate the altitude angle At
        altitude = 90 - data['Solar Zenith Angle']
        altitude.clip(0, 90, inplace=True)
        # calculate the azimuth angle Az
        # NOTICE: degree to radian
        dec = np.deg2rad(dec)
        altitude = np.deg2rad(altitude)
        latitude = np.deg2rad(latitude)
        cos_Az = (np.sin(dec) - np.sin(altitude) * np.sin(latitude)) / (np.cos(altitude) * np.cos(latitude))
        cos_Az.clip(-1, 1, inplace=True)
        azimuth = np.rad2deg(np.arccos(cos_Az))
        azimuth = np.where(data['Hour Angle'] < 0, azimuth, 360 - azimuth)
        power = self.get_power(G_bh, G_dh, altitude, azimuth, temperature)
        power = pd.DataFrame(power, columns=['Power'])
        if expand_frame:
            return pd.concat([data, power], axis=1)
        else:
            return power


def decline_func(values: np.ndarray, a: float, r: float) -> np.ndarray:
    # y = 1 / (1 + a * (exp(r * t) - 1))
    decline_factor = a * (np.exp(r * values) - 1)
    return 1 / (1 + decline_factor)


class RealPanel(IdealPanel):

    def __init__(self) -> None:
        super().__init__()

    def get_gain(self, aging: float, damage: float, deposit: float, shading: float) -> float:
        """
        aging:      days        (days >= 0)
        damage:     percent     (0.0 <= damage <= 1.0, ideal: 0.0)
        deposit:    whatever    (deposit >=0)
        shading:    percent     (0.0 <= shading <= 1.0, ideal: 0.0)
        """
        gain_aging = decline_func(aging / 365, 0.01, 0.1)
        gain_damage = 1 - damage
        gain_deposit = decline_func(deposit, 0.1, 2)
        gain_shading = 1 - shading
        return gain_aging * gain_damage * gain_deposit * gain_shading

    def simulate(self, data: pd.DataFrame, latitude: float, expand_frame: bool = False) -> pd.DataFrame:
        power = super().simulate(data, latitude, expand_frame)
        aging = data['Aging']
        damage = data['Damage']
        deposit = data['Deposit']
        shading = data['Shading']
        power['Real Power'] = power['Power'] * self.get_gain(aging, damage, deposit, shading)
        return power


def virtual_scene(data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    N = len(data)
    P1 = 36
    # cleaning times
    P2 = 3
    # deposit decline
    P3 = 4e-5
    # aging
    data['Aging'] = np.arange(N) / 144 + np.random.randint(3650)
    # damage
    a = np.random.random(P1)
    b = interp1d(np.arange(P1), a, kind='zero')(np.linspace(0, P1 - 1, N))
    c = np.where(b > 0.8, (b - 0.8) * 5, 0)
    data['Damage'] = c
    # shading
    a = np.random.random(N)
    b = np.convolve(a, np.hanning(9))[0:N]
    b = b / b.max()
    c = np.power(b, 6) * np.random.random()
    data['Shading'] = c
    # deposit: +0.003 per day
    r = np.random.random(N)
    clean = np.random.randint(0, N, size=P2)
    d = np.empty(N)
    d[0] = r[0]
    for i in range(1, N):
        if i in clean:
            d[i] = 0
        else:
            d[i] = d[i - 1] + 2 * P3 * r[i]
    data['Deposit'] = d
    return (data, clean)


if __name__ == '__main__':
    rp = RealPanel()
    b = rp.battery
    # plot I-U character
    b.update_env(200, 25)
    b.plot_IU(u_range=np.linspace(0, 40, 401), label="Irradiance = 200 W/m²")
    b.update_env(600, 25)
    b.plot_IU(u_range=np.linspace(0, 40, 401), label="Irradiance = 600 W/m²")
    b.update_env(1000, 25)
    b.plot_IU(u_range=np.linspace(0, 40, 401), label="Irradiance = 1000 W/m²")
    plt.title("I-U Curve of the PV Panel")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current (A)")
    plt.legend()
    plt.grid()
    plt.show()
    # plot P-U character
    b.update_env(200, 25)
    b.plot_PU(u_range=np.linspace(0, 40, 401), label="Irradiance = 200 W/m²")
    b.update_env(600, 25)
    b.plot_PU(u_range=np.linspace(0, 40, 401), label="Irradiance = 600 W/m²")
    b.update_env(1000, 25)
    b.plot_PU(u_range=np.linspace(0, 40, 401), label="Irradiance = 1000 W/m²")
    plt.title("P-U Curve of the PV Panel")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Power (W)")
    plt.legend()
    plt.grid()
    plt.show()
    # plot P-G character
    g = np.linspace(0, 150, 151)
    plt.plot(g, rp.get_power(200, g, 30, 0, 15))
    plt.title("P-G Curve of the Solar Field")
    plt.xlabel("Diffuse Horizontal Irradiation / DHI (W/m²)")
    plt.ylabel("Power (W)")
    plt.text(0,
             5000,
             "Area: 100m²\nDNI: 200 W/m²\nAltitute: 30°\nTemperature: 15℃\nSun-tracking: No",
             bbox=dict(
                 boxstyle="round",
                 ec=(0.8, 0.8, 0.8),
                 fc=(1., 1., 1.),
             ))
    plt.grid()
    plt.show()
