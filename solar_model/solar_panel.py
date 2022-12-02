import numpy as np
import matplotlib.pyplot as plt


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

    def plot_IU(self, u_range: np.ndarray = np.linspace(0, 40, 81)) -> None:
        """
        Please use "plt.show()" to view the figure.
        """
        c2 = (self.U_m / self.U_oc - 1) / np.log(1 - self.I_m / self.I_sc)
        c1 = (1 - self.I_m / self.I_sc) * np.exp(-self.U_m / c2 / self.U_oc)
        i = self.I_sc * (1 - c1 * (np.exp(u_range / c2 / self.U_oc) - 1))
        flt = i >= 0
        plt.plot(u_range[flt], i[flt])
        plt.scatter(self.U_m, self.I_m)

    def plot_PU(self, u_range: np.ndarray = np.linspace(0, 40, 81)) -> None:
        """
        Please use "plt.show()" to view the figure.
        """
        c2 = (self.U_m / self.U_oc - 1) / np.log(1 - self.I_m / self.I_sc)
        c1 = (1 - self.I_m / self.I_sc) * np.exp(-self.U_m / c2 / self.U_oc)
        i = self.I_sc * (1 - c1 * (np.exp(u_range / c2 / self.U_oc) - 1))
        p = u_range * i
        flt = p >= 0
        plt.plot(u_range[flt], p[flt])
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
        self.ang_Z = 30.0      # the altitude angle, 0 means parallel to the ground
        self.ang_P = 180.0     # the azimuth angle, 0 means facing north

    def get_power(self, G_bh: float, G_dh: float, altitude: float, azimuth: float, temperature: float) -> float:
        G_st = self.__get_irradiance(G_bh, G_dh, altitude, azimuth)
        eta_ref = self.battery.eta
        T_c_ref = self.battery.T_ref
        T_c = temperature + ((self.battery.NOCT - 20) / 800) * G_st
        eta = eta_ref * self.eta_MPPT * (1 - self.beta * (T_c - T_c_ref))
        power = eta * self.area * G_st
        return power

    def __get_irradiance(self, G_bh: float, G_dh: float, altitude: float, azimuth: float) -> float:
        G_gh = G_bh + G_dh
        cos_theta = np.sin(self.ang_Z) * np.cos(altitude) * np.cos(self.ang_P - azimuth) + np.sin(altitude) * np.cos(self.ang_Z)
        G_b = G_bh * cos_theta
        G_d = G_dh * (1 + np.cos(self.ang_Z / 2)) * 2 / 3
        G_r = G_gh * self.rho_reflection * (1 - np.cos(self.ang_Z / 2)) / 2
        G_st = G_b + G_d + G_r
        return G_st


def decline_func(values: np.ndarray, a: float, r: float) -> np.ndarray:
    # y = 1 / (1 + a * (exp(r * t) - 1))
    decline_factor = a * (np.exp(r * values) - 1)
    return 1 / (1 + decline_factor)


class RealPanel(IdealPanel):

    def __init__(self) -> None:
        super().__init__()
        self.age = 0
        self.damage = 0.0
        self.deposit = 0.0
        self.shading = 0.0

    def update_state(self, age: int, damage: float, deposit: float, shading: float) -> None:
        self.age = age
        self.damage = damage
        self.deposit = deposit
        self.shading = shading

    def get_power(self, G_bh: float, G_dh: float, altitude: float, azimuth: float, temperature: float) -> float:
        # real factors
        gain_aging = decline_func(self.age / 365, 0.01, 0.1)
        gain_damage = 1 - self.damage
        gain_deposit = decline_func(self.deposit, 0.1, 2)
        gain_shading = 1 - self.shading
        # calculate power
        power = super().get_power(G_bh, G_dh, altitude, azimuth, temperature)
        power = power * gain_aging * gain_damage * gain_deposit * gain_shading
        return power


if __name__ == '__main__':
    rp = RealPanel()
    b = rp.battery
    # plot I-U character
    b.update_env(200, 25)
    b.plot_IU()
    b.update_env(600, 25)
    b.plot_IU()
    b.update_env(1000, 25)
    b.plot_IU()
    plt.show()
    # plot P-U character
    b.update_env(200, 25)
    b.plot_PU()
    b.update_env(600, 25)
    b.plot_PU()
    b.update_env(1000, 25)
    b.plot_PU()
    plt.show()
    # plot P-G character
    g = np.linspace(0, 1000, 101)
    plt.plot(g, rp.get_power(g, 0.1 * g, 30, 0, 20))
    plt.show()
