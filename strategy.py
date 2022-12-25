from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from dowhy import gcm

from solar_model.climate import climate_loader
from solar_model.solar_panel import RealPanel, virtual_scene

# parameters
# data and simulation
DATASET_PATH = Path("D:\\work\\YuCai\\projects\\data\\5031581_31.28_121.47_2020.csv")
LATITUDE = 31.28
CLEAN_PLAN = [pd.Timestamp("2020-04-01, 10:30:00"), pd.Timestamp("2020-09-01, 13:50:00"), pd.Timestamp("2020-12-20, 09:45:00")]
REPAIR_PLAN = [pd.Timestamp("2020-05-20, 14:20:00"), pd.Timestamp("2020-11-10, 09:20:00")]
# causal model
ROOT_NODES = ["DHI", "DNI", "GHI", "Temperature", "Aging", "Shading", "Last Clean", "Last Repair"]
NON_ROOT_NODES = ["Damage", "Deposit", "Real Power"]
CAUSAL_GRAPH = nx.DiGraph([("DHI", "Real Power"), ("DNI", "Real Power"), ("GHI", "Real Power"), ("Temperature", "Real Power"),
                           ("Aging", "Real Power"), ("Damage", "Real Power"), ("Deposit", "Real Power"),
                           ("Shading", "Real Power"), ("Last Clean", "Deposit"), ("Last Repair", "Damage")])
# counterfactual
REPAIR_ASSUMPUTION = [
    pd.Timestamp("2020-02-10, 09:30:00"),
    pd.Timestamp("2020-04-13, 09:30:00"),
    pd.Timestamp("2020-06-08, 09:30:00"),
    pd.Timestamp("2020-08-10, 09:30:00"),
    pd.Timestamp("2020-10-12, 09:30:00"),
    pd.Timestamp("2020-12-14, 09:30:00"),
]

if __name__ == '__main__':
    # load data and simulate
    climate_data = climate_loader(DATASET_PATH)
    runtime_data = virtual_scene(climate_data, REPAIR_PLAN, CLEAN_PLAN)
    panel = RealPanel()
    simulation_data = panel.simulate(runtime_data, LATITUDE, expand_frame=True)

    # build causal model and train
    training_data = simulation_data.resample("D").mean()
    causal_model = gcm.InvertibleStructuralCausalModel(CAUSAL_GRAPH)
    for n in ROOT_NODES:
        causal_model.set_causal_mechanism(n, gcm.EmpiricalDistribution())
    for n in NON_ROOT_NODES:
        causal_model.set_causal_mechanism(n, gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
    gcm.fit(causal_model, training_data)
    # calculate the counterfactual
    cf_power = gcm.counterfactual_samples(causal_model, {'Last Clean': lambda x: x % 60},
                                          observed_data=simulation_data)["Real Power"]
    cf_power.index = simulation_data.index

    # plot the conclusion
    ideal_power = simulation_data['Power']
    real_power = simulation_data['Real Power']

    ideal_power_res = ideal_power.resample('W').mean()
    real_power_res = real_power.resample('W').mean()
    cf_power_res = cf_power.resample('W').mean()
    power_improvement = cf_power_res - real_power_res

    plt.plot(ideal_power_res.index, ideal_power_res.values, label="Ideal Output Power")
    plt.plot(real_power_res.index, real_power_res.values, label="Real Output Power")
    plt.plot(cf_power_res.index, cf_power_res.values, label="Counterfactual Output Power")
    plt.plot(power_improvement.index, power_improvement.values, label="Power Improvement")

    plt.title("Output Power and Improvement Per Week")
    plt.xlabel("Date")
    plt.ylabel("Average Output Power Per Week (W)")
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
