import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

logger_name = os.getcwd() + "/crossover_ins_ener.csv"
log_level = 1

data = pd.read_csv(logger_name, names=["algo", "experiment_id", "dataset", "seed", "generation",
                                        "training_fitness", "timing", "pop_node_count",
                                        "test_fitness", "elite_size", "log_level"])

print("read")
data["elite_size"] = data["elite_size"].astype(float)
print(np.isinf(data["elite_size"]).any())

argument = "elite_size"
name_ds = "instanbul"

data = data[data['dataset'] == name_ds]
median_node_count = data.groupby(['algo', 'generation'])[argument].median().reset_index()

plt.figure(figsize=(12, 12))

for algo in median_node_count['algo'].unique():
    subset = median_node_count[median_node_count['algo'] == algo]
    plt.plot(subset['generation'], subset[argument], label=algo, linewidth=6)

if argument == 'elite_size':
    plt.ylim(0, 2500)

plt.legend()
plt.show()
plt.savefig(os.getcwd() + f"{argument}_{name_ds}.png")
plt.close()