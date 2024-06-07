import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid")

# Substitution dictionary
substitutions = {
    'SlimGSGP_1_mul_False': 'SLIM_mul_ABS',
    'SlimGSGP_1_mul_True': 'SLIM_mul_SIG1',
    'SlimGSGP_2_sum_False': 'SLIM_sum_SIG2',
    'SlimGSGP_2_sum_True': 'SLIM_sum_SIG2'
}


def replace_substring(value, substitutions):
    for old, new in substitutions.items():
        value = value.replace(old, new)
    return value


logger_name = os.getcwd() + "/crossover_16may.csv"

for name_ds in ["resid_build_sale_price", "toxicity", "concrete", "instanbul", "ppb", "energy"]:
    logger_no_crossover = os.getcwd() + f"/results_def/slim_{name_ds}.csv"
    log_level = 1

    data_full = pd.read_csv(logger_name, names=["algo", "experiment_id", "dataset", "seed", "generation",
                                                "training_fitness", "timing", "pop_node_count",
                                                "test_fitness", "elite_size", "log_level"])
    data_full['algo'] = data_full['algo'].apply(replace_substring, substitutions=substitutions)

    data_full_no_cross = pd.read_csv(logger_no_crossover, names=["algo", "experiment_id", "dataset", "seed", "generation",
                                                                 "training_fitness", "timing", "pop_node_count",
                                                                 "test_fitness", "elite_size", "log_level"])
    data_full_no_cross['algo'] = data_full_no_cross['algo'].apply(replace_substring, substitutions=substitutions)

    data_full["elite_size"] = data_full["elite_size"].astype(float)

    if not os.path.exists(os.getcwd() + f"/plots/{name_ds}"):
        os.mkdir(os.getcwd() + f"/plots/{name_ds}")

    palette = sns.color_palette("husl", 8)  # Use a colorful palette

    for argument in ["elite_size", "test_fitness"]:
        for name_algo in ["SLIM_mul_ABS", "SLIM_mul_SIG1", "SLIM_sum_SIG2"]:
            for cross in ["sc", "sdc"]:
                data = data_full.copy()
                data_no_cross = data_full_no_cross.copy()
                data_no_cross = data_no_cross[data_no_cross["algo"] == name_algo]
                median_node_count_no_cross = data_no_cross.groupby(['algo', 'generation'])[argument].median().reset_index()

                data = data[data['dataset'] == name_ds]
                data = data[data['algo'].str.contains(name_algo)]
                if cross:
                    data = data[data['algo'].str.contains(cross)]

                median_node_count = data.groupby(['algo', 'generation'])[argument].median().reset_index()

                plt.figure(figsize=(8, 6))  # Increase figure size

                for i, algo in enumerate(median_node_count['algo'].unique()):
                    subset = median_node_count[median_node_count['algo'] == algo]
                    plt.plot(subset['generation'], subset[argument], label=algo, linewidth=2,
                             color=palette[i % len(palette)])

                plt.plot(median_node_count_no_cross['generation'], median_node_count_no_cross[argument],
                         label=name_algo, linewidth=3, color="red")

                if argument == 'elite_size' and name_algo == "SLIM_sum_SIG2":
                    plt.ylim(0, 2500)

                if argument == "elite_size":
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.title(f"{name_ds.title()} - {argument.replace('_', ' ').title()}")
                plt.xlabel('Generation')
                plt.ylabel(argument.replace('_', ' ').title())
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plt.savefig(os.getcwd() + f"/plots/{name_ds}/{argument}_{name_algo}_{cross}.png")
                plt.close()
