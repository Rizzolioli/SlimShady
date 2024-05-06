import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def plot_all_median_node_counts(name_ds, argument):
    csv_path = os.getcwd() + f'/../res/slim_{name_ds}.csv'
    gsgp_path, gp_path = None, None
    caption = ''

    # Set paths and caption based on the argument
    if argument == 'elite_size':
        gsgp_path = os.getcwd() + f'/results/SIZE/RESULT_Median_Size_Of_Best_Config_GSGP_random_MS_04_09_2023_{name_ds.upper()}.txt'
        gp_path = os.getcwd() + f'/results/SIZE/RESULT_Median_Size_Of_Best_Config_STDGP_04_09_2023_{name_ds.upper()}.txt'
        caption = 'Size of the Elite'
    elif argument == 'training_fitness':
        gsgp_path = os.getcwd() + f'/results/FITNESS_ON_TRAINING/RESULT_Median_Best_Fitness_On_Training_Config_GSGP_random_MS_04_09_2023_{name_ds.upper()}.txt'
        gp_path = os.getcwd() + f'/results/FITNESS_ON_TRAINING/RESULT_Median_Best_Fitness_On_Training_Config_STDGP_04_09_2023_{name_ds.upper()}.txt'
        caption = 'Train Fitness'
    elif argument == 'test_fitness':
        gsgp_path = os.getcwd() + f'/results/FITNESS_ON_TEST/RESULT_Median_Best_Fitness_On_Test_Config_GSGP_random_MS_04_09_2023_{name_ds.upper()}.txt'
        gp_path = os.getcwd() + f'/results/FITNESS_ON_TEST/RESULT_Median_Best_Fitness_On_Test_Config_STDGP_04_09_2023_{name_ds.upper()}.txt'
        caption = 'Test Fitness'

    # Load data from the CSV file
    data = pd.read_csv(csv_path, names=["algo", "experiment_id", "dataset", "seed", "generation",
                                        "training_fitness", "timing", "pop_node_count",
                                        "test_fitness", "elite_size", "log_level"])

    # Example replacements
    data.replace({'algo': {'SlimGSGP_1_mul_False': 'SLIM-NORM*1',
                           'SlimGSGP_1_mul_True': 'SLIM-SIG*1',
                           'SlimGSGP_1_sum_False': 'SLIM-NORM+1',
                           'SlimGSGP_1_sum_True': 'SLIM-SIG+1',
                           'SlimGSGP_2_mul_False': 'SLIM*2',
                           'SlimGSGP_2_sum_False': 'SLIM+2'
                           }},
                 inplace=True)

    # Calculate the median node count for each algorithm and generation in the CSV data
    median_node_count = data.groupby(['algo', 'generation'])[argument].median().reset_index()

    # Load GSGP data
    with open(gsgp_path, 'r') as file:
        gsgp_values = [float(val) for val in file.read().split(',') if val.strip()]
    gsgp_data = pd.DataFrame({argument: gsgp_values})
    gsgp_data['generation'] = range(len(gsgp_data))
    gsgp_data['algo'] = 'GSGP'

    # Load GP data
    with open(gp_path, 'r') as file:
        gp_values = [float(val) for val in file.read().split(',') if val.strip()]
    gp_data = pd.DataFrame({argument: gp_values})
    gp_data['generation'] = range(len(gp_data))
    gp_data['algo'] = 'GP'

    # Set Seaborn style and palette
    sns.set_theme(style="white", palette="colorblind")
    palette = sns.color_palette("tab10", n_colors=len(median_node_count['algo'].unique()))

    plt.figure(figsize=(12, 6))

    # Adjust font sizes
    plt.rc('axes', titlesize=20)  # Title size
    plt.rc('axes', labelsize=18)  # Label size
    plt.rc('xtick', labelsize=20)  # X-tick size
    plt.rc('ytick', labelsize=20)  # Y-tick size
    plt.rc('legend', fontsize=12)  # Legend size

    # Plot data from the CSV file
    for i, algo in enumerate(median_node_count['algo'].unique()):
        subset = median_node_count[median_node_count['algo'] == algo]
        plt.plot(subset['generation'], subset[argument], label=algo, color=palette[i], linewidth=2)

    # Plot GSGP data
    plt.plot(gsgp_data['generation'], gsgp_data[argument], label='GSGP', color='blue', linewidth=2)

    # Plot GP data
    plt.plot(gp_data['generation'], gp_data[argument], label='GP', color='green', linewidth=2)

    plt.xlabel('Generation')
    plt.ylabel(caption)
    if argument == 'elite_size':
        plt.ylim(0, 2500)

    plt.title(name_ds.upper())
    plt.legend(title='Algorithm', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.getcwd() + f"/../plots/{argument}_{name_ds}_png")
    plt.close()


# Example usage
for name_ds in ["ppb", "instanbul", "toxicity", "resid_build_sale_price", "energy"]:
    for argument in ["training_fitness", "elite_size", "test_fitness"]:
        plot_all_median_node_counts(name_ds=name_ds, argument=argument)
