import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

base_palette = sns.color_palette("deep", 5)  # 'deep' is often used for its vibrant colors
# Create a figure for the legend
# Determine the number of columns for two rows, depending on the number of labels
num_labels = 6
columns = num_labels // 2 if num_labels % 2 == 0 else num_labels // 2 + 1

# Create a figure for the legend that's wider to accommodate more entries per line
fig_legend = plt.figure(figsize=(12, 3))  # Adjust figsize as needed to prevent clipping
ax_legend = fig_legend.add_subplot(111)

def plot_all_median_node_counts(name_ds, argument):
    csv_path = os.getcwd() + f'/results_def/slim_{name_ds}.csv'
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
    data.replace({'algo': {'SlimGSGP_1_mul_False': 'SLIM*ABS',
                           'SlimGSGP_1_mul_True': 'SLIM*SIG1',
                           'SlimGSGP_1_sum_False': 'SLIM+ABS',
                           'SlimGSGP_1_sum_True': 'SLIM+SIG1',
                           'SlimGSGP_2_mul_False': 'SLIM*SIG2',
                           'SlimGSGP_2_sum_False': 'SLIM+SIG2'
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

    plt.figure(figsize=(8, 8))

    # Set line colors and styles
    # Define specific colors for each algorithm, using shades of base colors for similar algorithms
    color_map = {
        'SLIM*SIG1': "#ff7f0e",  # Slightly lighter
        'SLIM*ABS': "#2ca02c",  # Slightly darker
        'SLIM*SIG2': "#1f77b4",  # Base color for multiplicative versions
        'SLIM+SIG1': "#ffbb78",  # Slightly lighter
        'SLIM+ABS': "#98df8a",  # Slightly darker
        'SLIM+SIG2': "#aec7e8"  # Base color for additive versions
    }
    style_map = {'GSGP': "solid", 'GP': "solid",
                 'SLIM*SIG1': "dashed", 'SLIM*ABS': "dashed", 'SLIM*SIG2': "dashed",
                 'SLIM+SIG1': "solid", 'SLIM+ABS': "solid", 'SLIM+SIG2': "solid"}
    # style_map = {
    #              'SLIM*SIG1': "solid", 'SLIM*ABS': "solid", 'SLIM*SIG2': "solid",
    # R             'SLIM+SIG1': "solid", 'SLIM+ABS': "solid", 'SLIM+SIG2': "solid"}

    # Adjust font sizes
    plt.rc('axes', titlesize=40)  # Title size
    plt.rc('axes', labelsize=40)  # Label size
    plt.rc('xtick', labelsize=40)  # X-tick size
    plt.rc('ytick', labelsize=40)  # Y-tick size
    plt.rc('legend', fontsize=20)  # Legend size

    # Plot data from the CSV file
    # Plot data for each algorithm
    for algo in median_node_count['algo'].unique():
        subset = median_node_count[median_node_count['algo'] == algo]
        plt.plot(subset['generation'], subset[argument], label=algo, color=color_map[algo], linestyle=style_map[algo],
                 linewidth=6)

    # Special handling for GSGP and GP data
    plt.plot(range(len(gsgp_values)), gsgp_values, label='GSGP', color="gray", linestyle='-', linewidth=6)
    plt.plot(range(len(gp_values)), gp_values, label='STDGP', color="k", linestyle='-', linewidth=5)


    plt.xlabel('Generation')
    plt.ylabel(caption)
    if argument == 'elite_size':
        plt.ylim(0, 2500)
    else:
        if name_ds == "toxicity":
            plt.ylim(1250, 2500)
        if name_ds == "instanbul":
            plt.ylim(0.011, 0.020)
        if name_ds == "energy":
            plt.ylim(1.5, 7.5)
        if name_ds == "ppb":
            plt.ylim(0, 80)
        if name_ds == "concrete":
            plt.ylim(5.2, 20)
        if name_ds == "resid": 
            plt.ylim(25, 130)

    #if argument == "test_fitness":
    #    plt.title(name_ds.upper())
    # plt.legend(title='Algorithm', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.getcwd() + f"/../plots/{argument}_{name_ds}.pdf", dpi=800)
    # plt.close()

    # After your plotting code, assume you have retrieved handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Define your custom order as a list of labels in the order you want them to appear
    custom_order = ['STDGP', 'GSGP', 'SLIM+SIG2', 'SLIM*SIG2', 'SLIM+ABS', 'SLIM*ABS', 'SLIM+SIG1', 'SLIM*SIG1']

    # Create new lists for handles and labels based on the custom order
    ordered_handles = []
    ordered_labels = []

    for label in custom_order:
        if label in labels:
            index = labels.index(label)
            ordered_handles.append(handles[index])
            ordered_labels.append(label)

    # Now save the legend separately
    handles, labels = plt.gca().get_legend_handles_labels()
    fig_legend = plt.figure(figsize=(10, 2))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(ordered_handles, ordered_labels, loc='center', ncol=4, frameon=False)
    ax_legend.axis('off')
    fig_legend.tight_layout()

    fig_legend.savefig(os.getcwd() + "/../legend.pdf", dpi=400)
    plt.close(fig_legend)  # Close the figure to free resources


# Example usage
for name_ds in ["concrete", "ppb", "instanbul", "toxicity", "resid_build_sale_price", "energy"]:
    for argument in ["training_fitness", "elite_size", "test_fitness"]:
        plot_all_median_node_counts(name_ds=name_ds, argument=argument)
