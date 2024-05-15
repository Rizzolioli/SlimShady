import pandas as pd
import os

# Load the original CSV file
for name_ds in ["ppb", "instanbul", "toxicity", "resid_build_sale_price", "energy"]:
    base_path = os.getcwd()
    csv_path = f'{base_path}/slim_{name_ds}.csv'
    csv_new_path = f'{base_path}/new_sig.csv'

    # Load the data with headers as specified
    original_df = pd.read_csv(csv_path, names=["algo", "experiment_id", "dataset", "seed", "generation",
                                               "training_fitness", "timing", "pop_node_count",
                                               "test_fitness", "elite_size", "log_level"], header=None)
    updates_df = pd.read_csv(csv_new_path, names=["algo", "experiment_id", "dataset", "seed", "generation",
                                                  "training_fitness", "timing", "pop_node_count",
                                                  "test_fitness", "elite_size", "log_level"], header=None)

    # Filter updates_df to include only rows where 'dataset' matches 'name_ds'
    updates_df_filtered = updates_df[updates_df['dataset'] == name_ds]

    # Remove unwanted rows from the original DataFrame based on 'algo'
    original_df = original_df[~original_df['algo'].isin(['SlimGSGP_1_mul_True', 'SlimGSGP_1_sum_True'])]

    # Concatenate the original and updated DataFrames
    concatenated_df = pd.concat([original_df, updates_df_filtered], ignore_index=True)

    # Define the new folder for results
    new_folder_path = os.path.join(base_path, 'results_def')

    # Create the directory if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Define the path for the new updated CSV file
    new_csv_path = os.path.join(new_folder_path, f'slim_{name_ds}.csv')

    # Save the concatenated DataFrame as a CSV in the new directory
    concatenated_df.to_csv(new_csv_path, index=False, header=None)

    print(f'Updated DataFrame saved to {new_csv_path}')

for name_ds in ["concrete"]:
    base_path = os.getcwd()
    csv_path = f'{base_path}/slim_{name_ds}.csv'
    csv_new_path = f'{base_path}/concrete_sig_n.csv'

    # Load the data with headers as specified
    original_df = pd.read_csv(csv_path, names=["algo", "experiment_id", "dataset", "seed", "generation",
                                               "training_fitness", "timing", "pop_node_count",
                                               "test_fitness", "elite_size", "log_level"], header=None)
    updates_df = pd.read_csv(csv_new_path, names=["algo", "experiment_id", "dataset", "seed", "generation",
                                                  "training_fitness", "timing", "pop_node_count",
                                                  "test_fitness", "elite_size", "log_level"], header=None)

    # Filter updates_df to include only rows where 'dataset' matches 'name_ds'
    updates_df_filtered = updates_df[updates_df['dataset'] == name_ds]

    # Remove unwanted rows from the original DataFrame based on 'algo'
    original_df = original_df[~original_df['algo'].isin(['SlimGSGP_1_mul_True', 'SlimGSGP_1_sum_True'])]

    # Concatenate the original and updated DataFrames
    concatenated_df = pd.concat([original_df, updates_df_filtered], ignore_index=True)

    # Define the new folder for results
    new_folder_path = os.path.join(base_path, 'results_def')

    # Create the directory if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # Define the path for the new updated CSV file
    new_csv_path = os.path.join(new_folder_path, f'slim_{name_ds}.csv')

    # Save the concatenated DataFrame as a CSV in the new directory
    concatenated_df.to_csv(new_csv_path, index=False, header=None)

    print(f'Updated DataFrame saved to {new_csv_path}')
