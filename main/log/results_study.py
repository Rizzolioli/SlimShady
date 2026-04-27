from utils.visualizations import show_results, get_experiment_results, get_column_names, verify_integrity
import os
import pandas as pd


columns = ["algo", "experiment_id", "dataset", "seed", "generation", "training_fitness", "timing", "pop_node_count"]
logger_name = os.getcwd() + "/slim_ppb.csv"
log_level = 1

df = pd.read_csv(logger_name)
print(df.columns)
# df = df.iloc[:-1]

#df_filtered = df[df['toxicity'] != 'concrete']

# Save the cleaned DataFrame to a new CSV file
# df_filtered.to_csv(os.getcwd() + '/filtered_data.csv', index=False)


show_results(x_var = "generation", y_var="nodes_count", experiment_id=-1, logger_name=logger_name, log_level=1, dataset=None)  # nodes_count

