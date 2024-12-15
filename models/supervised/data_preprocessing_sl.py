import pandas as pd
from pathlib import Path

project_root = Path().absolute().parents[1]

# TODO: Specify the dataset to use:
# - "new_data_array_vk.csv" for the full dataset
# - "DataSplit_Baseline_vk.csv" for the Baseline subset
# - "DataSplit_Benchmark_vk.csv" for the Benchmark subset
dataset_name = "new_data_array_vk.csv"

file_path = project_root / "data" / dataset_name
data = pd.read_csv(file_path, header=None)

data.columns = ['Label'] + [f'Feature_{i}' for i in range(1, data.shape[1])]

# Filter out unlabeled data
labeled_data = data[data['Label'] != 1]

# Split the data into features and labels
X = labeled_data.drop(columns='Label')
y = labeled_data['Label']








