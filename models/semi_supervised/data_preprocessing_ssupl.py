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

# Map unlabeled data from 1 to -1, because of the implementation details of the ssupl models
data['Label'] = data['Label'].replace(1, -1)

# Split the data into features and labels
X = data.drop(columns='Label')
y = data['Label']

# Split the data into labeled and unlabeled data
X_labeled = X[y != -1]
y_labeled = y[y != -1]
X_unlabeled = X[y == -1]
y_unlabeled = y[y == -1]