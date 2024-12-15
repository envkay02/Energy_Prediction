import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# TODO: Specify the dataset to use:
# - "new_data_array_vk.csv" for the full dataset
# - "DataSplit_Baseline_vk.csv" for the Baseline subset
# - "DataSplit_Benchmark_vk.csv" for the Benchmark subset
# To switch datasets, replace the end part of the path name accordingly.
data_path = '/content/drive/MyDrive/Energy_Prediction/models/self_supervised/data/new_data_array_vk.csv'
data = pd.read_csv(data_path, header=None)

data.columns = ['Label'] + [f'Feature_{i}' for i in range(1, data.shape[1])]

# Split the data into labeled and unlabeled data
labeled_data = data[data['Label'] != 1.0]
unlabeled_data = data[data['Label'] == 1.0]

# TODO: Specify the train-test split ratio:
# - Use test_size=0.2 for an 80/20 split
# - Use test_size=0.5 for a 50/50 split
# - Use test_size=0.9 for a 10/90 split
train_labeled, test_labeled = train_test_split(
    labeled_data, test_size=0.2, stratify=labeled_data['Label'], random_state=42
)

# Split the labeled data into features and labels
train_features = train_labeled.drop(columns='Label')
train_labels = train_labeled['Label']
test_features = test_labeled.drop(columns='Label')
test_labels = test_labeled['Label']

# Apply SMOTE to the labeled data to handle class imbalance
smote = SMOTE(random_state=42,
              # TODO: Uncomment the following line only if using the Benchmark subset with a 10/90 split
              # k_neighbors=4
              )
resampled_features, resampled_labels = smote.fit_resample(train_features, train_labels)

# Combine the resampled labeled data with the unlabeled data for the pretraining step
pretrain_features = pd.concat([
    resampled_features,
    unlabeled_data.drop(columns='Label')
])

# Standardize the features
scaler = StandardScaler()
X_pretrain = scaler.fit_transform(pretrain_features)
X_train = scaler.transform(resampled_features)
y_train = resampled_labels
X_test = scaler.transform(test_features)
y_test = test_labels