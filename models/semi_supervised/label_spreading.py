"""
References:
- https://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html#sklearn.semi_supervised.LabelSpreading
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score, confusion_matrix
from data_preprocessing_ssupl import X_labeled, y_labeled, X_unlabeled, y_unlabeled
from sklearn.model_selection import train_test_split
import pandas as pd

accuracies = []
class_accuracies = {i: [] for i in range(2, 11)}

for i in range(20):
    # TODO: Specify the train-test split ratio:
    # - Use test_size=0.2 for an 80/20 split
    # - Use test_size=0.5 for a 50/50 split
    # - Use test_size=0.9 for a 10/90 split
    X_train_labeled, X_test_labeled, y_train_labeled, y_test_labeled = train_test_split(
        X_labeled, y_labeled, test_size=0.2, stratify=y_labeled, random_state=42 + i)

    # Combine labeled and unlabeled data
    X_train_combined = pd.concat([X_train_labeled, X_unlabeled])

    # Standardize the features
    scaler = StandardScaler()
    X_train_combined_scaled = scaler.fit_transform(X_train_combined)
    X_train_labeled_scaled = X_train_combined_scaled[:len(X_train_labeled)]
    X_unlabeled_scaled = X_train_combined_scaled[len(X_train_labeled):]
    X_test_scaled = scaler.transform(X_test_labeled)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42 + i,
                  # TODO: Uncomment the following line only if using the Benchmark subset with a 10/90 split
                  # k_neighbors=4
                  )
    X_train_labeled_res, y_train_labeled_res = smote.fit_resample(X_train_labeled_scaled, y_train_labeled)

    # Combine the resampled labeled data with the unlabeled data
    X_train = np.concatenate((X_train_labeled_res, X_unlabeled_scaled))
    y_train = np.concatenate((y_train_labeled_res, y_unlabeled.values))

    # TODO: Set the n_neighbors parameter based on the chosen split:
    # - Use n_neighbors=10 for an 80/20 split
    # - Use n_neighbors=15 for a 50/50 split
    # - Use n_neighbors=20 for a 10/90 split
    label_spread_model = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.8, max_iter=1000)
    label_spread_model.fit(X_train, y_train)

    y_pred = label_spread_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test_labeled, y_pred)
    accuracies.append(accuracy)

    # Calculate accuracy for each class
    cm = confusion_matrix(y_test_labeled, y_pred, labels=range(2, 11))
    for j in range(2, 11):
        idx = j - 2
        class_accuracy = cm[idx, idx] / np.sum(cm[idx, :])
        class_accuracies[j].append(class_accuracy)

# Calculate mean and variance of accuracy scores
mean_accuracy = np.mean(accuracies)
variance = np.var(accuracies)

# Calculate mean accuracy for each class
mean_class_accuracies = {k: np.mean(v) for k, v in class_accuracies.items()}

print("Label Spreading:")
print(f"Mean Accuracy: {mean_accuracy:.5f}")
print(f"Variance: {variance:.5f}\n")

print("Classwise Mean Accuracies:")
for k, v in mean_class_accuracies.items():
    print(f"Class {k}: {v:.5f}")