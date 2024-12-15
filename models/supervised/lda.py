"""
References:
- https://scikit-learn.org/stable/modules/lda_qda.html
- https://scikit-learn.org/dev/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from data_preprocessing_sl import X, y
from imblearn.over_sampling import SMOTE

accuracies = []
class_accuracies = {i: [] for i in range(2, 11)}

for i in range(20):
    # TODO: Specify the train-test split ratio:
    # - Use test_size=0.2 for an 80/20 split
    # - Use test_size=0.5 for a 50/50 split
    # - Use test_size=0.9 for a 10/90 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + i, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42+i,
                  # TODO: Uncomment the following line only if using the Benchmark subset with a 10/90 split
                  #k_neighbors=4
                  )
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_res, y_train_res)

    y_pred = lda.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Calculate accuracy for each class
    cm = confusion_matrix(y_test, y_pred, labels=range(2, 11))
    for j in range(2, 11):
        idx = j - 2
        class_accuracy = cm[idx, idx] / np.sum(cm[idx, :])
        class_accuracies[j].append(class_accuracy)

# Calculate mean and variance of accuracy scores
mean_accuracy = np.mean(accuracies)
variance = np.var(accuracies)

# Calculate mean accuracy for each class
mean_class_accuracies = {k: np.mean(v) for k, v in class_accuracies.items()}

print("LDA:")
print(f"Mean Accuracy: {mean_accuracy:.5f}")
print(f"Variance: {variance:.5f}\n")

print("Classwise Mean Accuracies:")
for k, v in mean_class_accuracies.items():
    print(f"Class {k}: {v:.5f}")

