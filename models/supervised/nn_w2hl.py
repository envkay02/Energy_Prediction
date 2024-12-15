"""
References:
- https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
- https://www.geeksforgeeks.org/how-to-implement-neural-networks-in-pytorch/
- https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from data_preprocessing_sl import X, y

# Simple NN with two hidden layers
class TwoHiddenLayerNN(nn.Module):
    # Define the layers
    def __init__(self):
        super(TwoHiddenLayerNN, self).__init__()
        # TODO: Set the number of input features based on the chosen dataset:
        # - Use 51 input features for the full dataset
        # - Use 29 input features for the Benchmark subset
        # - Use 16 input features for the Baseline subset
        self.fc1 = nn.Linear(51, 128) # 51, 29 or 16 for the first param
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) # ReLU activation function for the first hidden layer
        x = torch.relu(self.fc2(x)) # ReLU activation function for the second hidden layer
        x = self.fc3(x) # Output layer
        return x


accuracies = []
class_accuracies = {i: [] for i in range(2, 11)}

for i in range(20):
    torch.manual_seed(42 + i)
    # TODO: Specify the train-test split ratio:
    # - Use test_size=0.2 for an 80/20 split
    # - Use test_size=0.5 for a 50/50 split
    # - Use test_size=0.9 for a 10/90 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + i, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Calculate class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # Initialize
    model_NN2 = TwoHiddenLayerNN()
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model_NN2.parameters(), lr=0.001)

    # Train
    num_epochs_NN2 = 20
    for epoch in range(num_epochs_NN2):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_NN2(inputs)
            loss = criterion(outputs, labels - 2)  # Adjust labels to start from 0
            loss.backward()
            optimizer.step()

    # Evaluate
    model_NN2.eval()
    with torch.no_grad():
        test_outputs = model_NN2(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)
        predicted = predicted + 2 # Adjust back to original labels
        accuracy = accuracy_score(y_test_tensor, predicted)
        accuracies.append(accuracy)

        # Calculate accuracy for each class
        cm = confusion_matrix(y_test_tensor, predicted, labels=range(2, 11))
        for j in range(2, 11):
            idx = j - 2
            class_accuracy = cm[idx, idx] / np.sum(cm[idx, :])
            class_accuracies[j].append(class_accuracy)

# Calculate mean and variance of accuracy scores
mean_accuracy = np.mean(accuracies)
variance = np.var(accuracies)

# Calculate mean accuracy for each class
mean_class_accuracies = {k: np.mean(v) for k, v in class_accuracies.items()}

print("Neural Network w. 2 hidden layers:")
print(f"Mean Accuracy: {mean_accuracy:.5f}")
print(f"Variance: {variance:.5f}\n")

print("Classwise Mean Accuracies:")
for k, v in mean_class_accuracies.items():
    print(f"Class {k}: {v:.5f}")