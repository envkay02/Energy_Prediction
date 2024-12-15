# README

This folder contains the implementations of the supervised, semi-supervised and self-supervised models used for the task of energy prediction. This file provides some guidelines to generate the same results as in the study. 

## Info

Throughout the code, you will find `TODO` comments indicating places where changes, that are mentioned in this file, are required.

## Note

This version of the code folder does not contain the data due to privacy concerns. If you want to run the code, please create the `data` and `self_supervised/data` folders and add the datasets. The data should be in the form of `.csv` files with the following names:

- "new_data_array_vk.csv" for the full dataset
- "DataSplit_Benchmark_vk.csv" for the Benchmark subset
- "DataSplit_Baseline_vk.csv" for the Baseline subset

---

## Supervised Models

### Change the Test Split
- Adjust the test split to the wanted value in each models implementation file.

### Change the Subset
- Change the wanted subset in `data_preprocessing_sl.py`.

### Simple Neural Network with 1 & 2 Hidden Layers
- Change the number of `in_features` according to the selected subset.

### LDA, Gradient Boosting, and Random Forest
- When using the Benchmark subset with a 10/90 split, set the `k_neighbors` parameter of SMOTE to `4`.

---

## Semi-Supervised Models

### Change the Test Split
- Adjust the test split to the wanted value in each models implementation file.

### Change the Subset
- Change the wanted subset in `data_preprocessing_ssupl.py`.

### Using the Benchmark Subset with a 10/90 Split
- Set the `k_neighbors` parameter of SMOTE to `4`.

### Label Propagation and Label Spreading
- Adjust the `n_neighbors` parameter according to the chosen split:
    - 80/20 split: `n_neighbors = 10`
    - 50/50 split: `n_neighbors = 15`
    - 10/90 split: `n_neighbors = 20`
    - **Exception**: For LabelPropagation with the Benchmark subset and 10/90 split, set `n_neighbors = 25`.

---

## Self-Supervised Models

### Important
- To run the SSL models, please extract or copy the `Energy_Prediction` folder to your Google Drive storage and run the notebooks in Google Colab. It is important that the name of the folder remains `Energy_Prediction`. Also, it could be necessary to adjust the paths in the notebooks and `Data_Preparation_...` files according to your Google Drive folder structure.

### Change the Test Split
- Adjust the test split to the wanted value in the `Data_Preparation_...` file for each model.

### Change the Subset
- Change the path to the wanted subset in the `Data_Preparation_...` file for each model.

### Using the Benchmark Subset with a 10/90 Split
- Set the `k_neighbors` parameter of SMOTE to `4` in the `Data_Preparation_...` file for each model.

### TabNet Model
- When using a 10/90 split, set the `n_shared` and `n_independent` parameters of the pretrainer to `2` in the model's implementation file.

---
