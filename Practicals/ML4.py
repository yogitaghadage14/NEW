# Implement K-Nearest Neighbors algorithm on diabetes.csv dataset. Compute confusion
# matrix, accuracy, error rate, precision and recall on the given dataset.
# Dataset link : https://www.kaggle.com/datasets/abdallamahgoub/diabetes

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Check the first few rows of the dataset
print(df.head())

# Step 3: Preprocessing the data
# Check for missing values and handle them (if any)
print(df.isnull().sum())

# If there are missing values, let's fill them with the mean or drop rows
df.fillna(df.mean(), inplace=True)

# Features (X) and target (y)
X = df.drop('Outcome', axis=1)  # Features (all columns except the target column)
y = df['Outcome']  # Target (Outcome is the label for diabetes)

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = knn.predict(X_test)

# Step 7: Evaluate the model
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Error Rate
error_rate = 1 - accuracy

# Precision
precision = precision_score(y_test, y_pred)

# Recall
recall = recall_score(y_test, y_pred)

# Print the results
print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
