# Mini Project - Build a machine learning model that predicts the type of people who survived
# the Titanic shipwreck using passenger data (i.e. name, age, gender, socio-economic class, etc.).
# Dataset Link: https://www.kaggle.com/competitions/titanic/data

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Explore the first few rows of the dataset
print(train_df.head())

# Check for missing values
print(train_df.isnull().sum())

# Fill missing values for Age with the median (since it's a numerical column)
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Fill missing values for Embarked with the most frequent value
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Drop 'Name', 'Ticket', and 'Cabin' columns as they might not be relevant for this model
train_df = train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Convert categorical columns (Gender and Embarked) into numerical values
label_encoder = LabelEncoder()
train_df['Sex'] = label_encoder.fit_transform(train_df['Sex'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])

# Check the cleaned data
print(train_df.head())

# Select features and target variable
X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_df['Survived']  # Target variable (0 = Did not survive, 1 = Survived)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the train-test split
print(X_train.shape, X_test.shape)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Preprocess the test set similarly as the train set
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)
test_df['Sex'] = label_encoder.transform(test_df['Sex'])
test_df['Embarked'] = label_encoder.transform(test_df['Embarked'])

# Select relevant features from the test set
X_test_final = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Make predictions
predictions = rf_model.predict(X_test_final)

# Prepare the final output for submission
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})

# Save the predictions to a CSV file for submission
submission.to_csv('titanic_predictions.csv', index=False)

print(submission.head())
