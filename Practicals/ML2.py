# Classify the email using the binary classification method. Email Spam detection has two
# states: a) Normal State – Not Spam, b) Abnormal State – Spam. Use K-Nearest Neighbors and
# Support Vector Machine for classification. Analyze their performance.
# Dataset link: The emails.csv dataset on the Kaggle


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('emails.csv')

# Check the first few rows of the data
print(df.head())

# Step 3: Preprocessing the data
# Assume 'text' column contains the email text, and 'label' column contains 'spam' or 'ham' (not spam)
df = df[['text', 'label']]

# Mapping the labels: spam -> 1, ham -> 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Step 4: Convert text data to numerical features (using TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train and evaluate K-Nearest Neighbors (KNN) model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Step 7: Train and evaluate Support Vector Machine (SVM) model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Step 8: Evaluate the models
def evaluate_model(y_true, y_pred, model_name):
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Evaluate KNN model
evaluate_model(y_test, y_pred_knn, "K-Nearest Neighbors (KNN)")

# Evaluate SVM model
evaluate_model(y_test, y_pred_svm, "Support Vector Machine (SVM)")
