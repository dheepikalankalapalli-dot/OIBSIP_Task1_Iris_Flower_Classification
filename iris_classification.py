# Iris Flower Classification

# Import Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# Load Dataset
df = pd.read_csv("iris.csv")

# Display data
print(df.head())

# Dataset information
print(df.info())

print(df.describe())


# Check missing values
print(df.isnull().sum())


# Data Visualization
sns.pairplot(df, hue="species")
plt.show()


# Features and Target
X = df.drop("species", axis=1)
y = df["species"]


# Encode target labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Logistic Regression Model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("Logistic Regression Accuracy:")
print(accuracy_score(y_test, lr_pred))


# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test, dt_pred))


# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

print("\nKNN Accuracy:")
print(accuracy_score(y_test, knn_pred))


# Best Model Evaluation
print("\nClassification Report:")
print(classification_report(y_test, knn_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, knn_pred)

sns.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Prediction Example
sample = np.array([[5.1,3.5,1.4,0.2]])

prediction = knn_model.predict(sample)

print("Predicted Flower Species:")
print(encoder.inverse_transform(prediction))
