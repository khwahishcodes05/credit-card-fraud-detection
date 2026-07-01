
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# Load Dataset
df = pd.read_csv("creditcard.csv")

# ==========================
# Basic EDA
# ==========================

print(df.head())
print(df.tail())
print(df.shape)
df.info()
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

# ==========================
# Data Cleaning
# ==========================

df = df.drop_duplicates().reset_index(drop=True)

print(df.duplicated().sum())
print(df["Class"].value_counts())

# ==========================
# Class Distribution
# ==========================

plt.figure(figsize=(12, 8))
ax = sns.countplot(data=df, x="Class")

for container in ax.containers:
    ax.bar_label(container)

plt.title("Distribution of Class Variable")
plt.show()

# ==========================
# Features and Target
# ==========================

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Under Sampling (Training Data Only)
# ==========================

train_df = pd.concat([X_train, y_train], axis=1)

legit = train_df[train_df["Class"] == 0]
fraud = train_df[train_df["Class"] == 1]

legit_sample = legit.sample(n=len(fraud), random_state=42)

train_df = pd.concat([legit_sample, fraud]) \
    .sample(frac=1, random_state=42) \
    .reset_index(drop=True)

X_train = train_df.drop("Class", axis=1)
y_train = train_df["Class"]

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Logistic Regression Model
# ==========================

model = LogisticRegression(
    max_iter=5000,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Prediction
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Model Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ==========================
# Cross Validation
# ==========================

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("CV Scores:", scores)
print("Mean CV Accuracy:", scores.mean())