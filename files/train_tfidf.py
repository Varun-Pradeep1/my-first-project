# -----------------------------
# Week 2: TF-IDF + Logistic Regression / SVM
# Using CSV dataset
# -----------------------------

# 1. Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -----------------------------
# 2. Load CSV Data
# -----------------------------
# Replace 'your_file.csv' with your file path
df = pd.read_csv("your_file.csv")

# Inspect the first few rows
print(df.head())

# Extract features and labels
texts = df["text"].astype(str)    # make sure all text is string
labels = df["target"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# -----------------------------
# 3. TF-IDF Vectorizer
# -----------------------------
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------------
# 4. Logistic Regression
# -----------------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_tfidf, y_train)
y_pred_lr = logreg.predict(X_test_tfidf)

print("===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr, average="weighted"))
print(classification_report(y_test, y_pred_lr))

# -----------------------------
# 5. SVM
# -----------------------------
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)

print("\n===== SVM =====")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("F1 Score:", f1_score(y_test, y_pred_svm, average="weighted"))
print(classification_report(y_test, y_pred_svm))

# -----------------------------
# 6. Error Analysis
# -----------------------------
print("\n===== Misclassified Examples (SVM) =====")
misclassified_indices = np.where(y_pred_svm != y_test)[0]
for i in misclassified_indices[:10]:   # show first 10 mistakes
    print("Text:", X_test.iloc[i])
    print("True:", y_test.iloc[i], "| Predicted:", y_pred_svm[i])
    print("-"*50)