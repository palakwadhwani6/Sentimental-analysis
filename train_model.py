# train.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import re, string, os

# ---------- Load Dataset ----------
df = pd.read_csv("Reviews.csv")
df = df[['Text', 'Score']].dropna()
df = df[df['Score'] != 3]  # remove neutral
df['Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)

# ---------- Text Cleaning ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Negation handling
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)
    return text

df['CleanText'] = df['Text'].apply(clean_text)

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    df['CleanText'], df['Sentiment'], test_size=0.2, random_state=42, stratify=df['Sentiment']
)

# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- Model ----------
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# ---------- Predictions ----------
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# ---------- Save Model ----------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# ---------- Graphs ----------
os.makedirs("static/images", exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("static/images/confusion_matrix.png")
plt.close()

# Accuracy Bar
plt.figure(figsize=(3, 3))
plt.bar(["Accuracy"], [acc * 100])
plt.title("Model Accuracy")
plt.ylabel("Percentage")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("static/images/accuracy.png")
plt.close()

print("✅ Model, vectorizer, and graphs saved successfully.")
