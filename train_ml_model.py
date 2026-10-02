
from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

PROJECT_DIR = Path(__file__).parent

train_df = pd.read_csv(PROJECT_DIR / "emails_train.csv")
test_df = pd.read_csv(PROJECT_DIR / "emails_test.csv")

# Remove rows with missing email text or labels
train_df = train_df.dropna(subset=["text", "label"])
test_df = test_df.dropna(subset=["text", "label"])

# Build a machine-learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=30000
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("Training SpamShield AI model...")

model.fit(train_df["text"], train_df["label"])

# Evaluate on emails held out from training
predictions = model.predict(test_df["text"])

accuracy = accuracy_score(test_df["label"], predictions)

print(f"\nTest accuracy: {accuracy:.2%}")
print("\nClassification report:")
print(classification_report(test_df["label"], predictions))

# Save the trained model
model_path = PROJECT_DIR / "spamshield_model.pkl"

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"\nModel saved successfully: {model_path}")