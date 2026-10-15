from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from features import normalize_query

def build_classifier():
    """Create a lightweight text classification pipeline."""
    features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=20000,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                sublinear_tf=True,
                max_features=20000,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def train_classifier(queries, labels):
    """Validate labeled queries and train the classifier."""
    if not queries or not labels:
        raise ValueError("Queries and labels cannot be empty.")

    if len(queries) != len(labels):
        raise ValueError("Queries and labels must have equal lengths.")

    if any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("Every query must be a non-empty string.")

    allowed_labels = {"whole_video", "frame", "clip"}

    if not set(labels).issubset(allowed_labels):
        raise ValueError(f"Labels must belong to {allowed_labels}.")

    if set(labels) != allowed_labels:
        raise ValueError("Training data must contain all three classes.")

    model = build_classifier()
    model.fit(queries, labels)
    return model


def save_classifier(model, path):
    """Save a trained classifier."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_classifier(path):
    """Load a previously saved classifier."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def predict_intent(model, query):
    """Return the predicted intent and class probability scores."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    query = normalize_query(query)

    prediction = model.predict([query])[0]
    probabilities = model.predict_proba([query])[0]
    classifier = model.named_steps["classifier"]

    scores = {
        label: float(score)
        for label, score in zip(classifier.classes_, probabilities)
    }

    return {
        "intent": prediction,
        "scores": scores,
    }