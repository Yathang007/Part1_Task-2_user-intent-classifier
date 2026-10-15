import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from classifier import train_classifier, predict_intent


DATA_PATH = Path("data/intents.csv")
MODEL_PATH = Path("models/intent_classifier.joblib")
RANDOM_STATE = 42


def main():
    # 1. Load the dataset.
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"query", "intent"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV must contain 'query' and 'intent' columns."
        )

    df = df.dropna(subset=["query", "intent"])
    df["query"] = df["query"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[df["query"] != ""]

    if df["intent"].nunique() != 3:
        raise ValueError(
            "Dataset must contain all three intent classes."
        )

    # 2. Split the data into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        df["query"],
        df["intent"],
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=df["intent"],
    )

    # 3. Train only on the training split.
    model = train_classifier(
        X_train.tolist(),
        y_train.tolist(),
    )

    # 4. Evaluate on unseen test examples.
    start_time = time.perf_counter()
    predictions = model.predict(X_test.tolist())
    elapsed = time.perf_counter() - start_time

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    print("\n=== User Intent Classifier Evaluation ===")
    print(f"Total examples: {len(df)}")
    print(f"Training examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(
        f"Average prediction latency: "
        f"{elapsed / len(X_test) * 1000:.3f} ms/query"
    )

    print("\n=== Classification Report ===")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["whole_video", "frame", "clip"],
            zero_division=0,
        )
    )

    print("=== Confusion Matrix ===")
    print(
        pd.DataFrame(
            confusion_matrix(
                y_test,
                predictions,
                labels=["whole_video", "frame", "clip"],
            ),
            index=["actual_whole_video", "actual_frame", "actual_clip"],
            columns=["pred_whole_video", "pred_frame", "pred_clip"],
        )
    )

    # 5. Save the model trained on the training split.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(
        "\nNote: These results use a small synthetic dataset, "
        "not the VideoMME benchmark."
    )


if __name__ == "__main__":
    main()
