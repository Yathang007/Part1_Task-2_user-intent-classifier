
import tempfile
from pathlib import Path

from classifier import (
    train_classifier,
    predict_intent,
    save_classifier,
    load_classifier,
)


def main():
    # Small test dataset for checking the classifier pipeline.
    queries = [
        "Summarize the entire video",
        "Give me an overview of this recording",
        "What is the video mainly about?",
        "What color is the car at 00:42?",
        "What object is visible in this frame?",
        "Read the text displayed on the screen",
        "What happens just before the player scores?",
        "Describe the sequence where the dog jumps",
        "Find the part where the speaker changes slides",
    ]

    labels = [
        "whole_video",
        "whole_video",
        "whole_video",
        "frame",
        "frame",
        "frame",
        "clip",
        "clip",
        "clip",
    ]

    # 1. Train the model.
    model = train_classifier(queries, labels)
    print("PASS: Model trained successfully.")

    # 2. Check predictions.
    test_queries = [
        "Summarize this video",
        "What color is the car?",
        "Describe what happens after the door opens",
    ]

    for query in test_queries:
        result = predict_intent(model, query)
        print(f"\nQuery: {query}")
        print(f"Predicted intent: {result['intent']}")
        print(f"Scores: {result['scores']}")

        assert result["intent"] in {
            "whole_video",
            "frame",
            "clip",
        }

    print("\nPASS: Predictions returned valid labels.")

    # 3. Check saving and reloading the model.
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "model.joblib"

        save_classifier(model, model_path)
        assert model_path.is_file()

        loaded_model = load_classifier(model_path)
        result = predict_intent(
            loaded_model,
            "Summarize the entire video",
        )

        assert result["intent"] in {
            "whole_video",
            "frame",
            "clip",
        }

    print("PASS: Model save and reload work.")
    print("\nAll basic classifier checks passed.")


if __name__ == "__main__":
    main()
