import argparse
from pathlib import Path

from classifier import load_classifier, predict_intent


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "intent_classifier.joblib"
)


def main():
    parser = argparse.ArgumentParser(
        description="Video Query Intent Classifier"
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model",
    )

    args = parser.parse_args()

    try:
        model = load_classifier(args.model)
    except (FileNotFoundError, OSError) as error:
        print(f"Error loading model: {error}")
        return

    print("\n====================================")
    print("     VIDEO INTENT CLASSIFIER")
    print("====================================")
    print("Classes: whole_video | frame | clip")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query = input("Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting classifier.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            print("Please enter a non-empty query.\n")
            continue

        try:
            result = predict_intent(model, query)

            print(f"\nPredicted intent: {result['intent']}")
            print("Class scores:")

            for label, score in sorted(
                result["scores"].items(),
                key=lambda item: item[1],
                reverse=True,
            ):
                print(f"  {label}: {score:.4f}")

            print()

        except ValueError as error:
            print(f"Error: {error}\n")


if __name__ == "__main__":
    main()
