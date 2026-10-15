import re

def normalize_query(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Query must be a string")

    text = re.sub(r"\s+", " ", text).strip()
    return text

if __name__ == "__main__":
    result = normalize_query("  Summarize   this video  ")
    print(result)