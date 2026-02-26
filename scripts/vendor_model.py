from pathlib import Path

from sentence_transformers import SentenceTransformer


def main() -> None:
    model_name = "all-MiniLM-L6-v2"
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "app" / "smart_tool_select" / "models" / model_name
    target.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name)
    model.save(str(target))

    print(f"Saved model to {target}")


if __name__ == "__main__":
    main()
