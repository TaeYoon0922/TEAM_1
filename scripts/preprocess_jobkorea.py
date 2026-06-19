import argparse
from pathlib import Path
import sys
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib.util import spec_from_file_location, module_from_spec


app_dir = Path(__file__).resolve().parents[1] / "app"
preproc_path = app_dir / "preprocessing.py"
if not preproc_path.exists():

    for p in app_dir.glob("*.py"):
        try:
            text = p.read_text(encoding="utf8")
        except Exception:
            continue
        if "class Preprocessor" in text:
            preproc_path = p
            break

print(f"Using preprocessing file: {preproc_path}")


if not preproc_path.exists():
    raise FileNotFoundError(f"Could not find preprocessing module in {app_dir}")

spec = spec_from_file_location("app.preprocessing", str(preproc_path))
module = module_from_spec(spec)
spec.loader.exec_module(module)
Preprocessor = module.Preprocessor
print("Loaded Preprocessor class successfully")


def process_file(input_csv: Path, output_csv: Path, cols: list[str]):
    df = pd.read_csv(input_csv)
    print(f"Loaded CSV with {len(df)} rows")
    p = Preprocessor(stopwords={"그", "이", "있다", "없다", "하는"})

    for col in cols:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found in CSV; skipping")
            continue

        tokens_col = f"{col}_tokens"
        nouns_col = f"{col}_nouns"

        def apply_text(x):
            try:
                if pd.isna(x):
                    return []
                return p.preprocess(str(x))
            except Exception:
                return []

        def apply_nouns(x):
            try:
                if pd.isna(x):
                    return []
                return p.extract_nouns(str(x))
            except Exception:
                return []

        df[tokens_col] = df[col].map(apply_text)
        df[nouns_col] = df[col].map(apply_nouns)
        print(f"Processed column '{col}'")


    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), 1):
        if i % 500 == 0:
            print(f"Progress: {i}/{total} rows processed")


    for c in list(df.columns):
        if c.endswith("_tokens") or c.endswith("_nouns"):
            df[c + "_str"] = df[c].map(lambda lst: " ".join(lst) if isinstance(lst, list) else "")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote preprocessed CSV to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cols", default="question,answer", help="Comma-separated columns to preprocess (default: question,answer)")
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_csv = Path(args.output)
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    process_file(input_csv, output_csv, cols)


if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
import sys
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib.util import spec_from_file_location, module_from_spec


app_dir = Path(__file__).resolve().parents[1] / "app"
preproc_path = app_dir / "preprocessing.py"
if not preproc_path.exists():

    for p in app_dir.glob("*.py"):
        try:
            text = p.read_text(encoding="utf8")
        except Exception:
            continue
        if "class Preprocessor" in text:
            preproc_path = p
            break

print(f"Using preprocessing file: {preproc_path}")


if not preproc_path.exists():
    raise FileNotFoundError(f"Could not find preprocessing module in {app_dir}")

spec = spec_from_file_location("app.preprocessing", str(preproc_path))
module = module_from_spec(spec)
spec.loader.exec_module(module)
Preprocessor = module.Preprocessor
print("Loaded Preprocessor class successfully")


def process_file(input_csv: Path, output_csv: Path, cols: list[str]):
    df = pd.read_csv(input_csv)
    print(f"Loaded CSV with {len(df)} rows")
    p = Preprocessor(stopwords={"그", "이", "있다", "없다", "하는"})

    for col in cols:
        if col not in df.columns:
            print(f"Warning: column '{col}' not found in CSV; skipping")
            continue

        tokens_col = f"{col}_tokens"
        nouns_col = f"{col}_nouns"

        def apply_text(x):
            try:
                if pd.isna(x):
                    return []
                return p.preprocess(str(x))
            except Exception:
                return []

        def apply_nouns(x):
            try:
                if pd.isna(x):
                    return []
                return p.extract_nouns(str(x))
            except Exception:
                return []

        df[tokens_col] = df[col].map(apply_text)
        df[nouns_col] = df[col].map(apply_nouns)
        print(f"Processed column '{col}'")


    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), 1):
        if i % 500 == 0:
            print(f"Progress: {i}/{total} rows processed")


    for c in list(df.columns):
        if c.endswith("_tokens") or c.endswith("_nouns"):
            df[c + "_str"] = df[c].map(lambda lst: " ".join(lst) if isinstance(lst, list) else "")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote preprocessed CSV to {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cols", default="question,answer", help="Comma-separated columns to preprocess (default: question,answer)")
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_csv = Path(args.output)
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    process_file(input_csv, output_csv, cols)


if __name__ == "__main__":
    main()
