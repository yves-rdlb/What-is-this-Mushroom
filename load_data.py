import pandas as pd
from pathlib import Path
from typing import Union

def load_mushroom_data(
    train_path: Union[str, Path] = None,
    edibility_path: Union[str, Path] = None,
) -> pd.DataFrame:
    """
    Load and merge mushroom dataset with edibility mapping.

    Args:
        train_path (str | Path, optional): Path to raw train.csv
        edibility_path (str | Path, optional): Path to species_edibility.csv

    Returns:
        pd.DataFrame with columns:
            - image_path (from train.csv)
            - label (original label from train.csv)
            - species (cleaned name from label)
            - edible (int: 0 = edible, 1 = not edible)
    """
    # Resolve project root relative to this file. This file now lives at repo root,
    # so its parent is the correct base directory for default paths.
    PROJECT_ROOT = Path(__file__).resolve().parent

    if train_path is None:
        train_path = PROJECT_ROOT / "raw_data" / "train.csv"
    if edibility_path is None:
        edibility_path = PROJECT_ROOT / "data" / "metadata" / "species_edibility.csv"

    # Load training data
    df = pd.read_csv(train_path)
    df["species"] = df["label"].str.split("/").str[-1].str.strip()

    # Load edibility map
    ed = pd.read_csv(edibility_path)
    # Normalize column names (strip + lower) and handle missing 'edible'
    ed.columns = ed.columns.str.strip().str.lower()

    # Ensure a 'species' column exists in mapping
    if "species" not in ed.columns:
        alt = [c for c in ed.columns if "species" in c]
        if alt:
            ed = ed.rename(columns={alt[0]: "species"})
        else:
            raise KeyError("species_edibility.csv must contain a 'species' column")

    # Build or normalize the 'edible' column to 0/1 where
    # 0 = edible, 1 = not edible (safer default for unknowns)
    if "edible" not in ed.columns:
        ed["edible"] = 1
    else:
        # If textual, map common labels
        if ed["edible"].dtype == object:
            mapping = {
                "edible": 0,
                "choice": 0,
                "good": 0,
                "ok": 0,
                "poisonous": 1,
                "toxic": 1,
                "inedible": 1,
                "unknown": 1,
                "hallucinogenic": 1,
            }
            ed["edible"] = (
                ed["edible"].astype(str).str.strip().str.lower().map(mapping)
            )
        # Default any unmapped/NaN to 1 (not edible)
        ed["edible"] = ed["edible"].fillna(1).astype(int)

    # Merge
    df = df.merge(ed, on="species", how="left")

    # Unknown species → code 1 (not edible)
    df["edible"] = df["edible"].fillna(1).astype(int)

    return df

def load_data(
    train_path: Union[str, Path] = None,
    edibility_path: Union[str, Path] = None,
) -> pd.DataFrame:
    """
    Alias for load_mushroom_data for convenience.

    Keeps the convention: 0 = edible, 1 = not edible.
    """
    return load_mushroom_data(train_path=train_path, edibility_path=edibility_path)

if __name__ == "__main__":
    df = load_mushroom_data()
    print(df.head())
    print(df["edible"].value_counts())
