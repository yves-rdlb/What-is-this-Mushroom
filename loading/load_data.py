import pandas as pd
import os

def load_mushroom_data(data_dir: str, target: str = "label"):
    """
    Load the mushroom dataset from train/val/test CSVs
    input:
    data_dir : str Path to the folder containing train.csv, val.csv, test.csv
    target : str, optional
        Name of the target column (default='class').

output:
    dict -> A dictionary with keys 'train', 'val', 'test'.
        Each value is a tuple (X, y) where:
            X = pd.DataFrame of features
            y = pd.Series of target labels
    """
    splits = {}
    for split in ["train", "val", "test"]:
        file_path = os.path.join(data_dir, f"{split}.csv")
        df = pd.read_csv(file_path)
        X = df.drop(columns=[target])
        y = df[target]
        splits[split] = (X, y)
    return splits
