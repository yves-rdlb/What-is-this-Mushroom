import json
from pathlib import Path

import pandas as pd

# ----- Paths anchored at project root -----
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # <- go up from /scripts to repo root
LOOKUPS_DIR  = PROJECT_ROOT / "lookups"
csv_path     = LOOKUPS_DIR / "species_edibility.csv"
json_path    = LOOKUPS_DIR / "species_classes.json"

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"CSV path:     {csv_path}")
print(f"JSON out:     {json_path}")

if not csv_path.exists():
    raise FileNotFoundError(f"Couldn't find CSV at {csv_path}. "
                            f"Does {LOOKUPS_DIR} contain species_edibility.csv?")

# ----- Load CSV (no header in your file) -----
df = pd.read_csv(csv_path, header=None, names=["species", "edible"])
df["species"] = (
    df["species"]
    .astype(str).str.strip().str.lower().str.replace(" ", "_")
)

# ----- Build unique class list -----
species = df["species"].dropna().unique().tolist()

# ----- Write JSON -----
json_path.parent.mkdir(parents=True, exist_ok=True)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(species, f, indent=2)

print(f"✅ Wrote {len(species)} classes to {json_path}")
