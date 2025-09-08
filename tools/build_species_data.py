#!/usr/bin/env python3
import json, csv, time, sys, re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dateutil import parser as dtparse

import requests
from pygbif import species as gbif_species
from pygbif import occurrences as gbif_occ

# ---------- CONFIG ----------
REPO_ROOT = Path(__file__).resolve().parents[1]  # repo/
LOOKUPS_DIR = REPO_ROOT / "lookups"
DIST_DIR = REPO_ROOT / "distributions"

SPECIES_JSON = LOOKUPS_DIR / "species_classes.json"         # your class list
POINTS_CSV   = DIST_DIR / "species_points.csv"              # output 1
PROFILE_CSV  = LOOKUPS_DIR / "species_profile_seed.csv"     # output 2

# Occurrence retrieval settings
TARGET_REGIONS = []  # e.g. ['GB','IE','FR'] for country codes; empty=[] means global
MAX_POINTS_PER_SPECIES = 250          # keep it reasonable for demo
MIN_POINTS_REQUIRED     = 10          # skip species with fewer than this many points
SAMPLE_SPREAD           = True        # de-duplicate nearby points a bit

# ---------- HELPERS ----------
def norm_species(name: str) -> str:
    return (
        name.strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "")
        .replace("'", "")
    )

def month_from_event_date(s: Optional[str]) -> int:
    if not s:
        return 0
    try:
        dt = dtparse.parse(s, default=datetime(2000,1,1))
        m = dt.month
        return m if 1 <= m <= 12 else 0
    except Exception:
        # sometimes GBIF has 'YYYY', 'YYYY-MM', or messy strings
        m = 0
        m_match = re.search(r"[-/](\d{1,2})[-/]", s)
        if m_match:
            try:
                m = int(m_match.group(1))
                if 1 <= m <= 12: return m
            except: pass
        return 0

def gbif_match_name(scientific_name: str) -> Optional[int]:
    try:
        m = gbif_species.name_backbone(name=scientific_name)
        return m.get("usageKey")
    except Exception:
        return None

def get_common_name_from_gbif_key(taxon_key: int) -> Optional[str]:
    # try vernacular names endpoint
    try:
        data = requests.get(f"https://api.gbif.org/v1/species/{taxon_key}/vernacularNames", timeout=15).json()
        names = data.get("results", [])
        # prefer English
        for n in names:
            if n.get("language") == "eng" and n.get("vernacularName"):
                return n["vernacularName"]
        # else any
        if names:
            return names[0].get("vernacularName")
    except Exception:
        pass
    return None

def get_distribution_strings(taxon_key: int) -> str:
    # GBIF distribution endpoint gives areas (countries/regions); we’ll compress to a short string
    try:
        data = requests.get(f"https://api.gbif.org/v1/species/{taxon_key}/distributions", timeout=20).json()
        areas = [d.get("area") for d in data.get("results", []) if d.get("area")]
        if not areas:
            return ""
        # collapse to a short readable string
        # keep most frequent 6 unique areas
        seen = []
        for a in areas:
            if a not in seen:
                seen.append(a)
        return " • ".join(seen[:6])
    except Exception:
        return ""

def fetch_occurrence_points(taxon_key: int, regions: List[str]) -> List[Dict]:
    params = {
        "taxonKey": taxon_key,
        "hasCoordinate": True,
        "limit": 300,   # will paginate
    }
    if regions:
        # we’ll fetch per region to bias to your geography
        points = []
        for cc in regions:
            off = 0
            while True:
                res = gbif_occ.search(**{**params, "country": cc, "offset": off})
                results = res.get("results", [])
                if not results: break
                points.extend(results)
                off += len(results)
                if off >= MAX_POINTS_PER_SPECIES: break
        return points[:MAX_POINTS_PER_SPECIES]
    else:
        # global
        points = []
        off = 0
        while True:
            res = gbif_occ.search(**{**params, "offset": off})
            results = res.get("results", [])
            if not results: break
            points.extend(results)
            off += len(results)
            if off >= MAX_POINTS_PER_SPECIES: break
        return points[:MAX_POINTS_PER_SPECIES]

def thin_points(rows: List[Dict], grid=0.1) -> List[Dict]:
    """Simple spatial thinning so the heatmap isn’t over-clustered by duplicates.
       grid=0.1 ~ ~11km at equator."""
    seen = set()
    out = []
    for r in rows:
        lat = r.get("decimalLatitude"); lon = r.get("decimalLongitude")
        if lat is None or lon is None: continue
        key = (round(lat / grid, 3), round(lon / grid, 3))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

# ---------- MAIN ----------
def main():
    LOOKUPS_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    if not SPECIES_JSON.exists():
        print(f"Missing species list: {SPECIES_JSON}", file=sys.stderr)
        sys.exit(1)

    species_list: List[str] = json.loads(SPECIES_JSON.read_text())
    # Some lists are dict mapping class index; handle both cases
    if isinstance(species_list, dict):
        species_list = list(species_list.values())

    # Open outputs
    pts_f = open(POINTS_CSV, "w", newline="", encoding="utf-8")
    prof_f = open(PROFILE_CSV, "w", newline="", encoding="utf-8")

    pts_writer = csv.writer(pts_f)
    prof_writer = csv.writer(prof_f)

    pts_writer.writerow(["species", "lat", "lon", "month", "source"])
    prof_writer.writerow(["scientific_name", "common_name", "edibility", "distribution", "habitat", "fruiting_season", "notes", "source_hint"])

    for idx, sci_name in enumerate(species_list, 1):
        sci_name = str(sci_name).strip()
        norm = norm_species(sci_name)
        print(f"[{idx}/{len(species_list)}] {sci_name} -> {norm}")

        # 1) match to GBIF
        key = gbif_match_name(sci_name)
        if not key:
            print(f"  ! no GBIF match")
            continue

        # 2) common name + distribution labels
        common = get_common_name_from_gbif_key(key) or ""
        dist_str = get_distribution_strings(key)

        # 3) occurrences
        try:
            occ = fetch_occurrence_points(key, TARGET_REGIONS)
            if SAMPLE_SPREAD and occ:
                occ = thin_points(occ, grid=0.1)
        except Exception as e:
            print(f"  ! occ error: {e}")
            occ = []

        # write points if enough data
        kept = 0
        months_present = set()
        for r in occ:
            lat = r.get("decimalLatitude"); lon = r.get("decimalLongitude")
            if lat is None or lon is None:
                continue
            m = month_from_event_date(r.get("eventDate")) or (r.get("month") if isinstance(r.get("month"), int) else 0)
            months_present.add(m if 1 <= m <= 12 else 0)
            pts_writer.writerow([norm, lat, lon, m, "GBIF"])
            kept += 1
        if kept < MIN_POINTS_REQUIRED:
            print(f"  ! too few points ({kept}); keeping anyway for now")

        # 4) profile row
        # Edibility/habitat are not consistently available in GBIF; we leave edibility blank (to be filled by your curated edibility CSV)
        # We’ll infer a crude season string from months_present.
        months = sorted([m for m in months_present if m != 0])
        if months:
            # e.g., 7,8,9 -> "Jul–Sep"
            month_names = ["", "Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            season = f"{month_names[months[0]]}–{month_names[months[-1]]}" if len(months) > 1 else month_names[months[0]]
        else:
            season = ""

        prof_writer.writerow([
            sci_name,
            common,
            "",                # edibility left blank here (use your edibility CSV to drive app safety)
            dist_str,
            "",                # habitat TBD (could be scraped from WP later)
            season,
            "",                # notes
            f"GBIF:{key}"
        ])

        # be polite to APIs
        time.sleep(0.15)

    pts_f.close()
    prof_f.close()
    print(f"\nWrote:\n  - {POINTS_CSV}\n  - {PROFILE_CSV}")

if __name__ == "__main__":
    main()
