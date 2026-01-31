from __future__ import annotations

import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


BASE_URL = "https://download.bls.gov/pub/time.series/ce/"
DATA_DIR = Path("")
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "out"


FILES = {
    "data": "ce.data.0.ALLCESSeries",
    "series": "ce.series",
    "industry": "ce.industry",
    "datatype": "ce.datatype",
    "period": "ce.period",
}


def download_file(name: str, dest: Path) -> None:
    url = BASE_URL + name
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"✓ Already downloaded: {dest}")
        return

    print(f"↓ Downloading {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"✓ Saved to {dest} ({dest.stat().st_size:,} bytes)")


def read_bls_tsv(path: Path) -> pd.DataFrame:
    """
    BLS time.series files are whitespace-delimited with headers.
    Some columns contain spaces (industry_name, datatype_text), but those are last columns,
    so we can read using regex sep and rely on fixed columns by file type.
    """
    # Use python engine to allow regex sep
    return pd.read_csv(path, sep=r"\s+", engine="python")


def load_tables(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}

    def read_tsv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep="\t", dtype="string")
        df.columns = [c.strip() for c in df.columns]
        return df

    tables["data"] = read_tsv(raw_dir / FILES["data"])
    tables["series"] = read_tsv(raw_dir / FILES["series"])
    tables["industry"] = read_tsv(raw_dir / FILES["industry"])
    tables["datatype"] = read_tsv(raw_dir / FILES["datatype"])
    tables["period"] = read_tsv(raw_dir / FILES["period"])

    return tables


def build_dictionary(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\nSERIES columns:")
    print(tables["series"].columns.tolist())

    print("\nINDUSTRY columns:")
    print(tables["industry"].columns.tolist())

    print("\nDATATYPE columns:")
    print(tables["datatype"].columns.tolist())

    s = tables["series"].copy()
    i = tables["industry"].copy()
    d = tables["datatype"].copy()

    # Normalize column names
    s.columns = [c.strip().lower() for c in s.columns]
    i.columns = [c.strip().lower() for c in i.columns]
    d.columns = [c.strip().lower() for c in d.columns]

    # ---- Identify key columns dynamically
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise KeyError(f"None of {candidates} found in columns: {df.columns.tolist()}")

    series_id_col = find_col(s, ["series_id"])
    industry_col  = find_col(s, ["industry_code", "industry", "industry_cd"])
    datatype_col = find_col(s, ["data_type_code", "data_type_code", "datatype", "datatype_cd"])

    industry_code_col = find_col(i, ["industry_code", "industry", "industry_cd"])
    datatype_code_col = find_col(d, ["data_type_code", "data_type", "data_type_cd"])

    industry_name_col = find_col(i, ["industry_name", "industry_text", "industry_title"])
    datatype_text_col = find_col(d, ["data_type_text", "data_type_name", "data_type_title"])

    # ---- Reduce to essentials and rename
    s = s[[series_id_col, industry_col, datatype_col, "seasonal"]].rename(
        columns={
            series_id_col: "series_id",
            industry_col: "industry_code",
            datatype_col: "datatype_code",
        }
    )

    i = i[[industry_code_col, industry_name_col]].rename(
        columns={
            industry_code_col: "industry_code",
            industry_name_col: "industry_name",
        }
    )

    d = d[[datatype_code_col, datatype_text_col]].rename(
        columns={
            datatype_code_col: "datatype_code",
            datatype_text_col: "datatype_text",
        }
    )

    # ---- Merge
    dict_df = (
        s.merge(i, on="industry_code", how="left")
         .merge(d, on="datatype_code", how="left")
    )

    return dict_df



def make_monthly_date(year: pd.Series, period: pd.Series) -> pd.Series:
    """
    period is like 'M01'...'M12'. We'll map to YYYY-MM-01.
    """
    month = period.str.replace("M", "", regex=False).astype(int)
    return pd.to_datetime(
        year.astype(str) + "-" + month.astype(str).str.zfill(2) + "-01",
        format="%Y-%m-%d",
        errors="coerce",
    )


def extract_proxy_dataset(
    tables: Dict[str, pd.DataFrame],
    series_dict: pd.DataFrame,
    industry_code_predicate,
    datatype_whitelist: Iterable[str],
    seasonal: Optional[str] = None,  # "S" or "U" often; keep None to allow both
) -> pd.DataFrame:
    data = tables["data"].copy()
    data.columns = [c.strip() for c in data.columns]

    # Keep monthly only
    data = data[data["period"].str.match(r"^M(0[1-9]|1[0-2])$")].copy()

    # Convert value to numeric
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    # Join metadata
    df = data.merge(series_dict, on="series_id", how="left")

    # Filter by industry
    df = df[industry_code_predicate(df["industry_code"])].copy()

    # Filter by datatype text
    df = df[df["datatype_text"].isin(list(datatype_whitelist))].copy()

    # Optional seasonal filter
    if seasonal is not None and "seasonal" in df.columns:
        df = df[df["seasonal"] == seasonal].copy()

    # Add date
    df["date"] = make_monthly_date(df["year"], df["period"])

    # Tidy columns
    keep = [
        "date", "year", "period",
        "series_id", "seasonal",
        "industry_code", "industry_name",
        "datatype_code", "datatype_text",
        "value",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].sort_values(["datatype_text", "industry_code", "date"]).reset_index(drop=True)

    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Download required files
    for key, fname in FILES.items():
        download_file(fname, RAW_DIR / fname)

    # 2) Load tables
    tables = load_tables(RAW_DIR)
    series_dict = build_dictionary(tables)

    # 3) Quick metadata search helper: show industry codes that match keywords
    def show_industries(keyword: str, n: int = 25):
        m = series_dict["industry_name"].str.contains(keyword, case=False, na=False)
        print(f"\nTop industries matching '{keyword}':")
        print(series_dict.loc[m, ["industry_code", "industry_name"]].drop_duplicates().head(n).to_string(index=False))

    show_industries("Automotive")
    show_industries("Graphic")
    show_industries("Software")
    show_industries("Computer Systems")

    # 4) Define proxy mappings (you should confirm exact codes exist in your metadata!)
    # We'll keep these flexible: match prefixes or exact codes.
    # IMPORTANT: CES industry_code system is BLS-specific; not always identical to NAICS formatting.
    # So we search by name and then filter by the code(s) you confirm from the printed list.

    # --- You will likely refine these after seeing the industry_code list ---
    mechanic_codes = {"8111"}      # example proxy: Automotive Repair and Maintenance
    graphic_codes  = {"54143", "541430"}  # example proxy: Graphic Design Services
    software_codes = {"511210", "5415"}   # example proxies: Software Publishers; Computer Systems Design

    # Helper to match either exact or prefix (in case CES uses higher-level group codes)
    def code_matches(codes: set[str]):
        def pred(series: pd.Series) -> pd.Series:
            s = series.fillna("")
            return s.isin(codes) | s.apply(lambda x: any(x.startswith(code) for code in codes))
        return pred

    measures = ["All employees", "Average hourly earnings", "Average weekly hours"]

    # 5) Extract datasets
    mechanic_df = extract_proxy_dataset(tables, series_dict, code_matches(mechanic_codes), measures, seasonal=None)
    graphic_df  = extract_proxy_dataset(tables, series_dict, code_matches(graphic_codes), measures, seasonal=None)
    software_df = extract_proxy_dataset(tables, series_dict, code_matches(software_codes), measures, seasonal=None)

    # 6) Save
    mechanic_df.to_csv(OUT_DIR / "mechanic_proxy_monthly.csv", index=False)
    graphic_df.to_csv(OUT_DIR / "graphic_designer_proxy_monthly.csv", index=False)
    software_df.to_csv(OUT_DIR / "software_developer_proxy_monthly.csv", index=False)

    print("\n✓ Wrote:")
    print("  ", OUT_DIR / "mechanic_proxy_monthly.csv")
    print("  ", OUT_DIR / "graphic_designer_proxy_monthly.csv")
    print("  ", OUT_DIR / "software_developer_proxy_monthly.csv")

    # 7) Safety check: if any are empty, it means your industry_code guesses didn't match CES codes.
    for name, df in [("mechanic", mechanic_df), ("graphic", graphic_df), ("software", software_df)]:
        if df.empty:
            print(
                f"\n⚠️ '{name}' dataset is empty. "
                "That usually means the CES industry_code you used doesn't exist as-is in ce.industry. "
                "Use the printed industry matches above to pick the exact codes present."
            )


if __name__ == "__main__":
    main()
