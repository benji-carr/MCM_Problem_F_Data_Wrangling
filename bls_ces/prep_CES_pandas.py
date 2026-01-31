from pathlib import Path
import pandas as pd

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "data": "ce.data.0.ALLCESSeries",
    "series": "ce.series",
    "industry": "ce.industry",
    "datatype": "ce.datatype",
}

# -----------------------------
# PROXIES (NAICS-BASED)
# -----------------------------
CAREER_NAICS = {
    "mechanic": {
        "naics_prefixes": ["8111"],           # all automotive repair & maintenance
    },
    "graphic_designer": {
        "naics_exact": ["54143"]            # graphic design services
    },
    "software_developer": {
        "naics_exact": ["511210"],            # software publishers
        "naics_prefixes": ["5415"],           # computer systems design & related
    },
}

# CES datatype_text values in YOUR file
EMP_TEXT = "ALL EMPLOYEES, THOUSANDS"
AHE_TEXT = "AVERAGE HOURLY EARNINGS OF ALL EMPLOYEES"

MEASURES = {EMP_TEXT, AHE_TEXT}

# -----------------------------
# HELPERS
# -----------------------------
def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep="\t", engine="python", dtype="string")
    df.columns = [c.strip() for c in df.columns]
    return df


def make_date(year: pd.Series, period: pd.Series) -> pd.Series:
    month = period.str.replace("M", "", regex=False).astype(int)
    return pd.to_datetime(year.astype(str) + "-" + month.astype(str).str.zfill(2) + "-01")


def naics_mask(naics: pd.Series, exact=None, prefixes=None) -> pd.Series:
    s = naics.astype("string").str.strip()
    m = pd.Series(False, index=s.index, dtype="bool")
    if exact:
        exact = [str(x).strip() for x in exact]
        m |= s.isin(exact).fillna(False)
    if prefixes:
        prefixes = tuple(str(p).strip() for p in prefixes)
        m |= s.str.startswith(prefixes, na=False)
    return m


def collapse_career(df: pd.DataFrame) -> pd.DataFrame:
    emp = df[df["datatype_text"] == EMP_TEXT].copy()
    earn = df[df["datatype_text"] == AHE_TEXT].copy()

    # Employment (still in thousands)
    emp_m = (
        emp.groupby("date", as_index=False)["value"]
        .sum()
        .rename(columns={"value": "employment_thousands"})
    )

    # Employment-weighted avg hourly earnings
    earn = earn.merge(
        emp[["date", "industry_code", "value"]]
        .rename(columns={"value": "emp_weight_thousands"}),
        on=["date", "industry_code"],
        how="left",
    )

    earn_m = (
        earn.groupby("date")
        .apply(lambda g: (g["value"] * g["emp_weight_thousands"]).sum()
                         / g["emp_weight_thousands"].sum())
        .to_frame(name="avg_hourly_earnings")   # 👈 KEY FIX
        .reset_index()
    )

    out = (
        emp_m.merge(earn_m, on="date", how="left")
             .sort_values("date")
             .reset_index(drop=True)
    )

    return out


def main():
    # load
    data = read_tsv(RAW_DIR / FILES["data"])
    series = read_tsv(RAW_DIR / FILES["series"])
    industry = read_tsv(RAW_DIR / FILES["industry"])
    datatype = read_tsv(RAW_DIR / FILES["datatype"])

    # normalize names
    series = series.rename(columns={"data_type_code": "datatype_code"})
    datatype = datatype.rename(columns={"data_type_code": "datatype_code", "data_type_text": "datatype_text"})

    # strip key cols
    for df, cols in [
        (data, ["series_id", "period"]),
        (series, ["series_id", "industry_code", "datatype_code"]),
        (industry, ["industry_code", "naics_code"]),
        (datatype, ["datatype_code", "datatype_text"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype("string").str.strip()

    # numeric types
    data["year"] = data["year"].astype(int)
    data["value"] = pd.to_numeric(data["value"], errors="coerce")

    # dictionary (INCLUDES naics_code!)
    series_dict = (
        series[["series_id", "industry_code", "datatype_code", "seasonal"]]
        .merge(industry[["industry_code", "naics_code", "industry_name"]], on="industry_code", how="left")
        .merge(datatype[["datatype_code", "datatype_text"]], on="datatype_code", how="left")
    )

    # --- DEBUG: find graphic design industry codes available in series_dict
    hits = series_dict[
        series_dict["industry_name"].astype("string").str.contains("graphic design", case=False, na=False)
    ][["industry_code", "naics_code", "industry_name"]].drop_duplicates()

    print("\nDEBUG: industries containing 'graphic design' in CES series_dict:")
    print(hits.to_string(index=False))

    # Also check for broader creative categories
    hits2 = series_dict[
        series_dict["industry_name"].astype("string").str.contains("design", case=False, na=False)
    ][["industry_code", "naics_code", "industry_name"]].drop_duplicates().head(50)

    print("\nDEBUG: sample industries containing 'design' (first 50):")
    print(hits2.to_string(index=False))

    # monthly only
    data = data[data["period"].str.match(r"^M(0[1-9]|1[0-2])$")].copy()

    # join
    df = data.merge(series_dict, on="series_id", how="left")
    df["date"] = make_date(df["year"], df["period"])

    # normalize text and filter measures
    df["datatype_text"] = df["datatype_text"].astype("string").str.strip()
    df["naics_code"] = df["naics_code"].astype("string").str.strip()

    df = df[df["datatype_text"].isin(MEASURES)].copy()

    # quick debug counts
    print("\nDEBUG: measure counts")
    print(df["datatype_text"].value_counts().head(10))

    for career, spec in CAREER_NAICS.items():
        m = naics_mask(df["naics_code"], exact=spec.get("naics_exact"), prefixes=spec.get("naics_prefixes"))
        career_df = df[m].copy()

        if career_df.empty:
            print(f"⚠️ {career}: no rows matched NAICS filters {spec}")
            # show a hint
            print("Sample NAICS codes present:", df["naics_code"].dropna().drop_duplicates().head(20).to_list())
            continue

        ts = collapse_career(career_df)
        out_path = OUT_DIR / f"{career}_monthly_ces.csv"
        ts.to_csv(out_path, index=False)
        print(f"✓ Wrote {out_path} ({len(ts)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
