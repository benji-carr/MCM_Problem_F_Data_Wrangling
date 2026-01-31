from pathlib import Path
import pandas as pd

# Resolve paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"

FILES = {
    "mechanic": OUT_DIR / "mechanic_monthly_ces.csv",
    "graphic_designer": OUT_DIR / "graphic_designer_monthly_ces.csv",
    "software_developer": OUT_DIR / "software_developer_monthly_ces.csv",
}

print("\n=== CES OUTPUT SANITY CHECK ===")

for name, path in FILES.items():
    print(f"\n--- {name.upper()} ---")

    if not path.exists():
        print(f"❌ File not found: {path}")
        continue

    # Absolute path (what you asked for)
    print(f"📍 File path:\n{path.resolve()}")

    # Load
    df = pd.read_csv(path, parse_dates=["date"])

    # Basic sanity
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Date range:", df["date"].min(), "→", df["date"].max())

    # Quick data checks
    print("Employment min/max:",
          df["employment_thousands"].min(),
          df["employment_thousands"].max())

    print("Hourly earnings min/max:",
          df["avg_hourly_earnings"].min(),
          df["avg_hourly_earnings"].max())

    # Peek
    print("\nHead:")
    print(df.head(3).to_string(index=False))

print("\n=== DONE ===")
