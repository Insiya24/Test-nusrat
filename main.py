import os
import pandas as pd
from fastapi import FastAPI

# ----------------------------
# CONFIG
# ----------------------------
HF_URL = "https://huggingface.co/datasets/InsiyaMaryam/Makaan-data/resolve/main/Makaan_cleaned_final.csv"
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "Makaan_cleaned_final.csv")

app = FastAPI(title="NUSRAT Backend")

_df_cache = None


# ----------------------------
# DATA LOADER (DOWNLOAD ONCE)
# ----------------------------
def load_dataframe():
    global _df_cache

    if _df_cache is not None:
        return _df_cache

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print("📥 Downloading dataset from HuggingFace...")
        df = pd.read_csv(HF_URL)
        df.to_csv(CSV_PATH, index=False)
        print("✅ Dataset downloaded and stored locally")
    else:
        print("✅ Loading dataset from local disk")

    df = pd.read_csv(CSV_PATH)

    # Minimal required cleaning
    df = df.dropna(subset=["City_name", "Latitude", "Longitude"])

    _df_cache = df
    return df


# ----------------------------
# HEALTH CHECK (OPTIONAL BUT GOOD)
# ----------------------------
@app.get("/")
def health_check():
    return {"status": "NUSRAT backend running"}


# ----------------------------
# TEST ENDPOINT
# ----------------------------
@app.get("/map/cities")
def get_cities():
    df = load_dataframe()

    cities = sorted(df["City_name"].dropna().unique().tolist())

    return {
        "count": len(cities),
        "cities": cities
    }
