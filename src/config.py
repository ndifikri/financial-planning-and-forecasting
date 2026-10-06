"""Konfigurasi global proyek Financial Planning & Forecasting."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
FIG_DIR = ROOT / "figures"

# File mentah Kaggle: dicari di data/ dulu, lalu di ../archive (struktur folder assessment)
_RAW = DATA_DIR if (DATA_DIR / "users_data.csv").exists() else ROOT.parent / "archive"
RAW_TXN_CSV = _RAW / "transactions_data.csv"
USERS_CSV = _RAW / "users_data.csv"
CARDS_CSV = _RAW / "cards_data.csv"
MCC_JSON = _RAW / "mcc_codes.json"
TXN_PARQUET = DATA_DIR / "transactions.parquet"   # hasil konversi CSV -> Parquet (dibuat otomatis)

SEED = 42
LAST_COMPLETE_MONTH = "2019-10-01"   # data berakhir 2019-10-31
HORIZONS = [1, 2, 3]                  # forecast 1-3 bulan ke depan (direct strategy)
N_LAGS = 12
TRAIN_ORIGINS_YEARS = 4               # jendela origin training (bulan origin terakhir 4 tahun)
BACKTEST_CUTOFFS = ["2018-10-01", "2019-01-01", "2019-04-01", "2019-07-01"]  # origin terakhir yg diketahui
QUANTILES = [0.1, 0.5, 0.9]

# LLM
LLM_MODEL = "gpt-6-luna"
