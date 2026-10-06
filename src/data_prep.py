"""Load & agregasi data transaksi menjadi panel bulanan dan seri harian."""
import duckdb
import pandas as pd

from . import config as C
from .categories import MCC_TO_CATEGORY, GAMBLING_MCC


def _money(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("$", "", regex=False).astype(float)


def load_users() -> pd.DataFrame:
    u = pd.read_csv(C.USERS_CSV)
    for c in ["per_capita_income", "yearly_income", "total_debt"]:
        u[c] = _money(u[c])
    return u.rename(columns={"id": "client_id"})


def load_cards() -> pd.DataFrame:
    c = pd.read_csv(C.CARDS_CSV)
    c["credit_limit"] = _money(c["credit_limit"])
    return c


def client_static_features() -> pd.DataFrame:
    """Fitur statis per nasabah (profil & kartu) yang diketahui sebelum periode forecast."""
    u = load_users()
    c = load_cards()
    cagg = c.groupby("client_id").agg(
        n_cards=("id", "count"),
        n_credit_cards=("card_type", lambda s: (s == "Credit").sum()),
        total_credit_limit=("credit_limit", lambda s: s[c.loc[s.index, "card_type"] == "Credit"].sum()),
        total_debit_limit=("credit_limit", lambda s: s[c.loc[s.index, "card_type"] != "Credit"].sum()),
    ).reset_index()
    out = u[["client_id", "current_age", "retirement_age", "gender", "per_capita_income",
             "yearly_income", "total_debt", "credit_score", "num_credit_cards"]].merge(cagg, on="client_id", how="left")
    out["monthly_income"] = out["yearly_income"] / 12
    out["dti"] = out["total_debt"] / out["yearly_income"].clip(lower=1)
    out["is_female"] = (out.pop("gender") == "Female").astype(int)
    return out


def _con():
    con = duckdb.connect()
    con.execute("SET enable_progress_bar=false")
    mcc_df = pd.DataFrame({"mcc": list(MCC_TO_CATEGORY), "category": list(MCC_TO_CATEGORY.values())})
    con.register("mcc_map", mcc_df)
    con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{C.TXN_PARQUET}')")
    return con


def monthly_category_panel() -> pd.DataFrame:
    """Panel client x kategori x bulan.

    - Transaksi dengan `errors` (ditolak: saldo kurang, PIN salah, dll) TIDAK dihitung sebagai belanja.
    - amount > 0  -> pengeluaran (gross expense)
    - amount < 0  -> refund/kredit balik (inflow)
    Bulan tanpa transaksi diisi 0 (grid lengkap sejak bulan pertama nasabah aktif).
    """
    con = _con()
    df = con.execute("""
        SELECT t.client_id, date_trunc('month', ts)::DATE AS month, COALESCE(m.category,'services') AS category,
               SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS expense,
               SUM(CASE WHEN amount < 0 THEN -amount ELSE 0 END) AS refund,
               SUM(CASE WHEN amount > 0 THEN 1 ELSE 0 END) AS n_txn
        FROM t LEFT JOIN mcc_map m USING (mcc)
        WHERE errors IS NULL
        GROUP BY 1,2,3
    """).df()
    df["month"] = pd.to_datetime(df["month"])
    first = df.groupby("client_id")["month"].min()
    last = pd.Timestamp(C.LAST_COMPLETE_MONTH)
    months = pd.date_range(df["month"].min(), last, freq="MS")
    cats = sorted(df["category"].unique())
    grid = pd.MultiIndex.from_product([first.index, months, cats], names=["client_id", "month", "category"]).to_frame(index=False)
    grid = grid[grid["month"] >= grid["client_id"].map(first)]
    panel = grid.merge(df, on=["client_id", "month", "category"], how="left").fillna({"expense": 0, "refund": 0, "n_txn": 0})
    return panel


def monthly_client_behaviour() -> pd.DataFrame:
    """Sinyal perilaku per nasabah per bulan (untuk fitur & health score)."""
    con = _con()
    g = ",".join(map(str, GAMBLING_MCC))
    df = con.execute(f"""
        SELECT client_id, date_trunc('month', ts)::DATE AS month,
               SUM(CASE WHEN errors LIKE '%Insufficient Balance%' THEN 1 ELSE 0 END) AS n_insufficient,
               SUM(CASE WHEN errors IS NOT NULL THEN 1 ELSE 0 END) AS n_errors,
               SUM(CASE WHEN errors IS NULL AND use_chip = 'Online Transaction' AND amount>0 THEN amount ELSE 0 END) AS online_expense,
               SUM(CASE WHEN errors IS NULL AND mcc IN ({g}) AND amount>0 THEN amount ELSE 0 END) AS gambling_expense,
               COUNT(DISTINCT CASE WHEN errors IS NULL THEN merchant_id END) AS n_merchants
        FROM t GROUP BY 1,2
    """).df()
    df["month"] = pd.to_datetime(df["month"])
    return df


def daily_portfolio_series() -> pd.DataFrame:
    """Total outflow kartu harian seluruh nasabah (untuk forecast likuiditas/cash flow bank)."""
    con = _con()
    df = con.execute("""
        SELECT ts::DATE AS date,
               SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END) AS outflow,
               SUM(CASE WHEN amount < 0 THEN -amount ELSE 0 END) AS refund,
               COUNT(*) AS n_txn
        FROM t WHERE errors IS NULL GROUP BY 1 ORDER BY 1
    """).df()
    df["date"] = pd.to_datetime(df["date"])
    return df
