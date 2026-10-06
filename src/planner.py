"""Budget planner, cash flow projection, financial health score, dan savings recommendation.

Lapisan ini mengubah output forecast menjadi angka yang bisa ditindaklanjuti nasabah/RM.
Semua aturan bersifat transparan (rule-based) di atas forecast ML, agar mudah diaudit.
"""
import numpy as np
import pandas as pd

from .categories import CATEGORY_META

# ---------- parameter kebijakan (bisa dikalibrasi tim bisnis) ----------
HEALTH_WEIGHTS = {"savings_rate": 0.30, "dti": 0.20, "credit": 0.15, "volatility": 0.15,
                  "discretionary": 0.10, "liquidity_stress": 0.10}
BANDS = [(70, "Sehat"), (50, "Waspada"), (0, "Rentan")]
MAX_CUT_PER_CATEGORY = 0.25        # pemotongan maksimum yang disarankan per kategori diskresioner
TARGET_SAVINGS_RATE = 0.10         # target minimal tabungan 10% pendapatan
EMERGENCY_FUND_MONTHS = 3          # dana darurat = 3x pengeluaran esensial bulanan
MAX_SAVING_RATE = 0.30             # rekomendasi auto-save dibatasi maks 30% pendapatan (realistis & tidak memberatkan)


def _lin(x, bad, good):
    """Skor 0-100 linear antara nilai 'bad' (0) dan 'good' (100)."""
    s = (np.asarray(x, float) - bad) / (good - bad)
    return np.clip(s, 0, 1) * 100


def health_components(df: pd.DataFrame) -> pd.DataFrame:
    """df: satu baris per nasabah dengan kolom:
    monthly_income, exp_forecast_total, refund_mean, dti, credit_score, spend_cv_12,
    discretionary_share, insufficient_12
    """
    savings_rate = (df["monthly_income"] + df["refund_mean"] - df["exp_forecast_total"]) / df["monthly_income"].clip(lower=1)
    comp = pd.DataFrame(index=df.index)
    comp["s_savings_rate"] = _lin(savings_rate, -0.25, 0.20)
    comp["s_dti"] = _lin(df["dti"], 4.0, 0.5)
    comp["s_credit"] = _lin(df["credit_score"], 580, 800)
    comp["s_volatility"] = _lin(df["spend_cv_12"], 0.50, 0.10)
    comp["s_discretionary"] = _lin(df["discretionary_share"], 0.40, 0.15)
    comp["s_liquidity_stress"] = _lin(df["insufficient_12"], 24, 2)
    comp["projected_savings_rate"] = savings_rate
    return comp


def health_score(df: pd.DataFrame, weights=HEALTH_WEIGHTS) -> pd.DataFrame:
    comp = health_components(df)
    score = sum(comp[f"s_{k}"] * w for k, w in weights.items())
    out = comp.copy()
    out["health_score"] = score.round(1)
    out["health_band"] = pd.cut(score, bins=[-1, 50, 70, 101], labels=["Rentan", "Waspada", "Sehat"], right=False)
    return out


def peer_benchmark(cat_hist: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Median rasio pengeluaran kategori / pendapatan per kelompok pendapatan (kuintil).
    cat_hist: client_id, category, mean_12 (rata2 12 bln terakhir)."""
    st = static[["client_id", "monthly_income"]].copy()
    st["income_band"] = pd.qcut(st["monthly_income"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    d = cat_hist.merge(st, on="client_id")
    d["ratio"] = d["mean_12"] / d["monthly_income"].clip(lower=1)
    bench = d.groupby(["income_band", "category"], observed=True)["ratio"].median().rename("peer_ratio").reset_index()
    return st[["client_id", "income_band"]], bench


def build_budget(cat_fc: pd.DataFrame, client: pd.DataFrame, bench: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """Budget bulan depan per kategori.

    cat_fc: client_id, category, forecast (mean h=1), p90 (h=1)
    client: client_id, monthly_income, projected surplus info
    Aturan:
    - kategori esensial / transfer / lainnya: budget = forecast (tidak dipotong)
    - kategori diskresioner: jika di atas median peer (kelompok pendapatan sama), sarankan pemotongan
      sebesar selisihnya, maksimal 25% dari forecast.
    - batas alert = P90 forecast (pengeluaran di atas ini memicu notifikasi "over budget").
    """
    d = cat_fc.merge(bands, on="client_id").merge(bench, on=["income_band", "category"], how="left")
    d = d.merge(client[["client_id", "monthly_income"]], on="client_id")
    d["type"] = d["category"].map(lambda c: CATEGORY_META[c][1])
    d["label"] = d["category"].map(lambda c: CATEGORY_META[c][0])
    peer_amt = d["peer_ratio"] * d["monthly_income"]
    excess = (d["forecast"] - peer_amt).clip(lower=0)
    cut = np.minimum(excess, MAX_CUT_PER_CATEGORY * d["forecast"])
    d["suggested_cut"] = np.where(d["type"] == "discretionary", cut, 0.0)
    d["budget"] = d["forecast"] - d["suggested_cut"]
    d["alert_threshold"] = np.maximum(d["p90"], d["budget"])
    d["peer_median"] = peer_amt
    return d


def savings_plan(client: pd.DataFrame, budget: pd.DataFrame) -> pd.DataFrame:
    """Rekomendasi tabungan bulanan per nasabah.

    surplus_forecast = pendapatan + refund - forecast pengeluaran (mean)
    surplus_after_plan = surplus_forecast + total suggested_cut
    recommended_saving = min(80% x surplus_after_plan, 30% pendapatan), min 0  (sisakan buffer)
    """
    cuts = budget.groupby("client_id")["suggested_cut"].sum().rename("total_cut")
    ess = budget[budget["type"] == "essential"].groupby("client_id")["forecast"].sum().rename("essential_monthly")
    d = client.set_index("client_id").join(cuts).join(ess).fillna({"total_cut": 0})
    d["surplus_forecast"] = d["monthly_income"] + d["refund_mean"] - d["exp_forecast_total"]
    d["surplus_pessimistic"] = d["monthly_income"] + d["refund_mean"] - d["exp_p90_total"]
    d["surplus_after_plan"] = d["surplus_forecast"] + d["total_cut"]
    d["recommended_saving"] = np.minimum(0.8 * d["surplus_after_plan"], MAX_SAVING_RATE * d["monthly_income"]).clip(lower=0)
    d["saving_rate_after_plan"] = d["recommended_saving"] / d["monthly_income"].clip(lower=1)
    d["emergency_fund_target"] = EMERGENCY_FUND_MONTHS * d["essential_monthly"]
    d["months_to_emergency_fund"] = np.where(d["recommended_saving"] > 0,
                                             d["emergency_fund_target"] / d["recommended_saving"].replace(0, np.nan), np.nan)

    def action(r):
        if r["surplus_after_plan"] < 0:
            return "DEFISIT: amankan arus kas, pangkas diskresioner & hindari utang konsumtif baru"
        if r["surplus_forecast"] < 0:
            return "SEIMBANGKAN: ikuti budget hemat untuk menutup defisit, lalu auto-save kecil"
        if r["saving_rate_after_plan"] >= 0.20 and r["surplus_after_plan"] > 0:
            return "SURPLUS TINGGI: auto-debit tabungan berencana + tawarkan deposito/investasi"
        if r["saving_rate_after_plan"] >= TARGET_SAVINGS_RATE:
            return "SURPLUS: auto-debit tabungan berencana setelah gajian"
        return "SURPLUS TIPIS: bangun dana darurat dulu, mulai auto-save nominal kecil"
    d["recommended_action"] = d.apply(action, axis=1)
    return d.reset_index()
