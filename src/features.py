"""Feature engineering leakage-safe untuk forecast pengeluaran per nasabah x kategori.

Setiap baris = (client, category, origin month t, horizon h).
Semua fitur dihitung dari data s/d akhir bulan t (moment prediksi), target = expense bulan t+h.
"""
import warnings
import numpy as np
import pandas as pd

from . import config as C
from .categories import CATEGORIES, CATEGORY_META


def to_wide(panel: pd.DataFrame, value: str = "expense"):
    """Panel long -> matriks (series x month). Bulan sebelum nasabah aktif = NaN."""
    w = panel.pivot_table(index=["client_id", "category"], columns="month", values=value, aggfunc="sum")
    w = w.sort_index(axis=1)
    return w


def build_supervised(panel: pd.DataFrame, behaviour: pd.DataFrame, static: pd.DataFrame,
                     origins, horizons=C.HORIZONS, with_target: bool = True) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    W = to_wide(panel, "expense")
    R = to_wide(panel, "refund").reindex_like(W)
    N = to_wide(panel, "n_txn").reindex_like(W)
    months = W.columns
    Y = W.values.astype("float32")
    RF = R.values.astype("float32")
    NT = N.values.astype("float32")
    idx = W.index.to_frame(index=False)

    # total per client (semua kategori)
    tot = panel.groupby(["client_id", "month"])["expense"].sum().unstack().reindex(columns=months)
    T = tot.reindex(idx["client_id"]).values.astype("float32")

    beh = behaviour.set_index(["client_id", "month"])
    INS = beh["n_insufficient"].unstack().reindex(columns=months).reindex(idx["client_id"]).fillna(0).values

    st = static.set_index("client_id")
    stat_cols = ["current_age", "yearly_income", "monthly_income", "total_debt", "dti", "credit_score",
                 "num_credit_cards", "n_credit_cards", "total_credit_limit", "is_female"]
    S = st.reindex(idx["client_id"])[stat_cols].reset_index(drop=True)

    month_pos = {m: i for i, m in enumerate(months)}
    frames = []
    for o in pd.to_datetime(origins):
        t = month_pos[o]
        if t < C.N_LAGS - 1:
            continue
        lags = Y[:, t - C.N_LAGS + 1: t + 1][:, ::-1]           # lag0 = bulan t
        valid = ~np.isnan(lags[:, 2])                           # min. 3 bulan histori
        base = pd.DataFrame({"client_id": idx["client_id"], "category": idx["category"]})
        base["origin"] = o
        f = {}
        for k in range(C.N_LAGS):
            f[f"lag_{k}"] = lags[:, k]
        with np.errstate(all="ignore"):
            f["roll_mean_3"] = np.nanmean(lags[:, :3], 1)
            f["roll_mean_6"] = np.nanmean(lags[:, :6], 1)
            f["roll_mean_12"] = np.nanmean(lags[:, :12], 1)
            f["roll_std_6"] = np.nanstd(lags[:, :6], 1)
            f["roll_std_12"] = np.nanstd(lags[:, :12], 1)
            f["roll_median_12"] = np.nanmedian(lags[:, :12], 1)
            l24 = Y[:, max(0, t - 23): t + 1]
            f["roll_mean_24"] = np.nanmean(l24, 1)
            f["roll_mean_36"] = np.nanmean(Y[:, max(0, t - 35): t + 1], 1)
            f["exp_mean"] = np.nanmean(Y[:, : t + 1], 1)
            wts = 0.85 ** np.arange(C.N_LAGS)                    # EWMA (lag0 bobot terbesar)
            msk = ~np.isnan(lags)
            f["ewm_12"] = np.nansum(np.nan_to_num(lags) * wts, 1) / (msk * wts).sum(1)
            f["max_12"] = np.nanmax(lags, 1)
            f["min_12"] = np.nanmin(lags, 1)
            f["zero_share_12"] = np.nanmean((lags[:, :12] == 0).astype(float), 1)
            f["trend_3v12"] = f["roll_mean_3"] / (f["roll_mean_12"] + 1)
            tl = T[:, t - 11: t + 1][:, ::-1]
            f["client_tot_lag0"] = tl[:, 0]
            f["client_tot_mean_3"] = np.nanmean(tl[:, :3], 1)
            f["client_tot_mean_12"] = np.nanmean(tl[:, :12], 1)
            f["cat_share_12"] = f["roll_mean_12"] / (f["client_tot_mean_12"] + 1)
            f["refund_mean_12"] = np.nanmean(RF[:, t - 11: t + 1], 1)
            f["ntxn_mean_3"] = np.nanmean(NT[:, t - 2: t + 1], 1)
            f["ntxn_mean_12"] = np.nanmean(NT[:, t - 11: t + 1], 1)
            f["insufficient_12"] = np.nansum(INS[:, t - 11: t + 1], 1)
            f["spend_to_income_12"] = f["client_tot_mean_12"] / (S["monthly_income"].values + 1)
            f["tenure_months"] = np.sum(~np.isnan(Y[:, : t + 1]), 1)
        feat = pd.concat([base, pd.DataFrame(f), S], axis=1)[valid]
        for h in horizons:
            fh = feat.copy()
            fh["h"] = h
            tm = t + h
            fh["target_month"] = o + pd.DateOffset(months=h)
            fh["target_moy"] = fh["target_month"].dt.month
            fh["days_in_target_month"] = fh["target_month"].dt.days_in_month
            # same month last year relative to target (diketahui di origin karena 12-h >= 0)
            fh["seasonal_lag"] = Y[valid.nonzero()[0], tm - 12] if tm - 12 >= 0 else np.nan
            if with_target:
                if tm >= len(months):
                    continue
                fh["y"] = Y[valid.nonzero()[0], tm]
            frames.append(fh)
    out = pd.concat(frames, ignore_index=True)
    cats = CATEGORIES + ["ALL"]
    out["cat_type"] = pd.Categorical(out["category"].map(lambda c: CATEGORY_META.get(c, ("", "all"))[1]),
                                     categories=["essential", "discretionary", "transfer", "other", "all"])
    out["category"] = pd.Categorical(out["category"], categories=cats)
    return out


def client_total_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Agregasi panel kategori -> total pengeluaran per nasabah (category='ALL')."""
    tp = panel.groupby(["client_id", "month"], as_index=False)[["expense", "refund", "n_txn"]].sum()
    tp["category"] = "ALL"
    return tp


FEATURES = (
    [f"lag_{k}" for k in range(C.N_LAGS)]
    + ["roll_mean_3", "roll_mean_6", "roll_mean_12", "roll_std_6", "roll_std_12", "roll_median_12",
       "roll_mean_24", "roll_mean_36", "exp_mean", "ewm_12", "max_12", "min_12",
       "zero_share_12", "trend_3v12", "client_tot_lag0", "client_tot_mean_3", "client_tot_mean_12",
       "cat_share_12", "refund_mean_12", "ntxn_mean_3", "ntxn_mean_12", "insufficient_12",
       "spend_to_income_12", "tenure_months",
       "current_age", "yearly_income", "monthly_income", "total_debt", "dti", "credit_score",
       "num_credit_cards", "n_credit_cards", "total_credit_limit", "is_female",
       "h", "target_moy", "days_in_target_month", "seasonal_lag", "category", "cat_type"]
)
CAT_FEATURES = ["category", "cat_type"]
