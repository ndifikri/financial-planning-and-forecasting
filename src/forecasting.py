"""Model forecast pengeluaran: baseline, LightGBM global, backtest rolling-origin, metrik."""
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from .features import FEATURES, CAT_FEATURES

BASELINES = {
    "naive (bulan lalu)": "lag_0",
    "seasonal naive (bulan sama thn lalu)": "seasonal_lag",
    "moving avg 3 bln": "roll_mean_3",
    "moving avg 12 bln": "roll_mean_12",
    "moving avg 24 bln": "roll_mean_24",
}

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, seed=C.SEED, verbose=-1, num_threads=2)
N_ROUNDS = 400


def train_lgb(df: pd.DataFrame, objective: str = "tweedie", alpha: float | None = None, n_rounds: int = N_ROUNDS):
    params = {**LGB_PARAMS, "objective": objective}
    if objective == "quantile":
        params["alpha"] = alpha
    ds = lgb.Dataset(df[FEATURES], df["y"], categorical_feature=CAT_FEATURES)
    return lgb.train(params, ds, num_boost_round=n_rounds)


def train_window(sup: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Hanya baris yang target-nya sudah terobservasi pada cutoff (anti-leakage)."""
    start = cutoff - pd.DateOffset(years=C.TRAIN_ORIGINS_YEARS)
    return sup[(sup["target_month"] <= cutoff) & (sup["origin"] >= start)]


def backtest(sup: pd.DataFrame, cutoffs=C.BACKTEST_CUTOFFS, quantiles=(0.1, 0.9), log=print) -> pd.DataFrame:
    """Rolling-origin backtest. Pada tiap cutoff: train dgn data <= cutoff, forecast h=1..3."""
    out = []
    for c in pd.to_datetime(cutoffs):
        tr = train_window(sup, c)
        te = sup[sup["origin"] == c].copy()
        m = train_lgb(tr, "tweedie")
        te["pred_lgb"] = m.predict(te[FEATURES])
        for q in quantiles:
            mq = train_lgb(tr, "quantile", alpha=q)
            te[f"pred_q{int(q*100)}"] = mq.predict(te[FEATURES])
        for name, col in BASELINES.items():
            te[f"pred_{col}"] = te[col].fillna(te["roll_mean_12"])
        te["cutoff"] = c
        out.append(te)
        log(f"cutoff {c.date()} | train {len(tr):,} rows | test {len(te):,} rows")
    return pd.concat(out, ignore_index=True)


# ---------------- metrik ----------------
def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return np.abs(y - p).sum() / np.abs(y).sum()


def metric_table(bt: pd.DataFrame, pred_cols: dict, level_keys=None) -> pd.DataFrame:
    """Metrik per model. level_keys: agregasi sebelum hitung metrik (mis. total per nasabah)."""
    rows = []
    df = bt
    if level_keys is not None:
        df = bt.groupby(level_keys, observed=True)[["y"] + list(pred_cols.values())].sum().reset_index()
    sn = pred_cols.get("seasonal naive (bulan sama thn lalu)")
    mae_sn = np.abs(df["y"] - df[sn]).mean() if sn else np.nan
    for name, col in pred_cols.items():
        e = df[col] - df["y"]
        rows.append({"model": name, "WAPE": wape(df["y"], df[col]), "MAE": e.abs().mean(),
                     "RMSE": np.sqrt((e ** 2).mean()), "Bias (%)": e.sum() / df["y"].sum() * 100,
                     "MASE vs sNaive": e.abs().mean() / mae_sn})
    return pd.DataFrame(rows).set_index("model")


def coverage(y, lo, hi):
    y = np.asarray(y)
    return float(((y >= lo) & (y <= hi)).mean())
