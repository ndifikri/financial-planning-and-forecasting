"""Forecast outflow kartu harian level portfolio (perencanaan likuiditas / cash flow bank)."""
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import config as C

H = 30
CUTOFFS = ["2019-06-01", "2019-07-01", "2019-08-01", "2019-09-01"]  # forecast 30 hari mulai tanggal ini


def _cal(df):
    d = df["date"]
    out = pd.DataFrame(index=df.index)
    out["dow"] = d.dt.dayofweek
    out["dom"] = d.dt.day
    out["month"] = d.dt.month
    out["doy"] = d.dt.dayofyear
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    out["trend"] = (d - pd.Timestamp("2010-01-01")).dt.days
    return out


def _lag_feats(s: pd.Series, origin_idx: int):
    """Fitur dari histori s/d origin (tanpa melihat masa depan) - dipakai utk semua h."""
    hist = s.iloc[: origin_idx]
    return {"m7": hist.iloc[-7:].mean(), "m28": hist.iloc[-28:].mean(), "m91": hist.iloc[-91:].mean(),
            "m365": hist.iloc[-365:].mean(), "s28": hist.iloc[-28:].std()}


def _supervised(daily: pd.DataFrame, origins_idx):
    rows = []
    s = daily["outflow"]
    cal = _cal(daily)
    for oi in origins_idx:
        lf = _lag_feats(s, oi)
        for h in range(1, H + 1):
            ti = oi + h - 1
            if ti >= len(daily):
                break
            r = {**lf, **cal.iloc[ti].to_dict(), "h": h, "y": s.iloc[ti], "date": daily["date"].iloc[ti], "origin": daily["date"].iloc[oi]}
            rows.append(r)
    return pd.DataFrame(rows)


FEATS = ["m7", "m28", "m91", "m365", "s28", "dow", "dom", "month", "doy", "is_month_end", "h"]


def backtest(daily: pd.DataFrame, cutoffs=CUTOFFS) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    daily = daily.sort_values("date").reset_index(drop=True)
    s = daily.set_index("date")["outflow"].asfreq("D").ffill()
    res = []
    for c in pd.to_datetime(cutoffs):
        oi = int(daily.index[daily["date"] == c][0])
        hist = s.iloc[:oi]
        fut = s.iloc[oi: oi + H]
        f = pd.DataFrame({"date": fut.index, "y": fut.values})
        f["naive"] = hist.iloc[-1]
        f["seasonal_naive_7"] = np.resize(hist.iloc[-7:].values, len(f))
        f["ma_28"] = hist.iloc[-28:].mean()
        f["ma_365"] = hist.iloc[-365:].mean()
        ets = ExponentialSmoothing(hist.iloc[-730:], trend="add", damped_trend=True, seasonal="add",
                                   seasonal_periods=7).fit()
        f["ets"] = ets.forecast(len(f)).values
        # LightGBM global: origin harian selama 3 tahun terakhir sebelum cutoff
        tr_orig = range(oi - 3 * 365, oi - H + 1, 7)
        tr = _supervised(daily, tr_orig)
        tr = tr[tr["date"] < c]
        m = lgb.train({"objective": "l2", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 50,
                       "verbose": -1, "seed": C.SEED, "num_threads": 2}, lgb.Dataset(tr[FEATS], tr["y"]), 400)
        te = _supervised(daily, [oi])
        f["lgbm"] = m.predict(te[FEATS])
        f["cutoff"] = c
        res.append(f)
    return pd.concat(res, ignore_index=True)


def metrics(bt: pd.DataFrame, models=("naive", "seasonal_naive_7", "ma_28", "ma_365", "ets", "lgbm")):
    rows = []
    mae_sn = (bt["seasonal_naive_7"] - bt["y"]).abs().mean()
    for mname in models:
        e = bt[mname] - bt["y"]
        # akurasi total 30 hari (yang relevan utk perencanaan likuiditas bulanan)
        g = bt.groupby("cutoff")[["y", mname]].sum()
        rows.append({"model": mname, "WAPE harian": e.abs().sum() / bt["y"].sum(),
                     "MAE harian ($)": e.abs().mean(), "MASE vs sNaive7": e.abs().mean() / mae_sn,
                     "Error total 30 hari (%)": ((g[mname] - g["y"]).abs() / g["y"]).mean() * 100,
                     "Bias (%)": e.sum() / bt["y"].sum() * 100})
    return pd.DataFrame(rows).set_index("model")
