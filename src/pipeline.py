"""Pipeline produksi: train model final, simpan artefak, dan fungsi `score()` untuk batch job / API / n8n."""
import json
import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import data_prep as D
from . import features as F
from . import forecasting as M
from . import planner as P

MODEL_PATH = C.ARTIFACT_DIR / "models.joblib"


def train_final(sup_cat: pd.DataFrame, sup_tot: pd.DataFrame, cutoff=C.LAST_COMPLETE_MONTH, interval_scale: float = 1.0):
    c = pd.Timestamp(cutoff)
    tr_c, tr_t = M.train_window(sup_cat, c), M.train_window(sup_tot, c)
    models = {
        "cat_mean": M.train_lgb(tr_c, "tweedie"),
        "cat_p90": M.train_lgb(tr_c, "quantile", alpha=0.9),
        "tot_mean": M.train_lgb(tr_t, "tweedie"),
        "tot_p10": M.train_lgb(tr_t, "quantile", alpha=0.1),
        "tot_p90": M.train_lgb(tr_t, "quantile", alpha=0.9),
        "interval_scale": interval_scale,
        "trained_until": str(c.date()),
        "features": F.FEATURES,
    }
    C.ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    (C.ARTIFACT_DIR / "feature_list.json").write_text(json.dumps(F.FEATURES, indent=1), encoding="utf-8")
    return models


def _client_signals(panel, behaviour, origin):
    """Sinyal 12 bulan terakhir per nasabah untuk health score."""
    o = pd.Timestamp(origin)
    w = panel[(panel["month"] > o - pd.DateOffset(months=12)) & (panel["month"] <= o)]
    tot = w.groupby(["client_id", "month"])["expense"].sum()
    g = tot.groupby("client_id")
    sig = pd.DataFrame({"spend_cv_12": g.std() / g.mean().clip(lower=1)})
    w2 = w.assign(type=w["category"].map(lambda c: F.CATEGORY_META[c][1]))
    disc = w2[w2["type"] == "discretionary"].groupby("client_id")["expense"].sum()
    sig["discretionary_share"] = (disc / w2.groupby("client_id")["expense"].sum()).fillna(0)
    sig["refund_mean"] = w.groupby(["client_id", "month"])["refund"].sum().groupby("client_id").mean()
    b = behaviour[(behaviour["month"] > o - pd.DateOffset(months=12)) & (behaviour["month"] <= o)]
    sig["insufficient_12"] = b.groupby("client_id")["n_insufficient"].sum()
    cat_hist = w.groupby(["client_id", "category"])["expense"].mean().rename("mean_12").reset_index()
    return sig.fillna({"insufficient_12": 0}).reset_index(), cat_hist


def score(origin=C.LAST_COMPLETE_MONTH, models=None, panel=None, behaviour=None, static=None, active_only=True):
    """Jalankan forecast + planner untuk semua nasabah pada origin (akhir bulan terakhir yang lengkap).

    Return dict of DataFrames:
      forecast_category : client_id, category, h, target_month, forecast, p90
      forecast_total    : client_id, h, target_month, forecast, p10, p90
      client_plan       : 1 baris per nasabah: cash flow, health score, rekomendasi tabungan
      budget            : budget bulan depan per nasabah x kategori
    """
    models = models or joblib.load(MODEL_PATH)
    panel = panel if panel is not None else D.monthly_category_panel()
    behaviour = behaviour if behaviour is not None else D.monthly_client_behaviour()
    static = static if static is not None else D.client_static_features()
    o = pd.Timestamp(origin)
    s = models["interval_scale"]

    Xc = F.build_supervised(panel, behaviour, static, [o], with_target=False)
    Xt = F.build_supervised(F.client_total_panel(panel), behaviour, static, [o], with_target=False)
    if active_only:  # nasabah yang masih bertransaksi 3 bulan terakhir
        act = Xt.loc[Xt["roll_mean_3"] > 0, "client_id"].unique()
        Xc, Xt = Xc[Xc["client_id"].isin(act)], Xt[Xt["client_id"].isin(act)]

    fc = Xc[["client_id", "category", "h", "target_month"]].copy()
    fc["forecast"] = models["cat_mean"].predict(Xc[F.FEATURES]).clip(0)
    fc["p90"] = np.maximum(models["cat_p90"].predict(Xc[F.FEATURES]), fc["forecast"])
    fc["category"] = fc["category"].astype(str)

    ft = Xt[["client_id", "h", "target_month"]].copy()
    mean = models["tot_mean"].predict(Xt[F.FEATURES])
    p10, p90 = models["tot_p10"].predict(Xt[F.FEATURES]), models["tot_p90"].predict(Xt[F.FEATURES])
    ft["forecast"] = mean
    ft["p10"] = (mean - s * np.maximum(mean - p10, 0)).clip(0)
    ft["p90"] = mean + s * np.maximum(p90 - mean, 0)

    # ---- planner (bulan depan, h=1) ----
    sig, cat_hist = _client_signals(panel, behaviour, o)
    t1 = ft[ft["h"] == 1].rename(columns={"forecast": "exp_forecast_total", "p10": "exp_p10_total", "p90": "exp_p90_total"})
    client = (t1.drop(columns=["h"]).merge(static[["client_id", "monthly_income", "dti", "credit_score", "current_age"]], on="client_id")
              .merge(sig, on="client_id", how="left"))
    client["forecast_month"] = client.pop("target_month").dt.strftime("%B %Y")
    hs = P.health_score(client)
    client = pd.concat([client, hs], axis=1)

    bands, bench = P.peer_benchmark(cat_hist, static)
    c1 = fc[fc["h"] == 1][["client_id", "category", "forecast", "p90"]]
    budget = P.build_budget(c1, client, bench, bands)
    plan = P.savings_plan(client, budget)

    # cash flow 3 bulan ke depan
    cf = ft.merge(static[["client_id", "monthly_income"]], on="client_id").merge(sig[["client_id", "refund_mean"]], on="client_id", how="left")
    cf["net_cash_flow"] = cf["monthly_income"] + cf["refund_mean"].fillna(0) - cf["forecast"]
    cf["net_cash_flow_p10"] = cf["monthly_income"] + cf["refund_mean"].fillna(0) - cf["p90"]
    return {"forecast_category": fc, "forecast_total": ft, "cash_flow": cf, "client_plan": plan, "budget": budget}
