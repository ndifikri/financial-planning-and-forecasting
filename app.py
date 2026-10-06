"""Smart Financial Planner: aplikasi Streamlit.

Jalankan dari folder `solution`:
    pip install -r requirements.txt
    streamlit run app.py

Sumber data: model & cache yang sudah ada di `artifacts/` dan `data/` (hasil notebook).
Tidak perlu data mentah 1,2 GB.
"""
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src import llm_advisor as LLM
from src import pipeline as PL
from src.categories import CATEGORY_META

st.set_page_config(page_title="Smart Financial Planner", page_icon="💰", layout="wide")

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a9a8a2"
BAND_COLORS = {"Sehat": "#1baf7a", "Waspada": "#eda100", "Rentan": "#e34948"}
BAND_ORDER = ["Sehat", "Waspada", "Rentan"]
COMPONENT_LABELS = {
    "s_savings_rate": "Proyeksi savings rate (30%)",
    "s_dti": "Debt-to-income (20%)",
    "s_credit": "Credit score (15%)",
    "s_volatility": "Volatilitas pengeluaran (15%)",
    "s_discretionary": "Porsi diskresioner (10%)",
    "s_liquidity_stress": "Transaksi ditolak (10%)",
}


def usd(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Memuat model & menghitung forecast untuk semua nasabah...")
def load_all():
    panel = pd.read_parquet(C.DATA_DIR / "panel_monthly.parquet")
    beh = pd.read_parquet(C.DATA_DIR / "behaviour_monthly.parquet")
    static = pd.read_parquet(C.DATA_DIR / "client_static.parquet")
    out = PL.score(C.LAST_COMPLETE_MONTH, panel=panel, behaviour=beh, static=static)
    hist = panel.groupby(["client_id", "month"], as_index=False)["expense"].sum()
    metrics_path = C.ARTIFACT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    plan = out["client_plan"].copy()
    plan["health_band"] = plan["health_band"].astype(str)
    return {"plan": plan, "budget": out["budget"], "cash_flow": out["cash_flow"], "hist": hist, "metrics": metrics}


D = load_all()
plan, budget, cash_flow, hist = D["plan"], D["budget"], D["cash_flow"], D["hist"]
forecast_month = plan["forecast_month"].iloc[0]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("💰 Smart Financial Planner")
    st.caption(f"Data s/d Oktober 2019 · forecast {forecast_month} - Januari 2020 · {len(plan):,} nasabah aktif")
    bands = st.multiselect("Filter health band", BAND_ORDER, default=BAND_ORDER)
    ids = plan.loc[plan["health_band"].isin(bands)].sort_values("client_id")["client_id"].tolist()
    if not ids:
        st.warning("Tidak ada nasabah untuk filter ini.")
        st.stop()
    default_idx = ids.index(315) if 315 in ids else 0
    cid = st.selectbox("Pilih nasabah (client_id)", ids, index=default_idx)

    st.divider()
    st.subheader("AI Financial Coach")
    use_llm = st.toggle(f"Gunakan LLM ({C.LLM_MODEL})", value=True,
                        help="Jika API tidak terjangkau, otomatis memakai template cadangan.")
    key_in = st.text_input("OpenAI API key (opsional)", type="password",
                           help="Kosongkan untuk memakai OPENAI_API_KEY atau file .env")
    if key_in:
        os.environ["OPENAI_API_KEY"] = key_in

tab_client, tab_portfolio, tab_model = st.tabs(["👤 Nasabah", "🏦 Portfolio", "📈 Kinerja model"])

# ================================================================ tab nasabah
with tab_client:
    r = plan.set_index("client_id").loc[cid]
    b = budget[budget["client_id"] == cid].sort_values("forecast", ascending=False)
    band_color = BAND_COLORS.get(r["health_band"], GREY)

    st.markdown(f"### Nasabah #{cid} · rencana {forecast_month}")
    st.markdown(
        f"<span style='background:{band_color};color:white;padding:4px 12px;border-radius:12px;font-weight:600'>"
        f"{r['health_band']} · skor {r['health_score']:.0f}</span>&nbsp;&nbsp;{r['recommended_action']}",
        unsafe_allow_html=True)
    st.write("")

    k = st.columns(5)
    k[0].metric("Pendapatan bulanan", usd(r["monthly_income"]))
    k[1].metric("Forecast pengeluaran", usd(r["exp_forecast_total"]),
                help=f"Rentang P10-P90: {usd(r['exp_p10_total'])} - {usd(r['exp_p90_total'])}")
    k[2].metric("Proyeksi surplus", usd(r["surplus_forecast"]),
                delta=f"skenario buruk {usd(r['surplus_pessimistic'])}", delta_color="off")
    k[3].metric("Rekomendasi tabungan", f"{usd(r['recommended_saving'])}/bln",
                delta=f"{r['saving_rate_after_plan']:.0%} dari pendapatan", delta_color="off")
    k[4].metric("Target dana darurat", usd(r["emergency_fund_target"]))

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Pengeluaran bulanan: histori 24 bulan & forecast 3 bulan**")
        h = hist[(hist["client_id"] == cid) & (hist["month"] > pd.Timestamp(C.LAST_COMPLETE_MONTH) - pd.DateOffset(months=24))]
        f = cash_flow[cash_flow["client_id"] == cid].rename(columns={"target_month": "month"})
        band_ch = alt.Chart(f).mark_area(color="#b7d3f6", opacity=0.8).encode(
            x=alt.X("month:T", title=None), y=alt.Y("p10:Q", title="USD", axis=alt.Axis(format="$,.0f")), y2="p90:Q",
            tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("p10:Q", format="$,.0f", title="P10"),
                     alt.Tooltip("p90:Q", format="$,.0f", title="P90")])
        act = alt.Chart(h).mark_line(color="#52514e", strokeWidth=1.8).encode(
            x=alt.X("month:T", title=None), y="expense:Q",
            tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("expense:Q", format="$,.0f", title="aktual")])
        fc = alt.Chart(f).mark_line(color=BLUE, point=True, strokeWidth=2.5).encode(
            x=alt.X("month:T", title=None), y="forecast:Q",
            tooltip=[alt.Tooltip("month:T", format="%b %Y"), alt.Tooltip("forecast:Q", format="$,.0f", title="forecast")])
        inc = alt.Chart(pd.DataFrame({"y": [r["monthly_income"]]})).mark_rule(color="#1baf7a", strokeDash=[6, 4]).encode(y="y:Q")
        st.altair_chart((band_ch + act + fc + inc).properties(height=320), width="stretch")
        st.caption("Abu-abu = aktual · biru = forecast · area biru muda = rentang P10-P90 · garis hijau putus-putus = pendapatan")

    with right:
        st.markdown("**Komponen Financial Health Score (0-100)**")
        comp = pd.DataFrame({"komponen": [COMPONENT_LABELS[c] for c in COMPONENT_LABELS],
                             "skor": [float(r[c]) for c in COMPONENT_LABELS]})
        ch = alt.Chart(comp).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(
            x=alt.X("skor:Q", scale=alt.Scale(domain=[0, 100]), title=None),
            y=alt.Y("komponen:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["komponen", alt.Tooltip("skor:Q", format=".0f")])
        txt = ch.mark_text(align="left", dx=4, color="#1b2433").encode(text=alt.Text("skor:Q", format=".0f"))
        st.altair_chart((ch + txt).properties(height=320), width="stretch")

    # ---- budget planner (what-if)
    st.markdown(f"#### Budget planner {forecast_month}")
    s1, s2, s3 = st.columns(3)
    income = s1.number_input("Pendapatan bulanan (USD)", min_value=0.0, value=float(r["monthly_income"]), step=50.0)
    max_cut = s2.slider("Pemangkasan maks per kategori diskresioner", 0, 50, 25, step=5, format="%d%%") / 100
    save_share = s3.slider("Porsi surplus yang ditabung", 50, 100, 80, step=5, format="%d%%") / 100

    bb = b[["label", "type", "forecast", "peer_median", "p90"]].copy()
    excess = (bb["forecast"] - bb["peer_median"]).clip(lower=0)
    bb["hemat"] = np.where(bb["type"] == "discretionary", np.minimum(excess, max_cut * bb["forecast"]), 0.0)
    bb["budget"] = bb["forecast"] - bb["hemat"]
    bb["alert"] = np.maximum(bb["p90"], bb["budget"])
    surplus = income + r["refund_mean"] - r["exp_forecast_total"]
    after = surplus + bb["hemat"].sum()
    saving = max(0.0, min(save_share * after, PL.P.MAX_SAVING_RATE * income))

    m = st.columns(4)
    m[0].metric("Surplus sebelum budget", usd(surplus))
    m[1].metric("Potensi hemat", usd(bb["hemat"].sum()))
    m[2].metric("Surplus setelah budget", usd(after))
    m[3].metric("Auto-save disarankan", f"{usd(saving)}/bln",
                delta=f"{saving / income:.0%} pendapatan" if income else None, delta_color="off")

    type_label = {"essential": "Esensial", "discretionary": "Diskresioner", "transfer": "Transfer", "other": "Lainnya"}
    show = bb.assign(tipe=bb["type"].map(type_label)).rename(columns={
        "label": "Kategori", "forecast": "Forecast", "peer_median": "Median peer", "hemat": "Saran hemat",
        "budget": "Budget", "alert": "Alert (P90)"})[["Kategori", "tipe", "Forecast", "Median peer", "Saran hemat", "Budget", "Alert (P90)"]]
    money_cols = {c: st.column_config.NumberColumn(c, format="$%.0f") for c in ["Forecast", "Median peer", "Saran hemat", "Budget", "Alert (P90)"]}
    st.dataframe(show, hide_index=True, width="stretch", column_config={"tipe": "Tipe", **money_cols})
    st.caption("Kategori diskresioner yang di atas median peer (kuintil pendapatan sama) dipangkas sebesar selisihnya, "
               "maksimal sesuai slider. Notifikasi dikirim jika realisasi melewati Alert (P90). Auto-save dibatasi maks 30% pendapatan.")

    # ---- AI coach
    st.markdown("#### 🤖 AI Financial Coach")
    if st.button("Generate pesan coaching", type="primary"):
        row = {**plan[plan["client_id"] == cid].iloc[0].to_dict()}
        payload = LLM.build_payload(row, b.to_dict("records"))
        with st.spinner(f"Memanggil {C.LLM_MODEL}..." if use_llm else "Menyusun pesan..."):
            advice, source = LLM.generate_advice(payload, use_llm=use_llm)
        st.session_state["advice"] = {"cid": cid, "advice": advice, "source": source, "payload": payload}

    adv = st.session_state.get("advice")
    if adv and adv["cid"] == cid:
        a, src = adv["advice"], adv["source"]
        if src == "llm":
            st.success(f"Dihasilkan oleh {C.LLM_MODEL} · lolos validator angka")
        else:
            st.info(f"Sumber: {src}. Pesan dari template cadangan (LLM tidak dipakai atau tidak terjangkau).")
        with st.container(border=True):
            st.markdown(f"**{a.get('judul', '')}**")
            st.write(a.get("ringkasan", ""))
            for i in a.get("insight", []):
                st.markdown(f"- {i}")
            st.markdown("**Rekomendasi:**")
            for i, x in enumerate(a.get("rekomendasi", []), 1):
                st.markdown(f"{i}. {x}")
            st.markdown(f"👉 _{a.get('cta', '')}_")
        with st.expander("Payload yang dikirim ke LLM (tanpa PII)"):
            st.json(adv["payload"])

# ================================================================ tab portfolio
with tab_portfolio:
    st.markdown(f"### Ringkasan portfolio · {forecast_month}")
    p = plan
    n = len(p)
    k = st.columns(5)
    k[0].metric("Nasabah aktif", f"{n:,}")
    k[1].metric("Sehat", f"{(p['health_band'] == 'Sehat').mean():.0%}")
    k[2].metric("Rentan", f"{(p['health_band'] == 'Rentan').mean():.0%}")
    k[3].metric("Forecast outflow bulan depan", f"${p['exp_forecast_total'].sum() / 1e6:,.2f} jt")
    k[4].metric("Total rekomendasi auto-save", f"${p['recommended_saving'].sum() / 1e3:,.0f} rb/bln")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distribusi health band**")
        bd = p["health_band"].value_counts().reindex(BAND_ORDER).fillna(0).rename_axis("band").reset_index(name="nasabah")
        ch = alt.Chart(bd).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("band:N", sort=BAND_ORDER, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("nasabah:Q", title="nasabah"),
            color=alt.Color("band:N", scale=alt.Scale(domain=BAND_ORDER, range=[BAND_COLORS[x] for x in BAND_ORDER]), legend=None),
            tooltip=["band", "nasabah"])
        st.altair_chart((ch + ch.mark_text(dy=-8, color="#1b2433").encode(text="nasabah:Q")).properties(height=300), width="stretch")
    with c2:
        st.markdown("**Forecast pengeluaran vs pendapatan seluruh nasabah (3 bulan)**")
        cf = cash_flow.groupby("target_month", as_index=False)[["forecast", "monthly_income"]].sum()
        cf = cf.melt("target_month", var_name="seri", value_name="usd").replace({"forecast": "Forecast pengeluaran", "monthly_income": "Pendapatan"})
        ch = alt.Chart(cf).mark_bar().encode(
            x=alt.X("yearmonth(target_month):O", title=None, axis=alt.Axis(labelAngle=0, format="%b %Y")), xOffset="seri:N", y=alt.Y("usd:Q", title="USD", axis=alt.Axis(format="$~s")),
            color=alt.Color("seri:N", scale=alt.Scale(range=[BLUE, "#1baf7a"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("yearmonth(target_month):O", title="bulan"), "seri", alt.Tooltip("usd:Q", format="$,.0f")])
        st.altair_chart(ch.properties(height=300), width="stretch")

    st.markdown("**Rekomendasi aksi**")
    act = p.groupby("recommended_action").agg(nasabah=("client_id", "count"), rata2_surplus=("surplus_forecast", "mean"),
                                               rata2_tabungan=("recommended_saving", "mean")).reset_index().sort_values("nasabah", ascending=False)
    act["rata2_surplus"] = act["rata2_surplus"].map(usd)
    act["rata2_tabungan"] = act["rata2_tabungan"].map(usd)
    st.dataframe(act, hide_index=True, width="stretch", column_config={
        "recommended_action": "Aksi", "nasabah": "Nasabah",
        "rata2_surplus": "Rata-rata surplus/bln", "rata2_tabungan": "Rata-rata auto-save/bln"})

    st.markdown("**Early warning: defisit walau sudah ikut budget & band Rentan (prioritas RM)**")
    ew = p[(p["surplus_after_plan"] < 0) & (p["health_band"] == "Rentan")].sort_values("health_score")[
        ["client_id", "health_score", "monthly_income", "exp_forecast_total", "surplus_after_plan", "insufficient_12"]].copy()
    ew["surplus_after_plan"] = -ew["surplus_after_plan"]
    st.caption(f"{len(ew):,} nasabah ({len(ew) / n:.0%} portfolio)")
    st.dataframe(ew, hide_index=True, width="stretch", height=280, column_config={
        "client_id": "Nasabah", "health_score": st.column_config.NumberColumn("Skor", format="%.0f"),
        "monthly_income": st.column_config.NumberColumn("Pendapatan", format="$%.0f"),
        "exp_forecast_total": st.column_config.NumberColumn("Forecast pengeluaran", format="$%.0f"),
        "surplus_after_plan": st.column_config.NumberColumn("Defisit setelah budget", format="$%.0f"),
        "insufficient_12": st.column_config.NumberColumn("Transaksi ditolak 12 bln", format="%d")})
    st.download_button("⬇️ Unduh daftar early warning (CSV)", ew.to_csv(index=False).encode("utf-8"),
                       file_name="early_warning_rm.csv", mime="text/csv")

# ================================================================ tab model
with tab_model:
    st.markdown("### Kinerja model (backtest rolling-origin, Nov 2018 - Okt 2019)")
    mt = D["metrics"]
    if not mt:
        st.warning("artifacts/metrics.json belum ada. Jalankan notebook terlebih dahulu.")
    else:
        k = st.columns(3)
        k[0].metric("Coverage interval 80% (data uji)", f"{mt.get('coverage_80_test', float('nan')):.1%}")
        k[1].metric("AUC Health Score (stres 12 bln)", f"{mt.get('health_auc', float('nan')):.2f}")
        k[2].metric("Faktor kalibrasi interval", f"x{mt.get('interval_scale', 1):.2f}")
        for key, title in [("client_total", "Total pengeluaran per nasabah (cash flow)"),
                           ("category", "Nasabah x kategori (budget)"), ("portfolio", "Outflow harian portfolio (30 hari)")]:
            df = pd.DataFrame(mt.get(key, []))
            if df.empty:
                continue
            st.markdown(f"**{title}**")
            st.dataframe(df, hide_index=True, width="stretch")
    for img, cap in [("model_comparison.png", "WAPE per model"), ("forecast_examples.png", "Contoh forecast vs aktual"),
                     ("shap_importance.png", "Fitur paling berpengaruh (SHAP)")]:
        path = C.FIG_DIR / img
        if path.exists():
            st.image(str(path), caption=cap)
    st.caption("Data sintetis (Kaggle, Caixabank Tech) dalam USD. Pendapatan = yearly_income / 12 (asumsi).")
