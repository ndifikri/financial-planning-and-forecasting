"""AI Financial Coach berbasis LLM (gpt-6-luna).

Peran LLM dalam solusi:
1. `generate_advice()`  - menerjemahkan angka forecast, budget, health score & rekomendasi tabungan
   menjadi pesan coaching personal Bahasa Indonesia (untuk push notif / in-app / script RM).
2. `classify_new_mcc()` - mengusulkan kategori budget untuk kode merchant (MCC) baru.

Prinsip desain (guardrail):
- LLM TIDAK menghitung angka. Semua angka berasal dari model ML + planner (deterministik),
  dikirim sebagai JSON; LLM hanya menulis narasi.
- Output wajib JSON terstruktur; dicek dengan `validate_numbers()` -> setiap angka rupiah/dolar
  di output harus ada di input (anti-halusinasi). Jika gagal -> fallback template.
- Tanpa data identitas (nama, alamat, nomor kartu) yang dikirim ke LLM; hanya client_id & agregat.
"""
from __future__ import annotations

import json
import os
import re

from . import config as C
from .categories import CATEGORY_META

SYSTEM_PROMPT = """Kamu adalah "Asisten Keuangan" dari sebuah bank di Indonesia.
Tugasmu: menulis pesan coaching keuangan yang personal, hangat, singkat, dan bisa ditindaklanjuti
berdasarkan DATA JSON yang diberikan.

Aturan wajib:
1. Gunakan HANYA angka yang ada di DATA. Jangan menghitung, membulatkan ulang, atau mengarang angka baru.
   Tulis nominal persis seperti field yang berakhiran "_fmt".
2. Jangan menjanjikan imbal hasil, jangan memberi saran investasi spesifik (saham/reksa dana tertentu).
3. Jangan menghakimi. Nada suportif, bahasa Indonesia sehari-hari yang sopan (sapa dengan "Anda").
4. Jika health_band "Rentan" atau surplus negatif, prioritaskan langkah pengamanan arus kas.
5. Output HANYA JSON valid dengan skema:
{"judul": str (maks 60 karakter),
 "ringkasan": str (maks 2 kalimat),
 "insight": [str, str] (2 poin, masing-masing maks 1 kalimat),
 "rekomendasi": [str, str, str] (3 langkah konkret),
 "cta": str (ajakan aksi singkat untuk fitur bank: auto-save, atur budget, atau konsultasi RM)}"""


def _fmt(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def build_payload(client_row: dict, budget_rows: list[dict]) -> dict:
    """Siapkan payload JSON ringkas (tanpa PII) untuk LLM. Angka sudah diformat."""
    top = sorted([b for b in budget_rows if b["suggested_cut"] > 1], key=lambda b: -b["suggested_cut"])[:3]
    over = sorted(budget_rows, key=lambda b: -b["forecast"])[:3]
    return {
        "client_id": int(client_row["client_id"]),
        "bulan_forecast": client_row["forecast_month"],
        "health_score": round(float(client_row["health_score"]), 0),
        "health_band": str(client_row["health_band"]),
        "pendapatan_bulanan_fmt": _fmt(client_row["monthly_income"]),
        "forecast_pengeluaran_fmt": _fmt(client_row["exp_forecast_total"]),
        "forecast_pengeluaran_rentang_fmt": f"{_fmt(client_row['exp_p10_total'])} - {_fmt(client_row['exp_p90_total'])}",
        "proyeksi_surplus_fmt": _fmt(client_row["surplus_forecast"]),
        "proyeksi_surplus_skenario_buruk_fmt": _fmt(client_row["surplus_pessimistic"]),
        "rekomendasi_tabungan_bulanan_fmt": _fmt(client_row["recommended_saving"]),
        "target_dana_darurat_fmt": _fmt(client_row["emergency_fund_target"]),
        "aksi_utama": client_row["recommended_action"],
        "status_arus_kas": "defisit" if client_row["surplus_forecast"] < 0 else "surplus",
        "defisit_setelah_ikut_budget_fmt": _fmt(min(client_row["surplus_after_plan"], 0)),
        "kategori_terbesar": [{"kategori": b["label"], "forecast_fmt": _fmt(b["forecast"])} for b in over],
        "peluang_hemat": [{"kategori": b["label"], "forecast_fmt": _fmt(b["forecast"]),
                           "median_peer_fmt": _fmt(b["peer_median"]), "saran_budget_fmt": _fmt(b["budget"]),
                           "potensi_hemat_fmt": _fmt(b["suggested_cut"])} for b in top],
    }


_NUM = re.compile(r"\$\s?[\d.,]+")


def validate_numbers(output: dict, payload: dict) -> list[str]:
    """Kembalikan daftar nominal di output LLM yang TIDAK ada di payload (harus kosong)."""
    allowed = set(m.replace(" ", "") for m in _NUM.findall(json.dumps(payload, ensure_ascii=False)))
    found = set(m.replace(" ", "").rstrip(".,") for m in _NUM.findall(json.dumps(output, ensure_ascii=False)))
    allowed = {a.rstrip(".,") for a in allowed}
    return sorted(found - allowed)


def _client():
    from openai import OpenAI  # import lazy
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        env = next((p for p in [C.ROOT / ".env", C.ROOT.parent / ".env"] if p.exists()), None)
        if env:
            for line in env.read_text(encoding="utf-8").splitlines():
                k, _, v = line.partition("=")
                if k.strip().lower() in ("openai_api_key", "openai-api-key"):
                    key = v.strip().strip('"\'')
    return OpenAI(api_key=key, timeout=30, max_retries=1)


def call_llm(system: str, user: str, model: str = C.LLM_MODEL) -> dict:
    """Panggil OpenAI Responses API (fallback ke Chat Completions) dengan output JSON."""
    import openai
    client = _client()
    try:
        resp = client.responses.create(model=model, instructions=system, input=user,
                                       text={"format": {"type": "json_object"}})
        return json.loads(resp.output_text)
    except openai.BadRequestError:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)


def template_advice(p: dict) -> dict:
    """Fallback deterministik bila LLM tidak tersedia / output gagal validasi."""
    hemat = p["peluang_hemat"]
    rek = []
    if hemat:
        h = hemat[0]
        rek.append(f"Batasi {h['kategori']} di {h['saran_budget_fmt']} bulan ini (potensi hemat {h['potensi_hemat_fmt']}).")
    if p["rekomendasi_tabungan_bulanan_fmt"] != "$0":
        rek.append(f"Aktifkan auto-save {p['rekomendasi_tabungan_bulanan_fmt']} per bulan tepat setelah gajian.")
        rek.append(f"Bangun dana darurat hingga {p['target_dana_darurat_fmt']}.")
    else:
        rek.append("Tunda tabungan rutin dulu; pastikan tagihan & kebutuhan pokok terbayar sebelum belanja lain.")
        rek.append("Hindari cicilan atau utang konsumtif baru, dan jadwalkan konsultasi dengan RM kami.")
    status = "defisit" if p["status_arus_kas"] == "defisit" else "surplus"
    return {
        "judul": f"Rencana keuangan {p['bulan_forecast']}",
        "ringkasan": (f"Pengeluaran Anda bulan depan diperkirakan {p['forecast_pengeluaran_fmt']} "
                      f"(kisaran {p['forecast_pengeluaran_rentang_fmt']}), dengan proyeksi {status} {p['proyeksi_surplus_fmt']}."),
        "insight": [f"Skor kesehatan keuangan Anda {int(p['health_score'])} ({p['health_band']}).",
                    f"Kategori terbesar: {p['kategori_terbesar'][0]['kategori']} ({p['kategori_terbesar'][0]['forecast_fmt']})."],
        "rekomendasi": rek[:3],
        "cta": "Atur budget & auto-save di aplikasi sekarang.",
    }


def generate_advice(payload: dict, use_llm: bool = True) -> tuple[dict, str]:
    """Return (advice_json, source) dengan source in {'llm', 'template', 'template_after_llm_fail'}."""
    if use_llm:
        try:
            user = "DATA:\n" + json.dumps(payload, ensure_ascii=False, indent=1)
            out = call_llm(SYSTEM_PROMPT, user)
            bad = validate_numbers(out, payload)
            required = {"judul", "ringkasan", "insight", "rekomendasi", "cta"}
            if not bad and required <= set(out):
                return out, "llm"
            return template_advice(payload), "template_after_llm_fail"
        except Exception as e:  # jaringan diblokir, rate limit, dsb.
            return template_advice(payload), f"template ({type(e).__name__})"
    return template_advice(payload), "template"


MCC_PROMPT = """Klasifikasikan merchant category code (MCC) berikut ke SATU kategori budget nasabah.
Pilihan kategori (kode): {cats}.
Jawab JSON: {{"mcc": int, "category": kode, "confidence": 0-1, "alasan": str singkat}}."""


def classify_new_mcc(mcc: int, description: str) -> dict:
    cats = ", ".join(f"{k} ({v[0]})" for k, v in CATEGORY_META.items())
    return call_llm(MCC_PROMPT.format(cats=cats), json.dumps({"mcc": mcc, "description": description}))
