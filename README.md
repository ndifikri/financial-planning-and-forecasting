# 💰 Smart Financial Planner

Dashboard Streamlit untuk **perencanaan keuangan nasabah berbasis forecasting**. Aplikasi ini memprediksi pengeluaran 3 bulan ke depan, menyusun budget per kategori, menghitung Financial Health Score, memberi rekomendasi tabungan, dan menghasilkan pesan coaching personal dengan LLM.

Dibuat sebagai solusi studi kasus *Financial Planning & Forecasting* (Data Science Assessment).

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-9ACD32)
![DuckDB](https://img.shields.io/badge/DuckDB-1.1%2B-FFF000?logo=duckdb&logoColor=black)

---

## ✨ Fitur Utama

### 👤 Tab Nasabah
- **Profil keuangan ringkas**: pendapatan bulanan, forecast pengeluaran (dengan rentang P10-P90), proyeksi surplus, rekomendasi tabungan, dan target dana darurat.
- **Grafik histori 24 bulan + forecast 3 bulan** lengkap dengan interval prediksi dan garis pendapatan.
- **Financial Health Score (0-100)** dengan breakdown 6 komponen:

  | Komponen | Bobot |
  |---|---|
  | Proyeksi savings rate | 30% |
  | Debt-to-income | 20% |
  | Credit score | 15% |
  | Volatilitas pengeluaran | 15% |
  | Porsi pengeluaran diskresioner | 10% |
  | Transaksi ditolak (stres likuiditas) | 10% |

  Skor dikelompokkan ke tiga band: 🟢 **Sehat**, 🟡 **Waspada**, 🔴 **Rentan**.
- **Budget planner interaktif (what-if)**: ubah pendapatan, batas pemangkasan per kategori diskresioner, dan porsi surplus yang ditabung. Kategori yang di atas median peer (kuintil pendapatan sama) otomatis diberi saran hemat, plus ambang alert P90 untuk notifikasi.
- **🤖 AI Financial Coach**: membuat pesan coaching personal (judul, ringkasan, insight, rekomendasi, CTA) dari payload tanpa PII. Semua angka di output LLM divalidasi terhadap payload, dan otomatis jatuh ke template deterministik jika API tidak tersedia.

### 🏦 Tab Portfolio
- KPI portfolio: jumlah nasabah aktif, proporsi Sehat/Rentan, total forecast outflow, total potensi auto-save.
- Distribusi health band dan forecast pengeluaran vs pendapatan seluruh nasabah.
- Ringkasan rekomendasi aksi per segmen.
- **Early warning list** untuk Relationship Manager: nasabah band Rentan yang tetap defisit walau sudah ikut budget, bisa diunduh sebagai CSV.

### 📈 Tab Kinerja Model
- Hasil backtest rolling-origin (Nov 2018 - Okt 2019) untuk tiga level: total per nasabah, nasabah x kategori, dan outflow harian portfolio.
- Coverage interval 80%, AUC Health Score, dan faktor kalibrasi interval.
- Visual perbandingan model, contoh forecast, dan SHAP feature importance.

---

## 📊 Ringkasan Kinerja Model

| Level | Model | Hasil |
|---|---|---|
| Total pengeluaran per nasabah | LightGBM global (Tweedie + quantile) | WAPE **15,1%** (vs seasonal naive 21,0%) |
| Nasabah x kategori (budget) | LightGBM global | WAPE **39,7%** (vs seasonal naive 53,4%) |
| Outflow harian portfolio (30 hari) | Holt-Winters ETS | WAPE **2,6%** |
| Interval prediksi 80% | Quantile + kalibrasi conformal (x1,09) | Coverage **79,9%** out-of-sample |
| Financial Health Score | Rule-based, 6 komponen | AUC **0,89** untuk stres keuangan 12 bulan |

---

## 🚀 Cara Menjalankan

```bash
# 1. Clone repo
git clone https://github.com/ndifikri/financial-planning-and-forecasting.git
cd financial-planning-and-forecasting

# 2. (Opsional) buat virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependency
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

### 🔑 Konfigurasi LLM (opsional)

AI Financial Coach memakai OpenAI API. Isi API key dengan salah satu cara berikut:

- Environment variable: `OPENAI_API_KEY=sk-...`
- Langsung lewat kolom **OpenAI API key** di sidebar aplikasi

Tanpa API key, aplikasi tetap berjalan normal dan pesan coaching dibuat dari template cadangan.

---

## 📁 Dependensi Proyek

`app.py` adalah lapisan UI. Logika model dan data dibaca dari modul serta file hasil notebook berikut, yang perlu ada di folder yang sama saat aplikasi dijalankan:

```
.
├── app.py                 # aplikasi Streamlit (repo ini)
├── requirements.txt       # dependency Python (repo ini)
├── src/                   # config, pipeline scoring, planner, LLM advisor, mapping kategori
├── data/                  # panel_monthly, behaviour_monthly, client_static (.parquet)
├── artifacts/             # models.joblib, metrics.json, hasil backtest
└── figures/               # grafik kinerja model
```

Folder `data/` dan `artifacts/` dihasilkan oleh notebook end-to-end dari dataset mentah, sehingga aplikasi tidak perlu memproses ulang data transaksi 1,2 GB.

---

## 🗂️ Data

Dataset: **Financial Transactions Dataset: Analytics** (Kaggle, Caixabank Tech). Data sintetis berisi sekitar 13,3 juta transaksi kartu periode 2010 - Oktober 2019 dalam USD.

Asumsi: pendapatan bulanan = `yearly_income / 12`, karena dataset tidak memuat transaksi gaji.

---

## ⚠️ Batasan

- Data bersifat sintetis dan relatif stasioner, sehingga keunggulan ML atas moving average jangka panjang masih kecil. Pada data riil, fitur kalender lokal (tanggal gajian, Ramadan/Lebaran, Harbolnas, THR) diharapkan memperlebar gap.
- Kategori yang jarang/lumpy (mis. Travel) sulit diprediksi secara point forecast. Untuk kategori ini, ambang alert P90 lebih relevan.
- Financial Health Score masih rule-based dan perlu dikalibrasi bersama tim Risk. Skor ini **tidak** dimaksudkan sebagai satu-satunya dasar keputusan kredit.

---

## 🛠️ Tech Stack

**Python** · **Streamlit** · **Altair** · **pandas** · **NumPy** · **DuckDB** · **PyArrow** · **LightGBM** · **scikit-learn** · **statsmodels** · **SHAP** · **OpenAI API**
