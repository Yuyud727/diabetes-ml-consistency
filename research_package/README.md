# Cross-Dataset Consistency — Prediksi Diabetes

Implementasi lengkap dan dapat direplikasi untuk penelitian:

> **Analisis Cross-Dataset Consistency pada Model Machine Learning untuk Prediksi
> Diabetes Menggunakan Evaluasi Performa, Ranking, dan Explainability**

Kerangka ini menguji apakah sebuah model machine learning **konsisten** ketika
diuji pada tiga dataset diabetes yang berbeda secara demografis, di bawah protokol
eksperimen yang identik (*controlled comparison*). Konsistensi dinilai pada tiga
dimensi sekaligus: **performa**, **ranking**, dan **eksplanasi (SHAP)**.

---

## 1. Ringkasan Eksperimen

| Komponen | Spesifikasi |
|---|---|
| Dataset | PIDD (768×8), Diabetes Prediction (~100k×8), Early Stage (520×17) |
| Model | Random Forest, XGBoost, KNN, **Soft Voting Ensemble** |
| Preprocessing | Mean imputation → encoding → StandardScaler → SMOTE (hanya data latih) |
| Validasi | Stratified 10-Fold CV, `random_state=42` |
| Metrik | **F1 (utama)**, AUC-ROC, Recall, Accuracy, Precision |
| Explainability | TreeSHAP (RF/XGB), KernelSHAP (KNN & Ensemble) |
| Uji statistik | Wilcoxon, Friedman, Nemenyi, Spearman |

**Kriteria konsistensi (aturan konjungtif):**

1. Performa — `|ΔF1| ≤ 0.05` untuk semua pasang dataset
2. Ranking — `Spearman ρ ≥ 0.8` untuk semua pasang dataset
3. Eksplanasi — overlap top-3 SHAP `≥ 2 dari 3` pasang dataset

> Model dinyatakan **konsisten** bila memenuhi **ketiga** dimensi sekaligus.

---

## 2. Instalasi

```bash
# (disarankan) buat virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Python **3.11+** direkomendasikan.

---

## 3. Menyiapkan Dataset

Tempatkan tiga berkas CSV berikut:

```
datasets/pidd/pidd.csv
datasets/diabetes_prediction/diabetes_prediction.csv
datasets/early_stage/early_stage.csv
```

Pembantu pengunduhan otomatis:

```bash
python download_data.py
```

- **PIDD** diunduh otomatis dari mirror publik.
- **Early Stage** dicoba via paket `ucimlrepo` (UCI ID 529).
- **Diabetes Prediction** memerlukan akun Kaggle — ikuti instruksi yang dicetak
  (`kaggle datasets download -d iammustafatz/diabetes-prediction-dataset`).

Detail kolom target dan penanganan khusus tiap dataset ada di
[`datasets/README.md`](datasets/README.md) dan terpusat di `config.py`.

> Eksperimen **tetap berjalan** meski hanya sebagian dataset tersedia; dataset
> yang tidak ditemukan akan dilewati dengan peringatan.

---

## 4. Menjalankan  NOTED

```bash
python run_experiment.py
```

Opsi yang berguna:

```bash
python run_experiment.py --quick            # smoke test cepat (fold sedikit, SHAP ringkas)
python run_experiment.py --datasets pidd    # hanya satu dataset
python run_experiment.py --no-shap          # lewati SHAP (lebih cepat)
python run_experiment.py --no-figures       # lewati pembuatan figur
```

Seluruh hasil ditulis ke `outputs/` dan laporan otomatis ke
`docs/experiment_report.md`.

---

## 5. Struktur Proyek

```
research_package/
├── README.md
├── requirements.txt
├── run_experiment.py          # titik masuk utama
├── generate_report.py         # otomatisasi BAB 4 (Tahap 9)
├── download_data.py           # pembantu unduh dataset
├── config.py                  # SELURUH parameter terpusat (anti-hardcode)
│
├── datasets/                  # letakkan CSV di sini (lihat README di dalamnya)
│
├── src/
│   ├── preprocessing.py       # pipeline anti-leakage (impute/encode/scale/SMOTE)
│   ├── models.py              # factory RF, XGB, KNN, Soft Voting
│   ├── evaluation.py          # 10-fold CV + prediksi out-of-fold
│   ├── shap_analysis.py       # TreeSHAP / KernelSHAP, top-3 & peringkat
│   ├── stats_tests.py         # Wilcoxon, Friedman, Nemenyi, Spearman
│   ├── consistency.py         # evaluasi 3 dimensi + sintesis konjungtif
│   ├── visualization.py       # confusion matrix, ROC, SHAP, ranking, fitur
│   └── experiment.py          # orkestrator pipeline
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── outputs/
│   ├── tables/                # performance, delta_f1, ranking, wilcoxon, ...
│   ├── figures/               # confusion_matrix, roc_curve, shap_*, ...
│   └── logs/
│
└── docs/
    ├── methodology.md
    └── experiment_report.md   # dihasilkan otomatis
```

---

## 6. Tabel Keluaran (`outputs/tables/`)

| Berkas | Isi |
|---|---|
| `performance.csv` | F1/AUC/Recall/Accuracy/Precision (mean ± std) tiap sel |
| `delta_f1.csv` | `|ΔF1|` antar pasang dataset + status konsistensi performa |
| `wilcoxon.csv` | p-value Wilcoxon signed-rank antar pasang dataset |
| `ranking.csv` | peringkat model per dataset |
| `spearman.csv` | korelasi Spearman peringkat antar dataset |
| `friedman.csv` | χ², p-value Friedman |
| `nemenyi.csv` | matriks p-value Nemenyi post-hoc |
| `shap_top3.csv` | top-3 fitur SHAP tiap sel |
| `shap_overlap.csv` | overlap top-3 antar dataset + status dimensi eksplanasi |
| `shap_spearman.csv` | korelasi Spearman peringkat SHAP antar dataset |
| `consistency.csv` | **rekapitulasi konsistensi tiga dimensi (final)** |

---

## 7. Jaminan Anti-Kebocoran (Data Leakage)

- Pemisahan data (CV split) dilakukan **sebelum** preprocessing apa pun.
- Seluruh estimasi parameter (imputasi, encoding, scaling) di-`fit` **hanya** pada
  fold latih melalui satu objek `Pipeline`.
- **SMOTE diterapkan hanya pada data latih** di setiap fold (langkah di dalam
  `imblearn.Pipeline`), tidak pernah menyentuh data uji.
- Background SHAP diambil dari data latih.

Lihat `docs/methodology.md` untuk penjelasan metodologis ringkas yang selaras
dengan BAB 3.

---

## 8. Reprodusibilitas

`random_state=42` ditetapkan pada pembagian fold, inisialisasi model, dan SMOTE.
Hyperparameter dan urutan pipeline didokumentasikan di `config.py`, sehingga
eksperimen dapat diulang persis dan diverifikasi secara independen.

## 9. Catatan Performa

KernelSHAP (untuk KNN & Soft Voting) bersifat mahal secara komputasi. Untuk
menjaga waktu jalan tetap wajar, ukuran *background* dan jumlah sampel yang
dijelaskan dibatasi melalui `config.py` (`SHAP_*`), dan dataset sangat besar
disubsampling **khusus untuk SHAP** (tidak memengaruhi estimasi performa).
Gunakan `--quick` untuk verifikasi cepat end-to-end.
