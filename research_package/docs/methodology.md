# Metodologi Penelitian

Dokumen ini merangkum metodologi yang diimplementasikan pada *research package* ini.
Ringkasan ini selaras dengan BAB 3 (Metodologi Penelitian) dan menjadi acuan teknis
bagi reproduksibilitas eksperimen.

## 1. Tujuan dan Desain Eksperimen

Penelitian ini menganalisis **cross-dataset consistency** model *machine learning* untuk
prediksi diabetes pada tiga dimensi: **performa**, **ranking**, dan **explainability**.
Desain eksperimen bersifat faktorial **4 model × 3 dataset = 12 sel**, di mana setiap sel
dievaluasi menggunakan **Stratified 10-Fold Cross Validation**, sehingga total terdapat
**120 proses pelatihan dan evaluasi**.

## 2. Dataset

| Kode  | Dataset                                   | Dimensi (perkiraan) | Target    | Kelas Positif |
|-------|-------------------------------------------|---------------------|-----------|---------------|
| PIDD  | Pima Indians Diabetes Dataset             | 768 × 8             | `Outcome` | `1`           |
| DPD   | Diabetes Prediction Dataset               | ~100.000 × 8        | `diabetes`| `1`           |
| Early | Early Stage Diabetes Risk Prediction      | 520 × 16            | `class`   | `Positive`    |

Spesifikasi kolom, tipe fitur, dan target didefinisikan secara terpusat pada `config.py`
(kamus `DATASETS`).

## 3. Preprocessing (Anti Data Leakage)

Seluruh langkah preprocessing **di-*fit* hanya pada lipatan latih (train fold)** di dalam
*cross validation*, kemudian *transform* diterapkan pada lipatan uji. Hal ini dijamin
secara struktural dengan membungkus seluruh tahap dalam satu `Pipeline`
(`imblearn.pipeline.Pipeline`) sehingga tidak ada informasi dari data uji yang bocor ke
proses pelatihan.

Urutan langkah:

1. **Konversi 0 → NaN (khusus PIDD)** pada kolom `Glucose`, `BloodPressure`,
   `SkinThickness`, `Insulin`, `BMI`, karena nilai nol pada kolom tersebut tidak valid
   secara medis dan sesungguhnya merepresentasikan *missing value*.
2. **Mean Imputation** untuk fitur numerik (`SimpleImputer(strategy="mean")`) dan
   *most-frequent imputation* untuk fitur kategorikal.
3. **Encoding**: *One-Hot Encoding* untuk fitur nominal (mis. `gender`,
   `smoking_history`) dan *binary/label encoding* untuk fitur biner Yes/No.
4. **StandardScaler** pada fitur numerik.
5. **SMOTE** untuk penyeimbangan kelas — **hanya pada data latih**, diterapkan setelah
   scaling di dalam pipeline.

## 4. Model

Empat model dievaluasi (lihat `src/models.py`):

1. **Random Forest** — `n_estimators=300`, `random_state=42`.
2. **XGBoost** — `n_estimators=300`, `max_depth=6`, `learning_rate=0.1`,
   `tree_method="hist"`, `eval_metric="logloss"`.
3. **K-Nearest Neighbor (KNN)** — `n_neighbors=11`, `weights="distance"`.
4. **Soft Voting Ensemble** — rata-rata *soft voting* dari ketiga model di atas
   (`VotingClassifier(voting="soft")`). Model ini merupakan model utama penelitian.

## 5. Validasi

Validasi menggunakan `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` untuk
menjaga proporsi kelas pada tiap lipatan dan menjamin reproduksibilitas.

## 6. Metrik Evaluasi

- **F1-Score** (metrik utama, fokus kelas positif)
- **AUC-ROC**
- **Recall** (pendukung utama — penting untuk meminimalkan *false negative* diabetes)
- **Accuracy** (pendukung)
- **Precision** (pendukung)

Hasil disimpan ke `outputs/tables/performance.csv`.

## 7. Explainability (SHAP)

- **TreeSHAP** (`shap.TreeExplainer`) untuk Random Forest dan XGBoost.
- **KernelSHAP** (`shap.KernelExplainer` atas `predict_proba`) untuk KNN dan Soft Voting,
  dengan *background* hasil ringkasan `shap.kmeans` dari data latih.
- Dihitung **Mean Absolute SHAP**, **Top-3 Feature**, dan **Spearman SHAP Ranking**
  antar dataset.

Untuk menjaga efisiensi komputasi pada dataset besar (mis. DPD), jumlah sampel yang
dijelaskan dibatasi (lihat `SHAP_LARGE_DATASET_CAP` pada `config.py`).

## 8. Uji Statistik

- **Wilcoxon Signed-Rank Test** — membandingkan distribusi F1 antar pasangan dataset.
- **Friedman Test** — menguji perbedaan peringkat model secara global.
- **Nemenyi Post-Hoc** — uji lanjutan berpasangan bila Friedman signifikan
  (memerlukan paket `scikit-posthocs`).
- **Spearman Correlation** — mengukur kesamaan *ranking* model antar dataset.

## 9. Kriteria Konsistensi (Aturan Konjungtif)

Sebuah model dinyatakan **konsisten** apabila memenuhi **ketiga** dimensi sekaligus:

| Dimensi      | Kriteria                                          | Ambang                          |
|--------------|---------------------------------------------------|---------------------------------|
| Performa     | Selisih F1 absolut antar seluruh pasangan dataset | \|ΔF1\| ≤ 0,05                  |
| Ranking      | Korelasi Spearman *ranking* antar dataset         | ρ ≥ 0,80                        |
| Eksplanasi   | Irisan *Top-3* fitur SHAP antar dataset           | overlap ≥ 2 dari 3 pasangan     |

Formulasi:

```
Konsisten = Performa AND Ranking AND Eksplanasi
```

Ambang tersebut dikonfigurasi pada `config.py` (`DELTA_F1_THRESHOLD`,
`SPEARMAN_THRESHOLD`, `SHAP_TOP_K`, `SHAP_OVERLAP_MIN`, `ALPHA`).

## 10. Reproduksibilitas

- Seluruh komponen acak menggunakan `random_state = 42`.
- Konfigurasi terpusat pada `config.py`.
- Pipeline tunggal mencegah *data leakage*.
- Mode cepat tersedia melalui variabel lingkungan `RP_QUICK=1` (atau flag `--quick`)
  untuk uji asap (*smoke test*) dengan jumlah lipatan dan baris yang dikurangi.
