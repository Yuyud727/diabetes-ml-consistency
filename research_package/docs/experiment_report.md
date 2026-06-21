# BAB 4 — HASIL DAN PEMBAHASAN (Laporan Otomatis)

> Dokumen ini dihasilkan otomatis oleh `generate_report.py` dari berkas pada `outputs/`. Angka pada tabel terisi langsung dari hasil eksperimen; sel kosong menandakan hasil belum tersedia.

- Konfigurasi: `random_state=42`, `10-Fold Stratified CV`, mode quick = `False`.


## 4.1 Hasil Eksperimen

Eksperimen mencakup kombinasi 4 model × dataset aktif, masing-masing dievaluasi dengan 10-fold cross-validation.


**Tabel 4.1 Hasil Performa Seluruh Model pada Seluruh Dataset**

| Dataset   | Model   |   F1_mean |   F1_std |   AUC_mean |   AUC_std |   Recall_mean |   Recall_std |   Accuracy_mean |   Accuracy_std |   Precision_mean |   Precision_std |
|:----------|:--------|----------:|---------:|-----------:|----------:|--------------:|-------------:|----------------:|---------------:|-----------------:|----------------:|
| PIDD      | RF      |    0.6552 |   0.0624 |     0.8235 |    0.0422 |        0.6788 |       0.0767 |          0.7512 |         0.0439 |           0.6372 |          0.0693 |
| PIDD      | XGB     |    0.6412 |   0.083  |     0.8026 |    0.0449 |        0.6632 |       0.1102 |          0.7446 |         0.0484 |           0.6252 |          0.069  |
| PIDD      | KNN     |    0.6613 |   0.057  |     0.8097 |    0.0574 |        0.7907 |       0.0848 |          0.7174 |         0.0488 |           0.5714 |          0.0597 |
| PIDD      | SVE     |    0.6762 |   0.0707 |     0.8234 |    0.0474 |        0.7271 |       0.0952 |          0.759  |         0.0452 |           0.6353 |          0.0642 |
| DPD       | RF      |    0.7522 |   0.0127 |     0.9665 |    0.003  |        0.7496 |       0.0136 |          0.958  |         0.0022 |           0.7548 |          0.015  |
| DPD       | XGB     |    0.8004 |   0.0118 |     0.9775 |    0.0021 |        0.7078 |       0.0149 |          0.97   |         0.0017 |           0.9213 |          0.0131 |
| DPD       | KNN     |    0.6053 |   0.0095 |     0.938  |    0.0052 |        0.8352 |       0.0088 |          0.9074 |         0.0038 |           0.4748 |          0.0123 |
| DPD       | SVE     |    0.7464 |   0.0116 |     0.9722 |    0.0022 |        0.7765 |       0.0142 |          0.9551 |         0.0021 |           0.7187 |          0.0139 |
| Early     | RF      |    0.9875 |   0.0136 |     0.9992 |    0.0013 |        0.9875 |       0.0207 |          0.9846 |         0.0168 |           0.988  |          0.0198 |
| Early     | XGB     |    0.9811 |   0.0094 |     0.9969 |    0.0043 |        0.975  |       0.0187 |          0.9769 |         0.0115 |           0.988  |          0.0198 |
| Early     | KNN     |    0.9492 |   0.0496 |     0.9906 |    0.0123 |        0.9125 |       0.0813 |          0.9423 |         0.0516 |           0.9935 |          0.0194 |
| Early     | SVE     |    0.9762 |   0.0149 |     0.9989 |    0.002  |        0.9656 |       0.0295 |          0.9712 |         0.0177 |           0.9879 |          0.0199 |


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_KNN_PIDD.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_RandomForest_PIDD.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_SoftVoting_PIDD.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_XGBoost_PIDD.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_KNN_DPD.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_RandomForest_DPD.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_SoftVoting_DPD.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_XGBoost_DPD.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_KNN_Early.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_RandomForest_Early.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_SoftVoting_Early.png`


*Gambar 4.1 Confusion Matrix* — `outputs\figures\confusion_matrix\cm_XGBoost_Early.png`


*Gambar 4.2 Kurva ROC @ PIDD* — `outputs\figures\roc_curve\roc_PIDD.png`


*Gambar 4.2 Kurva ROC @ DPD* — `outputs\figures\roc_curve\roc_DPD.png`


*Gambar 4.2 Kurva ROC @ Early* — `outputs\figures\roc_curve\roc_Early.png`


## 4.2 Analisis Konsistensi Performa

**Tabel 4.2 Nilai ΔF1 Antar Dataset**

| Model   |   PIDD-DPD |   PIDD-Early |   DPD-Early | Konsisten (|dF1|<=0.05)   |
|:--------|-----------:|-------------:|------------:|:--------------------------|
| RF      |     0.097  |       0.3323 |      0.2353 | Tidak                     |
| XGB     |     0.1592 |       0.3399 |      0.1807 | Tidak                     |
| KNN     |     0.056  |       0.2879 |      0.3439 | Tidak                     |
| SVE     |     0.0702 |       0.3    |      0.2298 | Tidak                     |


## 4.3 Hasil Uji Wilcoxon

**Tabel 4.3 Hasil Wilcoxon Signed-Rank Test**

| Model   | Dataset Pair   |   statistic |   p_value | Signifikan (p<0.05)   |
|:--------|:---------------|------------:|----------:|:----------------------|
| RF      | PIDD-DPD       |           0 |     0.002 | Ya                    |
| RF      | PIDD-Early     |           0 |     0.002 | Ya                    |
| RF      | DPD-Early      |           0 |     0.002 | Ya                    |
| XGB     | PIDD-DPD       |           0 |     0.002 | Ya                    |
| XGB     | PIDD-Early     |           0 |     0.002 | Ya                    |
| XGB     | DPD-Early      |           0 |     0.002 | Ya                    |
| KNN     | PIDD-DPD       |           0 |     0.002 | Ya                    |
| KNN     | PIDD-Early     |           0 |     0.002 | Ya                    |
| KNN     | DPD-Early      |           0 |     0.002 | Ya                    |
| SVE     | PIDD-DPD       |           0 |     0.002 | Ya                    |
| SVE     | PIDD-Early     |           0 |     0.002 | Ya                    |
| SVE     | DPD-Early      |           0 |     0.002 | Ya                    |


## 4.4 Analisis Ranking Model

**Tabel 4.4 Ranking Model Berdasarkan F1**

| Model   |   PIDD |   DPD |   Early |
|:--------|-------:|------:|--------:|
| RF      |      3 |     2 |       1 |
| XGB     |      4 |     1 |       2 |
| KNN     |      2 |     4 |       4 |
| SVE     |      1 |     3 |       3 |


**Korelasi Spearman peringkat antar dataset**

| Pair       |   spearman_rho |   p_value | Konsisten (rho>=0.8)   |
|:-----------|---------------:|----------:|:-----------------------|
| PIDD-DPD   |           -0.8 |       0.2 | Tidak                  |
| PIDD-Early |           -0.6 |       0.4 | Tidak                  |
| DPD-Early  |            0.8 |       0.2 | Tidak                  |


*Gambar 4.3 Perbandingan Ranking Model* — `outputs\figures\ranking_plot\ranking_comparison.png`


## 4.5 Hasil Friedman Test dan Nemenyi

**Tabel 4.5 Hasil Friedman Test**

| Statistik   |   Nilai |
|:------------|--------:|
| chi_square  |  1.8    |
| p_value     |  0.6149 |
| df          |  3      |
| k_models    |  4      |
| n_datasets  |  3      |


**Nemenyi post-hoc (matriks p-value)**

|     |     RF |    XGB |    KNN |    SVE |
|:----|-------:|-------:|-------:|-------:|
| RF  | 1      | 0.9891 | 0.5854 | 0.9891 |
| XGB | 0.9891 | 1      | 0.7785 | 1      |
| KNN | 0.5854 | 0.7785 | 1      | 0.7785 |
| SVE | 0.9891 | 1      | 0.7785 | 1      |


## 4.6 Analisis Explainability SHAP


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_KNN_DPD.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_KNN_Early.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_KNN_PIDD.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_RandomForest_DPD.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_RandomForest_Early.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_RandomForest_PIDD.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_SoftVoting_DPD.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_SoftVoting_Early.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_SoftVoting_PIDD.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_XGBoost_DPD.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_XGBoost_Early.png`


*Gambar 4.4 SHAP Summary Plot* — `outputs\figures\shap_summary\shap_summary_XGBoost_PIDD.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_KNN_DPD.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_KNN_Early.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_KNN_PIDD.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_RandomForest_DPD.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_RandomForest_Early.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_RandomForest_PIDD.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_SoftVoting_DPD.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_SoftVoting_Early.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_SoftVoting_PIDD.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_XGBoost_DPD.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_XGBoost_Early.png`


*Gambar 4.5 SHAP Force Plot* — `outputs\figures\shap_force\shap_force_XGBoost_PIDD.png`


## 4.7 Analisis Stabilitas Fitur

**Tabel 4.6 Top-3 SHAP Feature**

| Dataset   | Model   | Top-1          | Top-2               | Top-3               |
|:----------|:--------|:---------------|:--------------------|:--------------------|
| PIDD      | RF      | Glucose        | BMI                 | Age                 |
| PIDD      | XGB     | Glucose        | BMI                 | Age                 |
| PIDD      | KNN     | Glucose        | Age                 | BMI                 |
| PIDD      | SVE     | Glucose        | BMI                 | Age                 |
| DPD       | RF      | HbA1c_level    | blood_glucose_level | age                 |
| DPD       | XGB     | HbA1c_level    | blood_glucose_level | age                 |
| DPD       | KNN     | HbA1c_level    | age                 | blood_glucose_level |
| DPD       | SVE     | HbA1c_level    | blood_glucose_level | age                 |
| Early     | RF      | Polyuria_Yes   | Polydipsia_Yes      | Gender_Male         |
| Early     | XGB     | Polydipsia_Yes | Polyuria_Yes        | Gender_Male         |
| Early     | KNN     | Polydipsia_Yes | Polyuria_Yes        | Gender_Male         |
| Early     | SVE     | Polyuria_Yes   | Polydipsia_Yes      | Gender_Male         |


**Overlap top-3 SHAP antar dataset**

| Model   |   PIDD-DPD |   PIDD-Early |   DPD-Early |   n_pairs_overlap | Konsisten (overlap>=2/3)   |
|:--------|-----------:|-------------:|------------:|------------------:|:---------------------------|
| RF      |          0 |            0 |           0 |                 0 | Tidak                      |
| XGB     |          0 |            0 |           0 |                 0 | Tidak                      |
| KNN     |          0 |            0 |           0 |                 0 | Tidak                      |
| SVE     |          0 |            0 |           0 |                 0 | Tidak                      |


*Gambar 4.6 Perbandingan Feature Importance SHAP* — `outputs\figures\feature_comparison\feature_comparison_KNN.png`


*Gambar 4.6 Perbandingan Feature Importance SHAP* — `outputs\figures\feature_comparison\feature_comparison_RandomForest.png`


*Gambar 4.6 Perbandingan Feature Importance SHAP* — `outputs\figures\feature_comparison\feature_comparison_SoftVoting.png`


*Gambar 4.6 Perbandingan Feature Importance SHAP* — `outputs\figures\feature_comparison\feature_comparison_XGBoost.png`


## 4.8 Sintesis Cross-Dataset Consistency

**Tabel 4.7 Rekapitulasi Konsistensi Tiga Dimensi**

| Model   | Performa   | Ranking   | Eksplanasi   | Konsisten   |
|:--------|:-----------|:----------|:-------------|:------------|
| RF      | Gagal      | Gagal     | Gagal        | TIDAK       |
| XGB     | Gagal      | Gagal     | Gagal        | TIDAK       |
| KNN     | Gagal      | Gagal     | Gagal        | TIDAK       |
| SVE     | Gagal      | Gagal     | Gagal        | TIDAK       |


Pada konfigurasi ini, belum ada model yang memenuhi ketiga dimensi sekaligus.


## 4.9 Diskusi Penelitian

_(Bagian diskusi naratif diisi pada dokumen BAB 4 berformat Word; laporan ini menyediakan basis numerik dan figur pendukungnya.)_
