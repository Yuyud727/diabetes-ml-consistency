# Dataset

Letakkan tiga berkas CSV pada struktur berikut:

```
datasets/
├── pidd/
│   └── pidd.csv
├── diabetes_prediction/
│   └── diabetes_prediction.csv
└── early_stage/
    └── early_stage.csv
```

Spesifikasi setiap dataset (dipakai oleh `config.py`):

| Dataset | Folder/berkas | Kolom target | Label positif | Catatan |
|---|---|---|---|---|
| Pima Indians Diabetes (PIDD) | `pidd/pidd.csv` | `Outcome` | `1` | 0 pada Glucose, BloodPressure, SkinThickness, Insulin, BMI → diperlakukan NaN |
| Diabetes Prediction (DPD) | `diabetes_prediction/diabetes_prediction.csv` | `diabetes` | `1` | kolom kategorikal `gender`, `smoking_history` di-one-hot otomatis |
| Early Stage Risk (Early) | `early_stage/early_stage.csv` | `class` | `Positive` | mayoritas fitur biner Yes/No → di-encode {0,1} |

## Cara memperoleh

### Otomatis
```bash
python download_data.py
```

### Manual
- **PIDD** — Kaggle: `uciml/pima-indians-diabetes-database`
  (atau mirror berheader mana pun dengan 8 fitur + kolom `Outcome`).
- **Diabetes Prediction** — Kaggle: `iammustafatz/diabetes-prediction-dataset`.
  ```bash
  pip install kaggle
  kaggle datasets download -d iammustafatz/diabetes-prediction-dataset
  unzip diabetes-prediction-dataset.zip -d datasets/diabetes_prediction/
  ```
- **Early Stage** — UCI ML Repository, ID **529**
  (*Early Stage Diabetes Risk Prediction*). Simpan sebagai
  `early_stage/early_stage.csv` dengan kolom target `class`.

## Verifikasi

Pastikan jumlah fitur dan label sesuai sebelum menjalankan eksperimen:

```python
import pandas as pd
for p in ["pidd/pidd.csv",
          "diabetes_prediction/diabetes_prediction.csv",
          "early_stage/early_stage.csv"]:
    df = pd.read_csv(f"datasets/{p}")
    print(p, df.shape, "->", df.columns.tolist())
```

> Jika nama kolom pada berkas Anda berbeda (mis. ejaan target), sesuaikan field
> `target` / `positive_label` pada `config.py`. Tidak ada parameter dataset yang
> perlu diubah di tempat lain.
