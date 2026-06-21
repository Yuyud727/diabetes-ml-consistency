#!/usr/bin/env python3
"""
download_data.py
================
Pembantu pengunduhan dataset.

    - PIDD  : diunduh otomatis dari mirror publik (berheader).
    - Early : dicoba via paket `ucimlrepo` (UCI ML Repository, ID 529).
    - DPD   : dataset Kaggle (memerlukan autentikasi) -> ditampilkan instruksi.

Jalankan:
    python download_data.py

Catatan: jika unduhan otomatis gagal (mis. tanpa internet), ikuti petunjuk
manual yang dicetak ke layar dan pada datasets/README.md.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets"

# Mirror PIDD publik berheader (Pregnancies,Glucose,...,Outcome)
PIDD_URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
]
PIDD_HEADER = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
               "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


def _download(url: str, dest: Path) -> bool:
    try:
        print(f"  mengunduh: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        dest.write_bytes(data)
        print(f"  tersimpan: {dest} ({len(data)} bytes)")
        return True
    except Exception as exc:
        print(f"  gagal: {exc}")
        return False


def get_pidd() -> bool:
    dest = DATA / "pidd" / "pidd.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in PIDD_URLS:
        if _download(url, dest):
            _ensure_pidd_header(dest)
            return True
    print("  >> PIDD gagal diunduh otomatis. Unduh manual dari Kaggle: "
          "uciml/pima-indians-diabetes-database")
    return False


def _ensure_pidd_header(path: Path) -> None:
    """Memastikan PIDD memiliki header kolom yang benar."""
    try:
        import pandas as pd
        df = pd.read_csv(path)
        # Jika kolom pertama bukan nama yang dikenal, anggap tanpa header
        if df.columns[0] not in PIDD_HEADER and df.shape[1] == len(PIDD_HEADER):
            df = pd.read_csv(path, header=None, names=PIDD_HEADER)
            df.to_csv(path, index=False)
            print("  header PIDD dinormalisasi.")
    except Exception as exc:
        print(f"  peringatan: gagal memverifikasi header PIDD: {exc}")


def get_early() -> bool:
    dest = DATA / "early_stage" / "early_stage.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ucimlrepo import fetch_ucirepo
        print("  mengambil Early Stage (UCI ID 529) via ucimlrepo ...")
        ds = fetch_ucirepo(id=529)
        X = ds.data.features
        y = ds.data.targets
        df = X.copy()
        # nama kolom target pada UCI 529 adalah 'class'
        target_col = y.columns[0]
        df["class"] = y[target_col].values
        df.to_csv(dest, index=False)
        print(f"  tersimpan: {dest} ({len(df)} baris)")
        return True
    except Exception as exc:
        print(f"  gagal otomatis ({exc}).")
        print("  >> Unduh manual: UCI ML Repository ID 529 "
              "(Early Stage Diabetes Risk Prediction), simpan sebagai "
              f"{dest} dengan kolom target 'class'.")
        return False


def guide_dpd() -> None:
    dest = DATA / "diabetes_prediction" / "diabetes_prediction.csv"
    print("\nDiabetes Prediction Dataset (DPD) — perlu akun Kaggle:")
    print("  1) Pasang Kaggle CLI : pip install kaggle")
    print("  2) Letakkan kaggle.json di ~/.kaggle/ (token API Kaggle).")
    print("  3) Unduh & ekstrak    :")
    print("     kaggle datasets download -d iammustafatz/diabetes-prediction-dataset")
    print(f"     unzip diabetes-prediction-dataset.zip -d {dest.parent}")
    print(f"  4) Pastikan file akhir: {dest} (kolom target 'diabetes').")


def main() -> int:
    print("=" * 60)
    print("PENGUNDUHAN DATASET")
    print("=" * 60)
    print("\n[1] PIDD")
    ok_pidd = get_pidd()
    print("\n[2] Early Stage")
    ok_early = get_early()
    print("\n[3] Diabetes Prediction")
    guide_dpd()

    print("\n" + "=" * 60)
    print("RINGKASAN")
    print(f"  PIDD  : {'OK' if ok_pidd else 'MANUAL DIBUTUHKAN'}")
    print(f"  Early : {'OK' if ok_early else 'MANUAL DIBUTUHKAN'}")
    print("  DPD   : MANUAL (Kaggle)")
    print("=" * 60)
    print("\nSetelah dataset siap, jalankan: python run_experiment.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
