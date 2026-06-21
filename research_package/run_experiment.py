#!/usr/bin/env python3
"""
run_experiment.py
=================
Titik masuk utama. Menjalankan seluruh pipeline eksperimen cross-dataset
consistency dengan satu perintah:

    python run_experiment.py

Opsi:
    --datasets pidd dpd early   pilih subset dataset (default: semua yang tersedia)
    --no-shap                   lewati analisis SHAP (lebih cepat)
    --no-figures                lewati pembuatan figur
    --quick                     mode cepat (smoke test): fold sedikit + SHAP ringkas

Mode quick juga dapat diaktifkan via environment variable: RP_QUICK=1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Eksperimen cross-dataset consistency prediksi diabetes."
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=["pidd", "dpd", "early"],
                        help="Subset dataset yang dijalankan.")
    parser.add_argument("--no-shap", action="store_true",
                        help="Lewati analisis SHAP.")
    parser.add_argument("--no-figures", action="store_true",
                        help="Lewati pembuatan figur.")
    parser.add_argument("--quick", action="store_true",
                        help="Mode cepat untuk verifikasi pipeline (smoke test).")
    parser.add_argument("--resume", action="store_true",
                        help="Lanjutkan dari checkpoint: skip dataset yang sudah selesai.")
    
    args = parser.parse_args()

    # Mode quick harus diset SEBELUM import config
    if args.quick:
        os.environ["RP_QUICK"] = "1"

    # Tambahkan src ke path
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))

    import experiment  # noqa: E402  (import setelah set env)

    result = experiment.run_all(
        dataset_keys=args.datasets,
        do_shap=not args.no_shap,
        do_figures=not args.no_figures,
    )

    if isinstance(result, dict) and result.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
