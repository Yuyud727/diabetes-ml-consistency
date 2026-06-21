"""
src/consistency.py
==================
Evaluasi konsistensi tiga dimensi dan sintesis akhir (aturan konjungtif).

Dimensi (sesuai BAB 3, Subbab 3.5):
    1. Performa   : |ΔF1| <= 0.05 untuk SEMUA pasang dataset
    2. Ranking    : Spearman rho >= 0.80 untuk SEMUA pasang dataset
    3. Eksplanasi : top-3 SHAP overlap >= 2 dari 3 pasang dataset

Keputusan akhir (konjungtif):
    Konsisten = Performa AND Ranking AND Eksplanasi
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config
from stats_tests import _safe_spearman

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dimensi 1: Performa (ΔF1) -> Tabel 4.2
# --------------------------------------------------------------------------- #
def delta_f1_table(mean_f1: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung |ΔF1| antar pasang dataset per model.

    mean_f1 : DataFrame rata-rata F1 (indeks=model label, kolom=dataset short)
    """
    short = {k: config.DATASETS[k]["short"] for k in config.DATASET_ORDER}
    rows = []
    for mname in config.MODEL_ORDER:
        label = config.MODEL_LABELS[mname]
        if label not in mean_f1.index:
            continue
        rec = {"Model": label}
        all_ok = True
        for (a, b) in config.DATASET_PAIRS:
            sa, sb = short[a], short[b]
            col = f"{sa}-{sb}"
            if sa in mean_f1.columns and sb in mean_f1.columns:
                d = abs(float(mean_f1.loc[label, sa]) - float(mean_f1.loc[label, sb]))
                rec[col] = round(d, 4)
                if d > config.DELTA_F1_THRESHOLD:
                    all_ok = False
            else:
                rec[col] = None
                all_ok = False
        rec["Konsisten (|dF1|<=0.05)"] = "Ya" if all_ok else "Tidak"
        rec["_perf_pass"] = all_ok
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Dimensi 2: Ranking (Spearman per model antar pasang)
# --------------------------------------------------------------------------- #
def ranking_consistency_per_model(mean_f1: pd.DataFrame) -> dict[str, bool]:
    """
    Catatan: Dimensi ranking pada kerangka ini menilai STABILITAS URUTAN MODEL
    antar dataset (korelasi peringkat keseluruhan model), bukan peringkat satu
    model. Konsistensi ranking karena itu bersifat global (berlaku sama untuk
    seluruh model pada satu pasang dataset). Fungsi ini mengembalikan apakah
    SEMUA pasang dataset memenuhi rho >= ambang; nilai diterapkan ke tiap model
    pada sintesis akhir.
    """
    short = {k: config.DATASETS[k]["short"] for k in config.DATASET_ORDER}
    ranks = mean_f1.rank(ascending=False, method="min")
    all_pass = True
    details = {}
    for (a, b) in config.DATASET_PAIRS:
        sa, sb = short[a], short[b]
        if sa not in ranks.columns or sb not in ranks.columns:
            all_pass = False
            continue
        rho, _ = _safe_spearman(ranks[sa].values, ranks[sb].values)
        details[f"{sa}-{sb}"] = rho
        if rho is None or rho < config.SPEARMAN_THRESHOLD:
            all_pass = False
    logger.info("  Ranking Spearman antar pasang: %s -> %s",
                {k: (round(v, 3) if v is not None else None) for k, v in details.items()},
                "LOLOS" if all_pass else "TIDAK")
    return {m: all_pass for m in config.MODEL_ORDER}


# --------------------------------------------------------------------------- #
# Dimensi 3: Eksplanasi (overlap top-3 SHAP)
# --------------------------------------------------------------------------- #
def shap_overlap_table(all_shap) -> pd.DataFrame:
    """
    Menghitung overlap top-3 fitur SHAP antar pasang dataset per model dan
    menentukan kelolosan dimensi eksplanasi (overlap >= 2 dari 3 pasang).
    """
    short = {k: config.DATASETS[k]["short"] for k in config.DATASET_ORDER}
    rows = []
    for mname in config.MODEL_ORDER:
        label = config.MODEL_LABELS[mname]
        rec = {"Model": label}
        n_pairs_with_overlap = 0
        valid_pairs = 0
        for (a, b) in config.DATASET_PAIRS:
            sa, sb = short[a], short[b]
            ra = all_shap.get(a, {}).get(mname)
            rb = all_shap.get(b, {}).get(mname)
            col = f"{sa}-{sb}"
            if ra and rb and ra.ok and rb.ok:
                inter = set(ra.top_k) & set(rb.top_k)
                rec[col] = len(inter)
                valid_pairs += 1
                if len(inter) >= 1:
                    # "overlap pada pasang" = ada >=1 fitur top-3 yang sama
                    n_pairs_with_overlap += 1
            else:
                rec[col] = None
        rec["n_pairs_overlap"] = n_pairs_with_overlap
        passed = (n_pairs_with_overlap >= config.SHAP_OVERLAP_MIN)
        rec[f"Konsisten (overlap>={config.SHAP_OVERLAP_MIN}/3)"] = "Ya" if passed else "Tidak"
        rec["_expl_pass"] = passed
        rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Sintesis akhir (Tabel 4.7)
# --------------------------------------------------------------------------- #
def synthesize(mean_f1: pd.DataFrame, all_shap) -> pd.DataFrame:
    """
    Menggabungkan ketiga dimensi menjadi rekapitulasi konsistensi konjungtif.
    """
    perf = delta_f1_table(mean_f1).set_index("Model")
    rank_pass = ranking_consistency_per_model(mean_f1)  # {model_name: bool}
    expl = shap_overlap_table(all_shap).set_index("Model")

    rows = []
    for mname in config.MODEL_ORDER:
        label = config.MODEL_LABELS[mname]
        p = bool(perf.loc[label, "_perf_pass"]) if label in perf.index else False
        r = bool(rank_pass.get(mname, False))
        e = bool(expl.loc[label, "_expl_pass"]) if label in expl.index else False
        consistent = p and r and e
        rows.append({
            "Model": label,
            "Performa": "Lolos" if p else "Gagal",
            "Ranking": "Lolos" if r else "Gagal",
            "Eksplanasi": "Lolos" if e else "Gagal",
            "Konsisten": "YA" if consistent else "TIDAK",
        })
    return pd.DataFrame(rows)
