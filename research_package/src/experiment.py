"""
src/experiment.py
=================
Orkestrator utama eksperimen cross-dataset consistency.
[docstring sama seperti sebelumnya]
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import preprocessing
import evaluation
import shap_analysis
import stats_tests as stats_mod
import consistency as cons_mod
import visualization as viz

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Util
# --------------------------------------------------------------------------- #
def setup_logging() -> None:
    config.ensure_dirs()
    logfile = config.LOG_DIR / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(logfile, mode="a", encoding="utf-8"),  # "a" bukan "w"
        ],
    )


def discover_available_datasets() -> list[str]:
    """Mengembalikan key dataset yang file CSV-nya tersedia."""
    available = []
    for key in config.DATASET_ORDER:
        path = Path(config.DATASETS[key]["path"])
        if path.exists() and path.stat().st_size > 0:
            available.append(key)
        else:
            logger.warning("Dataset '%s' tidak ditemukan di %s (dilewati).",
                           key, path)
    return available


def _save_csv(df: pd.DataFrame, name: str) -> None:
    if df is None:
        return
    path = config.TABLE_DIR / name
    df.to_csv(path, index=False)
    logger.info("  tabel disimpan: %s (%d baris)", name, len(df))


# --------------------------------------------------------------------------- #
# RESUME: deteksi dataset yang sudah selesai dilatih
# --------------------------------------------------------------------------- #
def _load_completed_datasets() -> dict[str, dict]:
    """
    Membaca performance.csv yang sudah ada dan merekonstruksi all_results
    sebagai dict ringan (hanya mean & std, tanpa per_fold/OOF).

    Mengembalikan:
        {dataset_key: {model_name: CellResult-like dict}}
        atau {} jika performance.csv belum ada / tidak valid.
    """
    perf_path = config.TABLE_DIR / "performance.csv"
    if not perf_path.exists():
        return {}

    try:
        df = pd.read_csv(perf_path)
    except Exception as exc:
        logger.warning("Gagal membaca performance.csv: %s", exc)
        return {}

    # Bangun reverse lookup: short -> key
    short_to_key = {v["short"]: k for k, v in config.DATASETS.items()}
    # Bangun reverse lookup: label -> model_name
    label_to_model = {v: k for k, v in config.MODEL_LABELS.items()}

    completed: dict[str, dict] = {}
    for _, row in df.iterrows():
        dkey = short_to_key.get(row.get("Dataset", ""))
        mname = label_to_model.get(row.get("Model", ""))
        if not dkey or not mname:
            continue

        # Rekonstruksi CellResult minimal dari kolom mean/std
        cell = evaluation.CellResult(dataset=dkey, model=mname)
        for m in config.METRICS:
            label = config.METRIC_LABELS[m]
            mean_col = f"{label}_mean"
            std_col  = f"{label}_std"
            if mean_col in row:
                cell.mean[m] = float(row[mean_col])
                cell.std[m]  = float(row.get(std_col, 0.0))
                # per_fold diisi array kosong (tidak bisa direcovery dari CSV)
                cell.per_fold[m] = np.full(config.N_SPLITS, cell.mean[m])

        completed.setdefault(dkey, {})[mname] = cell

    return completed


def _check_dataset_complete(cached: dict[str, dict], dkey: str) -> bool:
    """
    Dataset dianggap 'sudah selesai' jika semua model ada di cache
    DAN setiap model punya nilai F1_mean yang valid (bukan NaN).
    """
    if dkey not in cached:
        return False
    for mname in config.MODEL_ORDER:
        cell = cached[dkey].get(mname)
        if cell is None:
            return False
        if np.isnan(cell.mean.get("f1", float("nan"))):
            return False
    return True


def _partition_datasets(available: list[str]) -> tuple[list[str], list[str]]:
    """
    Membagi dataset menjadi:
        done   : sudah ada di performance.csv dan lengkap (di-skip)
        todo   : belum ada / tidak lengkap (perlu dilatih)
    """
    cached = _load_completed_datasets()
    done, todo = [], []
    for key in available:
        if _check_dataset_complete(cached, key):
            short = config.DATASETS[key]["short"]
            logger.info("  [RESUME] %-6s sudah selesai -> di-skip (load dari cache)",
                        short)
            done.append(key)
        else:
            todo.append(key)
    return done, todo


def _merge_performance_csv(new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gabungkan hasil baru dengan baris lama di performance.csv.
    Baris lama untuk dataset yang baru dilatih AKAN diganti (upsert).
    """
    perf_path = config.TABLE_DIR / "performance.csv"
    if not perf_path.exists() or new_df is None:
        return new_df

    try:
        old_df = pd.read_csv(perf_path)
    except Exception:
        return new_df

    # Dataset yang baru dilatih: hapus dari old, ganti dengan new
    new_datasets = new_df["Dataset"].unique().tolist()
    old_df = old_df[~old_df["Dataset"].isin(new_datasets)]
    merged = pd.concat([old_df, new_df], ignore_index=True)

    # Urutkan sesuai DATASET_ORDER & MODEL_ORDER
    short_order = [config.DATASETS[k]["short"] for k in config.DATASET_ORDER]
    model_order = [config.MODEL_LABELS[m] for m in config.MODEL_ORDER]
    merged["_ds_rank"] = merged["Dataset"].map(
        {s: i for i, s in enumerate(short_order)})
    merged["_m_rank"] = merged["Model"].map(
        {m: i for i, m in enumerate(model_order)})
    merged = (merged.sort_values(["_ds_rank", "_m_rank"])
                    .drop(columns=["_ds_rank", "_m_rank"])
                    .reset_index(drop=True))
    return merged


# --------------------------------------------------------------------------- #
# OOF RESTORE: prediksi OOF untuk dataset dari cache
# --------------------------------------------------------------------------- #
def _restore_oof_predictions(cached_results: dict[str, dict],
                              datasets_cache: dict) -> dict[str, dict]:
    """
    Untuk dataset yang diload dari cache (performance.csv),
    CellResult tidak punya y_true/y_pred/y_proba karena CSV tidak menyimpannya.

    Fungsi ini menjalankan cross_val_predict (BUKAN cross_validate)
    secara ringan hanya untuk mendapatkan OOF predictions —
    skor metrik TIDAK ditimpa (tetap dari cache).

    Dipanggil hanya jika do_figures=True, karena hanya figur yang butuh OOF.
    """
    import evaluation  # sudah diimport di atas, tapi eksplisit untuk kejelasan
    for dkey, cells in cached_results.items():
        if dkey not in datasets_cache:
            logger.warning(
                "  [OOF] Dataset '%s' tidak bisa dimuat ulang untuk figur — "
                "confusion matrix & ROC dilewati.", dkey
            )
            continue
        X, y = datasets_cache[dkey]
        short = config.DATASETS[dkey]["short"]
        logger.info("  [OOF] Restore prediksi OOF untuk figur: %s", short)
        for mname, cell in cells.items():
            # Sudah punya OOF → skip
            if cell.y_true is not None:
                continue
            try:
                from sklearn.model_selection import StratifiedKFold
                from sklearn.model_selection import cross_val_predict
                from preprocessing import build_preprocessor, build_pipeline
                from models import make_model
                cv = evaluation.get_cv()
                preprocessor = build_preprocessor(X)
                pipeline = build_pipeline(
                    preprocessor, make_model(mname), use_smote=True
                )
                y_pred = cross_val_predict(
                    pipeline, X, y, cv=cv, method="predict", n_jobs=None
                )
                try:
                    proba = cross_val_predict(
                        build_pipeline(build_preprocessor(X),
                                       make_model(mname), use_smote=True),
                        X, y, cv=cv, method="predict_proba", n_jobs=None
                    )
                    y_proba = proba[:, 1]
                except Exception:
                    y_proba = None
                # Isi OOF — skor metrik (mean/std) TIDAK disentuh
                cell.y_true  = np.asarray(y)
                cell.y_pred  = np.asarray(y_pred)
                cell.y_proba = y_proba
                logger.info("    %-12s OOF restored (n=%d)", mname, len(y_pred))
            except Exception as exc:
                logger.warning(
                    "    [OOF] Gagal restore '%s' @ '%s': %s", mname, dkey, exc
                )
    return cached_results


# --------------------------------------------------------------------------- #
# Pipeline utama
# --------------------------------------------------------------------------- #
def run_all(dataset_keys: list[str] | None = None,
            do_shap: bool = True,
            do_figures: bool = True) -> dict:
    t0 = time.time()
    setup_logging()

    logger.info("=" * 70)
    logger.info("CROSS-DATASET CONSISTENCY EXPERIMENT")
    logger.info("random_state=%d | folds=%d | quick=%s",
                config.RANDOM_STATE, config.N_SPLITS, config.QUICK_MODE)
    logger.info("=" * 70)

    available = dataset_keys or discover_available_datasets()
    if not available:
        logger.error(
            "Tidak ada dataset yang tersedia. Jalankan: python download_data.py "
            "atau letakkan CSV pada folder datasets/. Lihat README.md."
        )
        return {"error": "no_datasets"}

    # ------------------------------------------------------------------ #
    # RESUME: pisahkan dataset yang sudah selesai vs yang belum
    # ------------------------------------------------------------------ #
    cached_results = _load_completed_datasets()
    done_keys, todo_keys = _partition_datasets(available)

    logger.info("Status dataset:")
    logger.info("  Selesai (cache) : %s",
                [config.DATASETS[k]["short"] for k in done_keys] or "–")
    logger.info("  Perlu dilatih   : %s",
                [config.DATASETS[k]["short"] for k in todo_keys] or "–")

    if not todo_keys and not done_keys:
        return {"error": "no_datasets"}

    # --- 1. Muat dataset yang BELUM selesai saja ---
    datasets_new: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for key in todo_keys:
        try:
            datasets_new[key] = preprocessing.load_dataset(key)
        except Exception as exc:
            logger.error("Gagal memuat dataset '%s': %s", key, exc)

    # --- 2. Evaluasi performa hanya untuk dataset baru ---
    logger.info("-" * 70)
    logger.info("[1/9] Evaluasi performa (Stratified %d-Fold CV)", config.N_SPLITS)

    all_results: dict[str, dict] = {}

    # Masukkan hasil cache (sudah selesai) ke all_results
    for key in done_keys:
        all_results[key] = cached_results[key]

    # Latih hanya yang belum
    for key in todo_keys:
        if key not in datasets_new:
            continue
        X, y = datasets_new[key]
        all_results[key] = evaluation.evaluate_dataset(key, X, y)

    if not all_results:
        return {"error": "evaluation_failed"}

    # Simpan performance.csv: merge hasil baru dengan lama
    if todo_keys:
        new_perf = evaluation.results_to_performance_table(
            {k: all_results[k] for k in todo_keys if k in all_results}
        )
        performance = _merge_performance_csv(new_perf)
    else:
        # Semua dari cache, baca langsung
        performance = pd.read_csv(config.TABLE_DIR / "performance.csv")

    _save_csv(performance, "performance.csv")

    # Kumpulkan mean_f1 & f1_folds dari ALL results (cache + baru)
    mean_f1  = evaluation.collect_mean_metric(all_results, metric="f1")
    f1_folds = evaluation.collect_f1_folds(all_results)

    # --- 3. ΔF1 ---
    logger.info("[2/9] Analisis konsistensi performa (|ΔF1|)")
    delta = cons_mod.delta_f1_table(mean_f1)
    _save_csv(delta.drop(columns=["_perf_pass"], errors="ignore"), "delta_f1.csv")

    # --- 4. Wilcoxon ---
    logger.info("[3/9] Uji Wilcoxon signed-rank")
    wilcox = stats_mod.wilcoxon_across_pairs(f1_folds)
    _save_csv(wilcox, "wilcoxon.csv")

    # --- 5. Ranking + Spearman ---
    logger.info("[4/9] Ranking model & korelasi Spearman")
    ranks = stats_mod.model_ranks_per_dataset(mean_f1).reset_index().rename(
        columns={"index": "Model"})
    _save_csv(ranks, "ranking.csv")
    spearman_rank = stats_mod.spearman_ranking_across_pairs(mean_f1)
    _save_csv(spearman_rank, "spearman.csv")

    # --- 6. Friedman + Nemenyi ---
    logger.info("[5/9] Friedman test + Nemenyi post-hoc")
    friedman = stats_mod.friedman_test(mean_f1)
    friedman_df = pd.DataFrame({
        "Statistik": ["chi_square", "p_value", "df", "k_models", "n_datasets"],
        "Nilai": [
            None if friedman["chi_square"] is None else round(friedman["chi_square"], 4),
            None if friedman["p_value"] is None else round(friedman["p_value"], 4),
            friedman["df"], friedman["k_models"], friedman["n_datasets"],
        ],
    })
    _save_csv(friedman_df, "friedman.csv")
    nemenyi = stats_mod.nemenyi_posthoc(mean_f1)
    if nemenyi is not None:
        nemenyi.to_csv(config.TABLE_DIR / "nemenyi.csv")
        logger.info("  tabel disimpan: nemenyi.csv")

    # --- 7. SHAP ---
    all_shap: dict[str, dict] = {}
    if do_shap:
        logger.info("[6/9] Analisis explainability SHAP")
        # SHAP hanya dihitung untuk dataset yang punya data aktual (bukan pure cache)
        datasets_for_shap = {
            k: datasets_new[k]
            for k in todo_keys
            if k in datasets_new
        }
        # Untuk dataset dari cache, muat ulang datanya agar SHAP bisa dihitung
        for key in done_keys:
            try:
                datasets_for_shap[key] = preprocessing.load_dataset(key)
            except Exception as exc:
                logger.warning("Gagal muat ulang '%s' untuk SHAP: %s", key, exc)

        all_shap = shap_analysis.compute_all_shap(datasets_for_shap)
        top3 = shap_analysis.shap_to_top3_table(all_shap)
        _save_csv(top3, "shap_top3.csv")

        shap_spearman_rows = []
        for mname in config.MODEL_ORDER:
            df = stats_mod.spearman_shap_across_pairs(all_shap, mname)
            df.insert(0, "Model", config.MODEL_LABELS[mname])
            shap_spearman_rows.append(df)
        if shap_spearman_rows:
            _save_csv(pd.concat(shap_spearman_rows, ignore_index=True),
                      "shap_spearman.csv")
    else:
        logger.info("[6/9] SHAP dilewati (do_shap=False)")

    # --- 8. Konsistensi tiga dimensi ---
    logger.info("[7/9] Evaluasi konsistensi tiga dimensi")
    shap_overlap = cons_mod.shap_overlap_table(all_shap) if all_shap else pd.DataFrame()
    if not shap_overlap.empty:
        _save_csv(shap_overlap.drop(columns=["_expl_pass"], errors="ignore"),
                  "shap_overlap.csv")
    consistency = cons_mod.synthesize(mean_f1, all_shap)
    _save_csv(consistency, "consistency.csv")

    logger.info("\n%s", consistency.to_string(index=False))

    # --- 9. Figur ---
    if do_figures:
        logger.info("[8/9] Pembuatan figur")
        try:
            # Dataset dari cache tidak punya OOF predictions.
            # Restore dulu sebelum generate figur.
            if done_keys:
                logger.info(
                    "  [OOF] Dataset cache ditemukan (%s) — "
                    "restore OOF predictions untuk confusion matrix & ROC ...",
                    [config.DATASETS[k]["short"] for k in done_keys]
                )
                # Muat data aktual untuk dataset cache
                datasets_for_oof: dict = {}
                for key in done_keys:
                    try:
                        datasets_for_oof[key] = preprocessing.load_dataset(key)
                    except Exception as exc:
                        logger.warning("Gagal muat '%s' untuk OOF: %s", key, exc)

                # Restore OOF hanya pada cached_results (done_keys)
                cached_portion = {k: all_results[k] for k in done_keys
                                  if k in all_results}
                _restore_oof_predictions(cached_portion, datasets_for_oof)

                # Update all_results dengan OOF yang sudah direstored
                all_results.update(cached_portion)

            viz.generate_all_figures(all_results, all_shap, mean_f1)
        except Exception as exc:
            logger.error("Pembuatan figur sebagian gagal: %s", exc)

    # --- 10. Laporan otomatis ---
    logger.info("[9/9] Menyusun laporan otomatis (docs/experiment_report.md)")
    try:
        import generate_report
        generate_report.build_report()
    except Exception as exc:
        logger.error("Pembuatan laporan gagal: %s", exc)

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("SELESAI dalam %.1f detik. Lihat outputs/ dan docs/.", elapsed)
    logger.info("=" * 70)

    return {
        "performance": performance,
        "mean_f1": mean_f1,
        "delta_f1": delta,
        "wilcoxon": wilcox,
        "ranking": ranks,
        "spearman": spearman_rank,
        "friedman": friedman,
        "consistency": consistency,
        "all_results": all_results,
        "all_shap": all_shap,
    }