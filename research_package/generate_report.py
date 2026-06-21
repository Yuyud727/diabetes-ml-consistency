"""
generate_report.py
==================
Otomatisasi BAB 4 (Tahap 9).

Membaca seluruh tabel hasil pada outputs/tables/ dan figur pada outputs/figures/,
lalu menyusun docs/experiment_report.md dengan:
    - tabel hasil yang sudah terisi angka (atau [BELUM ADA] bila kosong),
    - caption gambar dan path figur yang sesuai,
    - kerangka analisis yang mengikuti struktur BAB 4.

Dijalankan otomatis di akhir run_experiment.py, atau manual:
    python generate_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402


def _read(name: str) -> pd.DataFrame | None:
    path = config.TABLE_DIR / name
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _md_table(df: pd.DataFrame | None, placeholder: str = "_[BELUM ADA HASIL]_") -> str:
    if df is None or df.empty:
        return placeholder + "\n"
    return df.to_markdown(index=False) + "\n"


def _list_figures(subdir: Path) -> list[str]:
    if not subdir.exists():
        return []
    return sorted(str(p.relative_to(config.ROOT_DIR)) for p in subdir.glob("*.png"))


def build_report() -> Path:
    config.ensure_dirs()

    performance = _read("performance.csv")
    delta = _read("delta_f1.csv")
    wilcoxon = _read("wilcoxon.csv")
    ranking = _read("ranking.csv")
    spearman = _read("spearman.csv")
    friedman = _read("friedman.csv")
    nemenyi_path = config.TABLE_DIR / "nemenyi.csv"
    nemenyi = pd.read_csv(nemenyi_path, index_col=0) if nemenyi_path.exists() else None
    shap_top3 = _read("shap_top3.csv")
    shap_overlap = _read("shap_overlap.csv")
    consistency = _read("consistency.csv")

    lines: list[str] = []
    add = lines.append

    add("# BAB 4 — HASIL DAN PEMBAHASAN (Laporan Otomatis)\n")
    add("> Dokumen ini dihasilkan otomatis oleh `generate_report.py` dari "
        "berkas pada `outputs/`. Angka pada tabel terisi langsung dari hasil "
        "eksperimen; sel kosong menandakan hasil belum tersedia.\n")
    add(f"- Konfigurasi: `random_state={config.RANDOM_STATE}`, "
        f"`{config.N_SPLITS}-Fold Stratified CV`, "
        f"mode quick = `{config.QUICK_MODE}`.\n")

    # 4.1
    add("\n## 4.1 Hasil Eksperimen\n")
    add("Eksperimen mencakup kombinasi 4 model × dataset aktif, masing-masing "
        f"dievaluasi dengan {config.N_SPLITS}-fold cross-validation.\n")
    add("\n**Tabel 4.1 Hasil Performa Seluruh Model pada Seluruh Dataset**\n")
    add(_md_table(performance))

    for dkey in config.DATASET_ORDER:
        short = config.DATASETS[dkey]["short"]
        cms = [f for f in _list_figures(config.FIG_CONFUSION) if f"_{short}." in f]
        for cm in cms:
            add(f"\n*Gambar 4.1 Confusion Matrix* — `{cm}`\n")
    for dkey in config.DATASET_ORDER:
        short = config.DATASETS[dkey]["short"]
        for roc in [f for f in _list_figures(config.FIG_ROC) if short in f]:
            add(f"\n*Gambar 4.2 Kurva ROC @ {short}* — `{roc}`\n")

    # 4.2
    add("\n## 4.2 Analisis Konsistensi Performa\n")
    add("**Tabel 4.2 Nilai ΔF1 Antar Dataset**\n")
    add(_md_table(delta))

    # 4.3
    add("\n## 4.3 Hasil Uji Wilcoxon\n")
    add("**Tabel 4.3 Hasil Wilcoxon Signed-Rank Test**\n")
    add(_md_table(wilcoxon))

    # 4.4
    add("\n## 4.4 Analisis Ranking Model\n")
    add("**Tabel 4.4 Ranking Model Berdasarkan F1**\n")
    add(_md_table(ranking))
    add("\n**Korelasi Spearman peringkat antar dataset**\n")
    add(_md_table(spearman))
    for rk in _list_figures(config.FIG_RANKING):
        add(f"\n*Gambar 4.3 Perbandingan Ranking Model* — `{rk}`\n")

    # 4.5
    add("\n## 4.5 Hasil Friedman Test dan Nemenyi\n")
    add("**Tabel 4.5 Hasil Friedman Test**\n")
    add(_md_table(friedman))
    if nemenyi is not None:
        add("\n**Nemenyi post-hoc (matriks p-value)**\n")
        add(nemenyi.to_markdown() + "\n")

    # 4.6
    add("\n## 4.6 Analisis Explainability SHAP\n")
    for ss in _list_figures(config.FIG_SHAP_SUMMARY):
        add(f"\n*Gambar 4.4 SHAP Summary Plot* — `{ss}`\n")
    for sf in _list_figures(config.FIG_SHAP_FORCE):
        add(f"\n*Gambar 4.5 SHAP Force Plot* — `{sf}`\n")

    # 4.7
    add("\n## 4.7 Analisis Stabilitas Fitur\n")
    add("**Tabel 4.6 Top-3 SHAP Feature**\n")
    add(_md_table(shap_top3))
    if shap_overlap is not None:
        add("\n**Overlap top-3 SHAP antar dataset**\n")
        add(_md_table(shap_overlap))
    for fc in _list_figures(config.FIG_FEATURE):
        add(f"\n*Gambar 4.6 Perbandingan Feature Importance SHAP* — `{fc}`\n")

    # 4.8
    add("\n## 4.8 Sintesis Cross-Dataset Consistency\n")
    add("**Tabel 4.7 Rekapitulasi Konsistensi Tiga Dimensi**\n")
    add(_md_table(consistency))
    if consistency is not None and not consistency.empty and "Konsisten" in consistency:
        konsisten = consistency.loc[consistency["Konsisten"] == "YA", "Model"].tolist()
        if konsisten:
            add(f"\nModel yang memenuhi ketiga dimensi: **{', '.join(konsisten)}**.\n")
        else:
            add("\nPada konfigurasi ini, belum ada model yang memenuhi ketiga "
                "dimensi sekaligus.\n")

    # 4.9
    add("\n## 4.9 Diskusi Penelitian\n")
    add("_(Bagian diskusi naratif diisi pada dokumen BAB 4 berformat Word; "
        "laporan ini menyediakan basis numerik dan figur pendukungnya.)_\n")

    out = config.DOCS_DIR / "experiment_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[generate_report] Laporan ditulis ke: {out}")
    return out


if __name__ == "__main__":
    build_report()
