# BTK Akademi Datathon 2026

`career_success_score` hedef değişkenini tahmin etmeye yönelik bir regresyon yarışması projesi.

## Klasör Yapısı

- `data/raw/` — Ham yarışma verileri (`train.csv`, `test_x.csv`, `sample_submission.csv`). Git'e dahil edilmez.
- `notebooks/` — Keşif ve modelleme not defterleri (ör. `01_baseline.ipynb`).
- `oof/` — Out-of-fold tahminleri (model karşılaştırması ve stacking için).
- `submissions/` — Kaggle'a yüklenecek gönderim dosyaları.
- `src/` — Yeniden kullanılabilir Python kodu (özellik mühendisliği, eğitim, yardımcı fonksiyonlar).
