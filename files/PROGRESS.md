# PROGRESS — BTK Akademi Datathon 2026

> Canlı takip dosyası. Her oturumda önce buraya bak, kaldığın yerden devam et, sonra güncelle.
> Son güncelleme: 10 Haziran 2026

## Durum özeti
- **Aşama:** Gün 4 — Full experiment notebook hazır, çalıştırılacak.
- **En iyi lokal CV:** 82.36 (LightGBM + FE + text stats, notebook 01)
- **Yapılan submission sayısı (bugün):** 0 / 5
- **Doğrulanmış metrik:** ✅ **MSE** (Mean Squared Error). CV'de MSE raporla, tahminleri [0,100] clip et.

## ✅ Yapıldı
- [x] Veri yapısı incelendi (train/test/sample_submission).
- [x] İlk EDA: hedef dağılımı, eksik kolonlar, kategorikler, korelasyonlar.
- [x] Eksiklik desenleri doğrulandı (internship_duration ↔ internship_count; github eksikliği bağımsız).
- [x] INSTRUCTIONS.md, PLAN.md, PROGRESS.md oluşturuldu.

## 🔄 Devam ediyor
- [ ] (boş)

## 📋 Yapılacaklar (öncelik sırasıyla)

### Hemen (kritik / bloklayıcı)
- [x] ~~Kaggle Evaluation sekmesinden METRİĞİ doğrula~~ → **MSE** (doğrulandı).
- [ ] **btkakademi.gov.tr kaydını tamamla**, takım adını not et.
- [ ] Submission dosya formatını sample_submission ile birebir eşle.

### Gün 1 — Temel
- [ ] Detaylı EDA: dağılımlar, target vs kategorik kırılımlar, feature importance ön bakış.
- [ ] 5-fold validasyon iskeletini kur (sabit seed, OOF saklama).
- [ ] Preprocessing pipeline: eksik flag'leri + imputation + tier ordinal encode.
- [ ] LightGBM baseline → ilk CV skoru → **ilk submission**.

### Gün 2 — Feature engineering + metin
- [ ] Oran/etkileşim ve toplam-skor feature'ları, her birini CV ile test et.
- [ ] TF-IDF (word+char) + Ridge/LGBM ile metin OOF feature'ı.
- [ ] CatBoost ve XGBoost ekle, CV karşılaştır.

### Gün 3 — Tuning + derin metin
- [ ] Optuna ile hiperparametre araması.
- [ ] Metin Seviye 2 (istatistik/keyword) ve gerekirse Seviye 3 (embeddings).
- [ ] Faydasız feature'ları ele (importance/permutation).

### Gün 4 — Ensemble
- [ ] Blend (CV-optimize ağırlık) ve stacking (Ridge meta).
- [ ] Sağlamlık: seed/fold varyasyonu, [0,100] clip kontrolü.
- [ ] Notebook'u temizle, baştan sona çalıştığını doğrula.

### Gün 5 — Final
- [ ] Son ince ayar.
- [ ] **2 final submission seç** (en iyi CV + çeşitli ikinci model).
- [ ] Jüri sunum taslağını çıkar (analiz + süreç + sonuç).

## 📊 Submission kayıt defteri
| # | Tarih | Açıklama (model/feature) | Lokal CV | Public LB | Final aday? |
|---|-------|--------------------------|----------|-----------|-------------|
| 1 | 10 Haz | LightGBM baseline (notebook 01) | 83.83 | — | hayır |

## 📈 CV deney kayıt defteri
| Deney | Model | Önemli değişiklik | CV skoru | Not |
|-------|-------|-------------------|----------|-----|
| 01 | Mean baseline | — | 230.65 | referans |
| 02 | LightGBM | baseline | 83.83 | → submission |
| 03 | LightGBM | + ratio/agg FE | 83.61 | küçük iyileşme |
| 04 | LightGBM | + text stats (keyword sentiment) | 82.36 | en iyi şu an |
| 05 | **02_full_experiment.ipynb** | isna flags + FE++ + TF-IDF OOF + LGBM+XGB+CB + ensemble | **çalıştırılacak** | hedef <78 |

## 🧠 Kararlar / öğrenilenler
- (Buraya "şunu denedik, çünkü..., sonuç..." notları yaz. Jüri sunumunun hammaddesi burası olacak.)
