# PLAN — BTK Akademi Datathon 2026

> Yarışma sonuna (14 Haziran 23:59) kadar tüm stratejimiz. `PROGRESS.md` bu planın ne kadarının
> yapıldığını takip eder. Plan canlıdır; öğrendikçe güncelleriz.

## 0. Hedef & başarı kriteri
- **Birincil hedef:** private leaderboard'da ilk 10 → jüri → ilk 3.
- **Pratik hedef:** mümkün olan en düşük **MSE** (doğrulandı), ama **lokal CV'ye göre** — public LB'ye değil.
- **Yan hedef:** temiz, açıklanabilir, tekrar üretilebilir notebook (jüri için zorunlu).

## 1. Validasyon stratejisi (HER ŞEYİN TEMELİ)
- **5-fold KFold** (shuffle, sabit seed). Skor 60/40 rastgele bölündüğü için sağlam CV şart.
- Hedef sürekli ama dağılımı dengeli kalsın diye **target'i binlere ayırıp StratifiedKFold** opsiyonunu da deneyeceğiz.
- **Kural:** Her değişikliği önce CV ile ölç. CV ↑ ama public LB ↓ ise CV'ye güven.
- Tüm modellerde **aynı fold'ları** kullan (adil karşılaştırma + stacking için OOF tahminleri sakla).
- Her submission'ın yanına lokal CV skorunu not et (PROGRESS.md'deki tabloya).

## 2. Preprocessing
- **Eksik değerler:**
  - `internship_duration_months`: `internship_count==0` ise 0, değilse medyan + `was_missing` flag.
  - `github_avg_stars`, `open_source_contribution_count`, `hr_interview_score`,
    `linkedin_profile_score`, `portfolio_score`, `english_exam_score`: medyan/0 imputation + her biri için `_isna` flag.
  - Eksiklik flag'leri çoğu zaman değerli sinyaldir — mutlaka tut.
- **Kategorik encoding:**
  - `university_tier` → **ordinal** (Tier1=1 ... Tier4=4 veya tersi; CV ile yön seç).
  - `department`, `target_role`, `hobby`, `preferred_social_media_platform` → ağaç modelleri için
    label/target encoding; lineer modeller için one-hot.
  - **CatBoost** kategorikleri ve metni native işler → ayrı bir kozumuz.
- **Hedef dönüşümü:** Metrik MSE → squared loss; ortalamaya sapmasız tahmin ideal, ekstra dönüşüm
  şart değil. Dağılım çarpıksa log/clip denenebilir ama önce CV ile kontrol et.
  Tahminleri **[0, 100] aralığına clip** etmeyi unutma (uç hatalar MSE'de kare ile cezalanır).

## 3. Feature engineering (ayrışma noktamız — sayısal taraf)
- **Oran/etkileşim feature'ları:**
  - `interview_rate = interviews_attended / (applications_sent + 1)` (dönüşüm oranı).
  - `award_per_hackathon = hackathon_awards / (hackathon_count + 1)`.
  - `stars_per_repo = github_avg_stars * github_repo_count` (toplam etki).
- **Toplam/ortalama skorlar:** teknik beceri skorlarının (coding, problem_solving, data_structures,
  sql, ml, backend, frontend, cloud, devops) ortalaması / max'ı / std'si.
- **Soft-skill bloğu:** communication, teamwork, leadership, presentation ortalaması.
- **Deneyim yoğunluğu:** internship + freelance + real_client_project + hackathon kombinasyonları.
- `age - (graduation_year - application_year)` gibi tutarlılık/zaman feature'ları.
- Her yeni feature'ı CV ile test et; faydasızları at (feature importance + permutation).

## 4. Metin: `mentor_feedback_text` (çoğu rakibin atlayacağı yer)
Aşamalı gideceğiz, en basitten başlayıp işe yarayanı bırakacağız:
- **Seviye 1 — TF-IDF:** Türkçe, word (1-2 gram) + char (3-5 gram) n-gram. Üstüne Ridge/LightGBM.
  Bu OOF tahminini ana modele feature olarak ver (stacking) veya birlikte eğit.
- **Seviye 2 — basit metin istatistikleri:** uzunluk, kelime sayısı, olumlu/olumsuz anahtar kelime
  sayımı ("dikkat çekici", "gelişmeli", "umut verici", "çalışması gerekiyor" vb.).
- **Seviye 3 (opsiyonel) — embeddings:** çok dilli sentence-transformer (örn. `paraphrase-multilingual-MiniLM`)
  ile cümle embedding'i → boyut indirip (PCA/SVD) ağaç modeline feature. Kaggle'da internet kapalıysa
  modeli "dataset" olarak ekle. Zaman kalırsa.
- **CatBoost text features** ile de doğrudan deneyeceğiz.

## 5. Modeller
- **Baseline'lar (gün 1):** ortalama tahmini → Ridge → tek LightGBM. Referans CV elde et.
- **Ana modeller:** LightGBM, XGBoost, **CatBoost** (kategorik+metin native).
- **Lineer model:** Ridge/ElasticNet (TF-IDF üzerinde) — ensemble çeşitliliği için.
- **Hiperparametre:** Optuna ile, fold'lara sadık kalarak. Önce kaba, sonra ince arama.

## 6. Ensembling
- Her modelin **OOF** ve test tahminlerini sakla.
- **Blend:** CV'yi optimize eden ağırlıklarla (veya basit ortalama) birleştir.
- **Stacking:** OOF tahminleri + birkaç güçlü feature üzerinde meta-model (Ridge).
- Çeşitlilik önemli: ağaç + lineer + metin modelleri farklı hatalar yapar → ensemble kazanır.

## 7. Submission stratejisi
- Günde 5 hak var → planlı kullan, her birini logla (PROGRESS.md tablosu).
- Public LB'yi sadece "kabaca tutarlı mı" diye kontrol et; karar mercii **lokal CV**.
- **Final 2 submission seçimi:**
  1. En iyi lokal CV'li ensemble.
  2. Ondan **farklı/çeşitli** ama sağlam ikinci bir model (risk dağıtımı).
  Aynı şeyin iki versiyonunu seçme.

## 8. Notebook / jüri hazırlığı (paralel yürüt)
- Kod baştan **temiz ve tekrar üretilebilir** olsun (seed sabit, hücreler sıralı çalışsın).
- Süreç boyunca "neden bu kararı verdik" notları tut → sunumun iskeleti hazır olur.
- İlk 10'a girersek: 5+3 dk sunum (analiz + süreç + sonuç) ve notebook paylaşımı.

## 9. Zaman planı (bugün 10 Haziran, deadline 14 Haziran 23:59)
- **Gün 1 (10 Haz):** Detaylı EDA, validasyon iskeleti, preprocessing, ilk LightGBM baseline + ilk submission.
- **Gün 2 (11 Haz):** Feature engineering (sayısal) + TF-IDF metin, CatBoost/XGBoost ekle, CV iyileştir.
- **Gün 3 (12 Haz):** Optuna tuning, metin Seviye 2-3, eksik kalan feature'lar.
- **Gün 4 (13 Haz):** Ensembling/stacking, sağlamlık kontrolü, notebook temizliği.
- **Gün 5 (14 Haz):** Final ince ayar, 2 submission seçimi, jüri sunum taslağı. **Erken bitir, son dakika bırakma.**

## 10. Riskler & kurallar (unutma)
- ⚠️ **btkakademi.gov.tr kaydını yap** — yoksa ilk 10'da bile diskalifiye.
- ⚠️ **Metriği Kaggle'dan doğrula** — yanlış metriğe optimize etmek zaman kaybı.
- ⚠️ Public LB'ye overfit etme (%40 private gizli).
- ⚠️ Veri sızıntısı yok: encoding/imputation/feature'ları **fold içinde fit et** (test/val'a sızdırma).
- ⚠️ Takım adı kayıttakiyle birebir aynı olsun.
