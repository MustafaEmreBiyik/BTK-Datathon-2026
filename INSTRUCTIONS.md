# INSTRUCTIONS — BTK Akademi Datathon 2026

> Bu dosya projenin kalıcı bağlamıdır. Yeni bir sohbete başlarken bunu paylaşman (ya da
> proje "custom instructions" alanına yapıştırman) yeterli — her şeyi baştan anlatmana gerek yok.

## 1. Biz kimiz, hedef ne
- Bilgisayar mühendisliği öğrencisi + 1 takım arkadaşı. Takım olarak yarışıyoruz.
- **Hedef:** Kaggle private leaderboard'da **ilk 10** → jüri sunumu → **ilk 3**.

## 2. Problem
- **Tür:** Regresyon (supervised).
- **Hedef değişken:** `career_success_score`, 0–100 arası sürekli değer.
- **Görev:** test setindeki her öğrenci için `career_success_score` tahmini.
- **Veri tipleri:** sayısal + kategorik + **doğal dil** (`mentor_feedback_text`, Türkçe).
  Yarışma açıklaması metin alanının kullanılmasını açıkça bekliyor — bu bizim ayrışma noktamız.

## 3. Dosyalar
- `train.csv` — 10.000 satır, 47 kolon (hedef dahil).
- `test_x.csv` — 10.000 satır, 46 kolon (hedef yok). *(Açıklamada adı `test.csv`.)*
- `sample_submission.csv` — submission formatı: `student_id, career_success_score`.

## 4. Veriye dair bilinmesi gerekenler (EDA özeti)
- **Hedef:** ort. ≈ 76.9, std ≈ 15.2, min 0, max 100. Sadece 1 adet 0 değeri (outlier, dokunma).
- **Eksik kolonlar (train):** `internship_duration_months` (1657), `english_exam_score` (953),
  `github_avg_stars` (910), `open_source_contribution_count` (910), `hr_interview_score` (780),
  `linkedin_profile_score` (668), `portfolio_score` (364).
  - `internship_duration_months` eksikliği çoğunlukla `internship_count = 0` demek → 0 + flag.
  - GitHub eksikliği repo sayısından bağımsız → indicator + imputation.
- **Kategorikler:** `department` (7), `university_tier` (4, **sıralı**: Tier1>Tier2>Tier3>Tier4),
  `target_role` (11), `hobby` (8), `preferred_social_media_platform` (6).
- **En güçlü sayısal sinyaller:** `project_quality_score` (~0.54), `technical_interview_score`,
  `problem_solving_score`, `coding_score`, `cloud_score`, `portfolio_score`.

## 5. Kritik kısıtlar (kurallardan)
- **Son tarih:** 14 Haziran 2026, 23:59. (Yarışma 9 Haziran 20:00'de başladı.)
- **Submission:** günde **maks 5**; finalde **2** submission seçilir.
- **Skor bölünmesi:** %60 public / %40 private, rastgele. → **Public leaderboard'a güvenme,
  lokal CV'ye güven.** Public'e overfit etmek finalde bizi yakar.
- **Kaggle sıralaması ≠ final sıralama.** İlk 10 → jüriye **5+3 dk sunum** + **notebook paylaşımı zorunlu**.
- **btkakademi.gov.tr kaydı şart** (yapılmazsa ilk 10'da bile diskalifiye). Takım adı kayıttakiyle aynı olmalı.
- Tek Kaggle hesabı; takımlar arası veri/çözüm transferi yasak.
- **Değerlendirme metriği: MSE (Mean Squared Error) — DOĞRULANDI.** Büyük hatalar kare ile
  cezalandırılır → tahminleri [0,100] clip et, uç değerlere dikkat. GBM'lerin `l2`/`squared_error`
  loss'u doğrudan metrikle uyumlu. CV'de de MSE raporla (Kaggle ile aynı dil).

## 6. Nasıl çalışıyoruz (önemli)
- **Rolüm: mentor/öğretmen ve eş-pilot.** Stratejiyi kurar, kodu açıklar, hataları ayıklar, fikir veririm.
- **Kodu siz yazıp çalıştırıp anlıyorsunuz.** İki sebep:
  1. İlk 10'a girersek jüriye süreci **kendiniz** anlatacak ve notebook paylaşacaksınız —
     anlamadığınız bir çözümü savunamazsınız.
  2. Kurallar işin kendi emeğinizle yapılmasını şart koşuyor.
- Yani her adımda "neden böyle yapıyoruz"u konuşuruz; kopyala-yapıştır değil, öğrenerek ilerleriz.

## 7. Planlama dosyaları
- **`PLAN.md`** — yarışma sonuna kadar tüm strateji ve faz planı.
- **`PROGRESS.md`** — yapılanlar, devam edenler, yapılacaklar + submission/CV kayıt defteri.
- Her oturumda önce `PROGRESS.md`'ye bakıp kaldığımız yerden devam ederiz.
