# EXP-A — Metin Derinleştirme (Text Deepening)
**Atanan:** Kişi 1 | **Tahmini süre:** 2–3 saat | **Hedef:** CV MSE < 79

> Yarışma açıklaması `mentor_feedback_text` kullanımını açıkça bekliyor.  
> Şu ana kadar sadece keyword sayımı yaptık (82.36). Bu deney metni ciddi biçimde işler.

---

## Ön Koşul
`02_full_experiment.ipynb`'yi **en azından "Apply FE" hücresine kadar** çalıştır.  
`train_texts`, `test_texts`, `X`, `X_test`, `y`, `kf` değişkenleri bellekte olsun.

---

## A1 — TF-IDF Alpha Arama (15 dk)

`make_tfidf_oof` fonksiyonu alpha=10 kullanıyor. CV üzerinde doğru değeri bul:

```python
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np

def tfidf_cv_alpha(train_texts, test_texts, y, kf, alpha):
    oof = np.zeros(len(train_texts))
    test_pred = np.zeros(len(test_texts))
    for tr_idx, val_idx in kf.split(range(len(train_texts))):
        tr_txt  = [train_texts[i] for i in tr_idx]
        val_txt = [train_texts[i] for i in val_idx]
        vec_w = TfidfVectorizer(analyzer='word',    ngram_range=(1, 3),
                                max_features=20000, sublinear_tf=True, min_df=2)
        vec_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5),
                                max_features=20000, sublinear_tf=True, min_df=2)
        Xtr  = sparse.hstack([vec_w.fit_transform(tr_txt),  vec_c.fit_transform(tr_txt)])
        Xval = sparse.hstack([vec_w.transform(val_txt),     vec_c.transform(val_txt)])
        Xte  = sparse.hstack([vec_w.transform(test_texts),  vec_c.transform(test_texts)])
        ridge = Ridge(alpha=alpha)
        ridge.fit(Xtr, y[tr_idx])
        oof[val_idx]  = ridge.predict(Xval)
        test_pred    += ridge.predict(Xte) / kf.n_splits
    mse = ((y - np.clip(oof, 0, 100)) ** 2).mean()
    return oof, test_pred, mse

results = {}
for alpha in [1, 5, 10, 30, 50, 100, 200]:
    _, _, mse = tfidf_cv_alpha(train_texts, test_texts, y, kf, alpha)
    results[alpha] = mse
    print(f'alpha={alpha:>5}  MSE={mse:.4f}')

best_alpha = min(results, key=results.get)
print(f'\nBest alpha: {best_alpha}  MSE: {results[best_alpha]:.4f}')
```

En iyi alpha ile tekrar çalıştır ve OOF'u kaydet:

```python
tfidf_a_oof, tfidf_a_test, tfidf_a_mse = tfidf_cv_alpha(
    train_texts, test_texts, y, kf, alpha=best_alpha
)
np.save('../oof/exp_a_tfidf_oof.npy',  tfidf_a_oof)
np.save('../oof/exp_a_tfidf_test.npy', tfidf_a_test)
print(f'TF-IDF (tuned) CV MSE: {tfidf_a_mse:.4f}')
```

---

## A2 — Sentence Embeddings: paraphrase-multilingual-MiniLM-L12-v2 (45 dk)

Bu model Türkçeyi native olarak destekler. 10k metin için encoding ~10-20 dk sürer.

```python
# Önce: pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import numpy as np

print('Model yükleniyor...')
sbert = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

all_texts = train_texts + test_texts
print(f'{len(all_texts)} metin encode ediliyor...')
emb = sbert.encode(all_texts, batch_size=64, show_progress_bar=True,
                   normalize_embeddings=True)

train_emb = emb[:len(train_texts)]   # (10000, 384)
test_emb  = emb[len(train_texts):]   # (10000, 384)
print(f'Embedding boyutu: {train_emb.shape}')
```

Embedding'den fold-safe OOF Ridge tahmini üret:

```python
def ridge_oof_on_emb(train_emb, test_emb, y, kf, n_components=128, alpha=10.0):
    """SVD ile boyut indir, ardından Ridge ile OOF tahmini üret."""
    # Tüm embedding'i birlikte fit et (boyut indirgeme için leakage yok — unsupervised)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_r = svd.fit_transform(train_emb)
    test_r  = svd.transform(test_emb)
    print(f'SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}')

    oof = np.zeros(len(train_r)); test_pred = np.zeros(len(test_r))
    for tr_idx, val_idx in kf.split(train_r):
        ridge = Ridge(alpha=alpha)
        ridge.fit(train_r[tr_idx], y[tr_idx])
        oof[val_idx]  = ridge.predict(train_r[val_idx])
        test_pred    += ridge.predict(test_r) / kf.n_splits
    mse = ((y - np.clip(oof, 0, 100)) ** 2).mean()
    print(f'SBERT + Ridge CV MSE: {mse:.4f}')
    return oof, test_pred, mse, train_r, test_r

sbert_oof, sbert_test, sbert_mse, train_emb_r, test_emb_r = ridge_oof_on_emb(
    train_emb, test_emb, y, kf, n_components=128, alpha=10.0
)
np.save('../oof/exp_a_sbert_oof.npy',  sbert_oof)
np.save('../oof/exp_a_sbert_test.npy', sbert_test)
np.save('../oof/exp_a_train_emb.npy',  train_emb_r)
np.save('../oof/exp_a_test_emb.npy',   test_emb_r)
```

---

## A3 — LightGBM + Embedding Features (45 dk)

Embedding'i doğrudan LightGBM feature'ı olarak ekle:

```python
import pandas as pd
import lightgbm as lgb

# 02_full_experiment.ipynb'den X ve X_test var
# Embedding kolonlarını ekle
emb_cols = [f'sbert_{i}' for i in range(train_emb_r.shape[1])]
X_sbert      = pd.concat([X.reset_index(drop=True),
                           pd.DataFrame(train_emb_r, columns=emb_cols)], axis=1)
X_test_sbert = pd.concat([X_test.reset_index(drop=True),
                           pd.DataFrame(test_emb_r,  columns=emb_cols)], axis=1)

# TF-IDF OOF'u da ekle (best_alpha ile üretilen)
X_sbert['tfidf_oof']      = tfidf_a_oof
X_test_sbert['tfidf_oof'] = tfidf_a_test

print(f'Feature matrix with embeddings: {X_sbert.shape}')

# run_lgbm fonksiyonu 02_full_experiment.ipynb'den
lgbm_sbert_oof, lgbm_sbert_test, lgbm_sbert_cv = run_lgbm(X_sbert, y, X_test_sbert)

np.save('../oof/exp_a_lgbm_sbert_oof.npy',  lgbm_sbert_oof)
np.save('../oof/exp_a_lgbm_sbert_test.npy', lgbm_sbert_test)
print(f'LGBM + SBERT CV MSE: {lgbm_sbert_cv:.4f}')
```

---

## A4 — Submission (Tek Model)

```python
import pandas as pd
from pathlib import Path

DATA   = Path('../datathon-2026')
SUBDIR = Path('../submissions')

# En iyi EXP-A sonucunu submission yap
best_preds = lgbm_sbert_test  # ya da karşılaştır, hangisi daha iyi CV'ye sahipse
best_cv    = lgbm_sbert_cv

sub = pd.DataFrame({
    'student_id': pd.read_csv(DATA / 'test_x.csv')['student_id'],
    'career_success_score': best_preds,
})
fname = f'sub_A_lgbm_sbert_cv{best_cv:.2f}.csv'
sub.to_csv(SUBDIR / fname, index=False)
print(f'Kaydedildi: {fname}')
```

---

## Rapor

Bu deneyin sonunda takım arkadaşına şunları söyle:

| Dosya | İçerik |
|---|---|
| `oof/exp_a_tfidf_oof.npy` | Tuned TF-IDF Ridge OOF (10k) |
| `oof/exp_a_tfidf_test.npy` | Tuned TF-IDF Ridge test pred (10k) |
| `oof/exp_a_sbert_oof.npy` | SBERT Ridge OOF (10k) |
| `oof/exp_a_sbert_test.npy` | SBERT Ridge test pred (10k) |
| `oof/exp_a_lgbm_sbert_oof.npy` | LGBM + SBERT OOF (10k) |
| `oof/exp_a_lgbm_sbert_test.npy` | LGBM + SBERT test pred (10k) |

**CV skorlarını PROGRESS.md'ye yaz** ve EXP-C için hazır ol.

---

## Beklenen Kazanım

| Adım | Beklenen CV MSE |
|---|---|
| Baseline (keyword stats) | 82.36 |
| TF-IDF tuned | ~80–81 |
| + SBERT embedding feature | ~78–80 |
| + LGBM ile birleşim | **< 79** hedef |
