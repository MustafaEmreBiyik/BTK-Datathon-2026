# EXP-B — CatBoost Native Metin + Optuna Tuning
**Atanan:** Kişi 2 | **Tahmini süre:** 2–3 saat | **Hedef:** CV MSE < 79

> İki güçlü silah: (1) CatBoost metni native işler — Türkçe için ayrı TF-IDF kurmana gerek yok.  
> (2) Optuna, LightGBM için insan-gözü-ile-bulunamayacak hiper-parametre kombinasyonlarını arar.

---

## Ön Koşul
`02_full_experiment.ipynb`'yi **"Apply FE" hücresine kadar** çalıştır.  
`X`, `X_test`, `y`, `kf`, `train_texts`, `test_texts`, `CAT_COLS` bellekte olsun.

Gerekli kütüphaneler:
```bash
pip install catboost optuna
```

---

## B1 — Target Encoding (Fold-Safe) (20 dk)

Kategorik kolonlar için hedef ortalaması encoding'i — her fold'da sadece train verisinden öğren:

```python
import numpy as np
import pandas as pd

def target_encode_oof(X_tr, y_tr, X_val, X_te, col, smoothing=20):
    """
    Fold-safe target encoding: val ve test için encode uygular.
    Train seti için bu fonksiyonu döngü içinde çağır.
    smoothing: küçük grupları global ortalamaya çeker.
    """
    global_mean = y_tr.mean()
    stats = (
        pd.DataFrame({'cat': X_tr[col].astype(str), 'y': y_tr})
        .groupby('cat')['y']
        .agg(['mean', 'count'])
    )
    smooth_enc = (
        (stats['mean'] * stats['count'] + global_mean * smoothing)
        / (stats['count'] + smoothing)
    )
    val_enc = X_val[col].astype(str).map(smooth_enc).fillna(global_mean).values
    te_enc  = X_te[col].astype(str).map(smooth_enc).fillna(global_mean).values
    return val_enc, te_enc


TE_COLS = ['department', 'target_role', 'hobby',
           'preferred_social_media_platform', 'university_tier']

def add_target_encoding_oof(X, y, X_test, kf, te_cols=TE_COLS):
    """Tüm fold'lar için target encoding OOF matrisini üret."""
    X, X_test = X.copy(), X_test.copy()

    for col in te_cols:
        oof_col   = np.zeros(len(X))
        test_cols = np.zeros(len(X_test))

        for tr_idx, val_idx in kf.split(X):
            val_enc, te_enc = target_encode_oof(
                X.iloc[tr_idx], y[tr_idx],
                X.iloc[val_idx], X_test,
                col=col,
            )
            oof_col[val_idx]  = val_enc
            test_cols        += te_enc / kf.n_splits

        X[f'{col}_te']      = oof_col
        X_test[f'{col}_te'] = test_cols

    return X, X_test

X_te, X_test_te = add_target_encoding_oof(X, y, X_test, kf)
print(f'Target encoding eklendi. Yeni boyut: {X_te.shape}')
```

---

## B2 — CatBoost Native Metin (60 dk)

CatBoost, `text_features` parametresi ile Türkçe metni **direkt** işleyebilir.  
Ayrı TF-IDF pipeline'ı yazmana gerek yok; bizim için içeride yapar.

```python
import catboost as cb
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path('../datathon-2026')

# Ham train/test (mentor_feedback_text dahil)
train_raw = pd.read_csv(DATA / 'train.csv')
test_raw  = pd.read_csv(DATA / 'test_x.csv')
y         = train_raw['career_success_score'].values

# Preprocessing + FE yap ama mentor_feedback_text'i de tut
from notebooks import build_all_features  # 02 notebook'taki fonksiyon yeniden tanımlanabilir

def build_for_catboost(df):
    """FE uygula ama metin kolonunu koru."""
    from pathlib import Path
    # Aşağıdaki importlar 02_full_experiment.ipynb çalıştırıldıktan sonra bellekte mevcutsa
    # doğrudan build_all_features kullan, yoksa burada tekrar tanımla.
    return build_all_features(df)

train_cb = build_for_catboost(train_raw)
test_cb  = build_for_catboost(test_raw)

# Kolon hazırlığı
DROP_FOR_CB = ['student_id', 'career_success_score']  # mentor_feedback_text'i BIRAKMIYORUZ
X_cb      = train_cb.drop(columns=DROP_FOR_CB)
X_test_cb = test_cb.drop(columns=['student_id'])

# CatBoost için tip düzenlemeleri
CAT_COLS_CB  = ['department', 'target_role', 'hobby',
                'preferred_social_media_platform', 'university_tier']
TEXT_COLS_CB = ['mentor_feedback_text']

for c in CAT_COLS_CB:
    X_cb[c]      = X_cb[c].astype(str).fillna('NA')
    X_test_cb[c] = X_test_cb[c].astype(str).fillna('NA')

for c in TEXT_COLS_CB:
    X_cb[c]      = X_cb[c].fillna('')
    X_test_cb[c] = X_test_cb[c].fillna('')

cat_idx  = [list(X_cb.columns).index(c) for c in CAT_COLS_CB]
text_idx = [list(X_cb.columns).index(c) for c in TEXT_COLS_CB]

print(f'CatBoost feature matrix: {X_cb.shape}')
print(f'Categorical indices: {cat_idx}')
print(f'Text indices: {text_idx}')
```

CatBoost CV çalıştır:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_cb      = np.zeros(len(X_cb))
test_pred_cb = np.zeros(len(X_test_cb))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_cb)):
    tr_pool  = cb.Pool(
        X_cb.iloc[tr_idx],  y[tr_idx],
        cat_features=cat_idx, text_features=text_idx,
    )
    val_pool = cb.Pool(
        X_cb.iloc[val_idx], y[val_idx],
        cat_features=cat_idx, text_features=text_idx,
    )
    te_pool  = cb.Pool(
        X_test_cb,
        cat_features=cat_idx, text_features=text_idx,
    )

    m = cb.CatBoostRegressor(
        iterations=4000, learning_rate=0.03,
        depth=7, l2_leaf_reg=3,
        loss_function='RMSE', eval_metric='RMSE',
        random_seed=42, verbose=0,
        early_stopping_rounds=200,
        # Native text processing options
        tokenizers=[{'tokenizer_id': 'Space', 'delimiter': ' ', 'lowercasing': 'true'}],
        dictionaries=[{'dictionary_id': 'BiGram', 'max_dictionary_size': '50000',
                       'gram_order': '2'}],
        feature_calcers=[{'calcer_type': 'BoW',
                          'dictionary_id': 'BiGram'}],
    )
    m.fit(tr_pool, eval_set=val_pool, use_best_model=True)
    oof_cb[val_idx]   = m.predict(val_pool)
    test_pred_cb     += m.predict(te_pool) / kf.n_splits
    fold_mse = ((y[val_idx] - np.clip(oof_cb[val_idx], 0, 100)) ** 2).mean()
    print(f'  fold {fold+1}: MSE={fold_mse:.4f}')

oof_cb, test_pred_cb = np.clip(oof_cb, 0, 100), np.clip(test_pred_cb, 0, 100)
cb_text_cv = ((y - oof_cb) ** 2).mean()
print(f'CatBoost (native text) CV MSE: {cb_text_cv:.4f}')

np.save('../oof/exp_b_cb_text_oof.npy',  oof_cb)
np.save('../oof/exp_b_cb_text_test.npy', test_pred_cb)
```

> **Not:** CatBoost native text bazen kurulum gerektiren C++ tokenizer'a bağlıdır.  
> Hata alırsan `tokenizers`, `dictionaries`, `feature_calcers` parametrelerini sil ve tekrar çalıştır — CatBoost yine de iyi sonuç verir, sadece text native olmaz.

---

## B3 — Optuna: LightGBM Hiper-Parametre Arama (60 dk)

100 trial yaklaşık 40–60 dk sürer. Gece çalıştır.

```python
import optuna
import lightgbm as lgb
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

# X_te: target encoding eklenmiş versiyon (B1'den)
X_opt, X_test_opt = X_te, X_test_te

# Kategorikleri hazırla (run_lgbm içinde de yapılıyor ama burada önceden yapalım)
CAT_COLS = ['department', 'target_role', 'hobby', 'preferred_social_media_platform']

def lgbm_cv_score(params, X, y, X_test, kf, cat_cols=CAT_COLS):
    X_c, _ = X.copy(), X_test.copy()
    for c in [c for c in cat_cols if c in X_c.columns]:
        X_c[c] = X_c[c].astype('category')

    oof = np.zeros(len(X_c))
    for tr_idx, val_idx in kf.split(X_c):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_c.iloc[tr_idx], y[tr_idx],
            eval_set=[(X_c.iloc[val_idx], y[val_idx])],
            eval_metric='mse',
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = m.predict(X_c.iloc[val_idx])
    return ((y - np.clip(oof, 0, 100)) ** 2).mean()


def objective(trial):
    params = dict(
        objective='regression', metric='mse', verbose=-1,
        random_state=42,
        n_estimators=trial.suggest_int('n_estimators', 1000, 6000),
        learning_rate=trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        num_leaves=trial.suggest_int('num_leaves', 31, 255),
        max_depth=trial.suggest_int('max_depth', 4, 12),
        min_child_samples=trial.suggest_int('min_child_samples', 5, 60),
        subsample=trial.suggest_float('subsample', 0.5, 1.0),
        subsample_freq=1,
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.4, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        min_split_gain=trial.suggest_float('min_split_gain', 0.0, 1.0),
    )
    return lgbm_cv_score(params, X_opt, y, X_test_opt, kf)


study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f'\nBest CV MSE : {study.best_value:.4f}')
print(f'Best params : {study.best_params}')
```

En iyi parametrelerle tam OOF çalıştır:

```python
best_params = study.best_params

# run_lgbm fonksiyonu 02_full_experiment.ipynb'den veya aşağıdaki mini versiyonla
lgbm_opt_oof      = np.zeros(len(X_opt))
lgbm_opt_test     = np.zeros(len(X_test_opt))
X_c, X_test_c = X_opt.copy(), X_test_opt.copy()
for c in [c for c in CAT_COLS if c in X_c.columns]:
    X_c[c]      = X_c[c].astype('category')
    X_test_c[c] = pd.Categorical(X_test_c[c], categories=X_c[c].cat.categories)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_c)):
    m = lgb.LGBMRegressor(**best_params)
    m.fit(X_c.iloc[tr_idx], y[tr_idx],
          eval_set=[(X_c.iloc[val_idx], y[val_idx])],
          eval_metric='mse',
          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    lgbm_opt_oof[val_idx] = m.predict(X_c.iloc[val_idx])
    lgbm_opt_test        += m.predict(X_test_c) / kf.n_splits
    print(f'  fold {fold+1}: MSE={((y[val_idx]-np.clip(lgbm_opt_oof[val_idx],0,100))**2).mean():.4f}')

lgbm_opt_oof, lgbm_opt_test = np.clip(lgbm_opt_oof, 0, 100), np.clip(lgbm_opt_test, 0, 100)
lgbm_opt_cv = ((y - lgbm_opt_oof) ** 2).mean()
print(f'Optuna LGBM CV MSE: {lgbm_opt_cv:.4f}')

np.save('../oof/exp_b_lgbm_opt_oof.npy',  lgbm_opt_oof)
np.save('../oof/exp_b_lgbm_opt_test.npy', lgbm_opt_test)
```

---

## B4 — Seed Averaging (10 dk, hızlı kazanım)

Aynı model farklı seed'lerle eğitilince tahminlerin ortalaması varyansı azaltır:

```python
# Optuna best_params ya da LGBM_PARAMS kullan
seeds = [42, 123, 2024, 7, 999]

multi_seed_oof  = np.zeros(len(X_c))
multi_seed_test = np.zeros(len(X_test_c))

for seed in seeds:
    seed_oof = np.zeros(len(X_c)); seed_test = np.zeros(len(X_test_c))
    params_s = {**best_params, 'random_state': seed}
    for tr_idx, val_idx in kf.split(X_c):
        m = lgb.LGBMRegressor(**params_s)
        m.fit(X_c.iloc[tr_idx], y[tr_idx],
              eval_set=[(X_c.iloc[val_idx], y[val_idx])],
              eval_metric='mse',
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        seed_oof[val_idx]  = m.predict(X_c.iloc[val_idx])
        seed_test         += m.predict(X_test_c) / kf.n_splits
    multi_seed_oof  += seed_oof  / len(seeds)
    multi_seed_test += seed_test / len(seeds)
    print(f'  seed={seed}: OOF MSE={((y-np.clip(seed_oof,0,100))**2).mean():.4f}')

multi_seed_oof, multi_seed_test = np.clip(multi_seed_oof, 0, 100), np.clip(multi_seed_test, 0, 100)
seed_avg_cv = ((y - multi_seed_oof) ** 2).mean()
print(f'Seed-averaged LGBM CV MSE: {seed_avg_cv:.4f}')

np.save('../oof/exp_b_seedavg_oof.npy',  multi_seed_oof)
np.save('../oof/exp_b_seedavg_test.npy', multi_seed_test)
```

---

## B5 — Submission

```python
import pandas as pd
from pathlib import Path
SUBDIR = Path('../submissions')
test_ids = pd.read_csv(Path('../datathon-2026') / 'test_x.csv')['student_id']

# En iyi EXP-B sonucunu kaydet
best_b = min([
    (lgbm_opt_cv, lgbm_opt_test, 'lgbm_optuna'),
    (cb_text_cv,  test_pred_cb,  'cb_text'),
    (seed_avg_cv, multi_seed_test, 'seed_avg'),
], key=lambda x: x[0])

sub = pd.DataFrame({'student_id': test_ids, 'career_success_score': best_b[1]})
fname = f'sub_B_{best_b[2]}_cv{best_b[0]:.2f}.csv'
sub.to_csv(SUBDIR / fname, index=False)
print(f'Kaydedildi: {fname}  |  CV MSE: {best_b[0]:.4f}')
```

---

## Rapor

| Dosya | İçerik |
|---|---|
| `oof/exp_b_cb_text_oof.npy` | CatBoost native text OOF |
| `oof/exp_b_cb_text_test.npy` | CatBoost native text test pred |
| `oof/exp_b_lgbm_opt_oof.npy` | Optuna-tuned LGBM OOF |
| `oof/exp_b_lgbm_opt_test.npy` | Optuna-tuned LGBM test pred |
| `oof/exp_b_seedavg_oof.npy` | 5-seed averaged LGBM OOF |
| `oof/exp_b_seedavg_test.npy` | 5-seed averaged LGBM test pred |

**CV skorlarını PROGRESS.md'ye yaz** ve EXP-C için OOF dosyalarını hazır tut.

---

## Beklenen Kazanım

| Adım | Beklenen CV MSE |
|---|---|
| Baseline (02 notebook LGBM) | ~82 |
| + Target encoding | ~81 |
| + CatBoost native text | ~79–80 |
| + Optuna tuning | **< 79** hedef |
| + Seed averaging | ~0.3 ek iyileşme |
