# EXP-C — Stacking Meta-Ensemble + Final Submission
**Atanan:** İkisi birlikte | **Tahmini süre:** 1 saat | **Ön koşul:** EXP-A ve EXP-B tamamlanmış olmalı

> Bu deney A ve B'nin tüm OOF tahminlerini toplar, meta-model eğitir, final 2 submission seçer.  
> Basit blend → Ridge stacking → optimal ağırlık arama sırasıyla uygulanır.

---

## Ön Koşul Kontrol

```python
from pathlib import Path
import numpy as np

OOF_DIR = Path('../oof')
expected = [
    # EXP-A
    'exp_a_tfidf_oof.npy', 'exp_a_tfidf_test.npy',
    'exp_a_sbert_oof.npy', 'exp_a_sbert_test.npy',
    'exp_a_lgbm_sbert_oof.npy', 'exp_a_lgbm_sbert_test.npy',
    # EXP-B
    'exp_b_cb_text_oof.npy', 'exp_b_cb_text_test.npy',
    'exp_b_lgbm_opt_oof.npy', 'exp_b_lgbm_opt_test.npy',
    'exp_b_seedavg_oof.npy', 'exp_b_seedavg_test.npy',
]
for f in expected:
    status = '✅' if (OOF_DIR / f).exists() else '❌ EKSİK'
    print(f'{status}  {f}')
```

Eksik dosya varsa ilgili deneyin son hücrelerini tekrar çalıştır.

---

## C1 — Tüm OOF'ları Yükle

```python
import numpy as np
import pandas as pd
from pathlib import Path

OOF_DIR = Path('../oof')
DATA    = Path('../datathon-2026')
SUBDIR  = Path('../submissions')

train = pd.read_csv(DATA / 'train.csv')
test  = pd.read_csv(DATA / 'test_x.csv')
y     = train['career_success_score'].values

def load(name):
    return np.load(OOF_DIR / name)

# ── EXP-A ────────────────────────────────────────────────────────────────────
a_tfidf_oof,  a_tfidf_test  = load('exp_a_tfidf_oof.npy'),  load('exp_a_tfidf_test.npy')
a_sbert_oof,  a_sbert_test  = load('exp_a_sbert_oof.npy'),  load('exp_a_sbert_test.npy')
a_lgbm_oof,   a_lgbm_test   = load('exp_a_lgbm_sbert_oof.npy'), load('exp_a_lgbm_sbert_test.npy')

# ── EXP-B ────────────────────────────────────────────────────────────────────
b_cb_oof,     b_cb_test     = load('exp_b_cb_text_oof.npy'),  load('exp_b_cb_text_test.npy')
b_lgbm_oof,   b_lgbm_test   = load('exp_b_lgbm_opt_oof.npy'), load('exp_b_lgbm_opt_test.npy')
b_seed_oof,   b_seed_test   = load('exp_b_seedavg_oof.npy'),  load('exp_b_seedavg_test.npy')

# ── 02_full_experiment.ipynb baseline (eğer kaydedildiyse) ───────────────────
# Yoksa 02 notebook'taki lgbm_oof, lgbm_test değerlerini de ekle
try:
    base_oof  = load('exp_base_lgbm_oof.npy')
    base_test = load('exp_base_lgbm_test.npy')
    has_base  = True
except FileNotFoundError:
    has_base  = False
    print('Baseline OOF yok — 02 notebook OOF eklenemiyor')

print('OOF yüklendi')
```

---

## C2 — Bireysel Skor Karşılaştırması

```python
models = {
    'A: TF-IDF (tuned)':   a_tfidf_oof,
    'A: SBERT Ridge':       a_sbert_oof,
    'A: LGBM + SBERT':      a_lgbm_oof,
    'B: CatBoost text':     b_cb_oof,
    'B: LGBM Optuna':       b_lgbm_oof,
    'B: LGBM seed-avg':     b_seed_oof,
}
if has_base:
    models['02: LGBM baseline'] = base_oof

print(f'{"Model":<25s}  CV MSE')
print('-' * 40)
for name, oof in sorted(models.items(), key=lambda x: ((x[1]-y)**2).mean()):
    mse = ((y - np.clip(x[1], 0, 100)) ** 2).mean() if isinstance(x, tuple) else \
          ((y - np.clip(oof, 0, 100)) ** 2).mean()
    print(f'{name:<25s}  {mse:.4f}')

# Düzeltelim:
print(f'\n{"Model":<25s}  CV MSE')
print('-' * 40)
for name, oof in sorted(models.items(), key=lambda kv: ((y - np.clip(kv[1], 0, 100))**2).mean()):
    mse = ((y - np.clip(oof, 0, 100)) ** 2).mean()
    print(f'{name:<25s}  {mse:.4f}')
```

---

## C3 — Optimal Blend (SLSQP)

```python
from scipy.optimize import minimize

# Sadece en iyi N modeli blend'e al (çok düşük performanslıları çıkar)
THRESHOLD_MSE = 85.0  # Bu değerin üzerindeki modelleri hariç tut

selected_oofs  = []
selected_tests = []
selected_names = []

test_preds_map = {
    'A: TF-IDF (tuned)':  a_tfidf_test,
    'A: SBERT Ridge':      a_sbert_test,
    'A: LGBM + SBERT':     a_lgbm_test,
    'B: CatBoost text':    b_cb_test,
    'B: LGBM Optuna':      b_lgbm_test,
    'B: LGBM seed-avg':    b_seed_test,
}
if has_base:
    test_preds_map['02: LGBM baseline'] = base_test

for name, oof in models.items():
    mse = ((y - np.clip(oof, 0, 100)) ** 2).mean()
    if mse < THRESHOLD_MSE:
        selected_oofs.append(oof)
        selected_tests.append(test_preds_map[name])
        selected_names.append(name)
        print(f'[IN]  {name:<25s}  MSE={mse:.4f}')
    else:
        print(f'[OUT] {name:<25s}  MSE={mse:.4f}  (threshold exceeded)')

print(f'\n{len(selected_names)} model blend\'e alındı')


def optimize_blend(oofs, tests, y, names):
    stacked = np.column_stack(oofs)

    def loss(w):
        return ((y - np.clip((stacked * w).sum(axis=1), 0, 100)) ** 2).mean()

    n = len(oofs)
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    bounds = [(0, 1)] * n

    best = None
    # Birden fazla başlangıç noktası → yerel minimuma takılmamak için
    for w0 in [np.ones(n) / n] + [np.eye(n)[i] for i in range(n)]:
        res = minimize(loss, w0, method='SLSQP', constraints=constraints, bounds=bounds,
                       options={'ftol': 1e-10, 'maxiter': 5000})
        if best is None or res.fun < best.fun:
            best = res

    w = best.x
    print('\nOptimal blend ağırlıkları:')
    for name, wi in sorted(zip(names, w), key=lambda x: -x[1]):
        if wi > 0.001:
            print(f'  {name:<25s}: {wi:.4f}')
    print(f'Blend CV MSE: {best.fun:.4f}')

    test_blend = np.clip((np.column_stack(tests) * w).sum(axis=1), 0, 100)
    return w, best.fun, test_blend


weights, blend_cv, blend_test = optimize_blend(
    selected_oofs, selected_tests, y, selected_names
)
```

---

## C4 — Ridge Stacking (Meta-Model)

Blend'den bir adım daha: OOF tahminlerini meta-feature olarak kullanıp Ridge eğit.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

meta_kf = KFold(n_splits=5, shuffle=True, random_state=42)

stacked_oof  = np.column_stack(selected_oofs)   # (10000, n_models)
stacked_test = np.column_stack(selected_tests)   # (10000, n_models)

meta_oof  = np.zeros(len(y))
meta_test = np.zeros(len(stacked_test))

for tr_idx, val_idx in meta_kf.split(stacked_oof):
    # RidgeCV: alpha'yı da CV içinde arar
    meta = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=3)
    meta.fit(stacked_oof[tr_idx], y[tr_idx])
    meta_oof[val_idx]  = meta.predict(stacked_oof[val_idx])
    meta_test         += meta.predict(stacked_test) / meta_kf.n_splits

meta_oof, meta_test = np.clip(meta_oof, 0, 100), np.clip(meta_test, 0, 100)
stack_cv = ((y - meta_oof) ** 2).mean()
print(f'Ridge Stacking CV MSE: {stack_cv:.4f}')
print(f'vs. Simple Blend:       {blend_cv:.4f}')
```

---

## C5 — Final 2 Submission Seçimi

Kural: iki submission **birbirinden farklı** olmalı — aynı şeyin iki versiyonunu seçme.

```python
# Tüm aday sonuçları topla
candidates = {
    f'blend_cv{blend_cv:.2f}':  (blend_cv,  blend_test),
    f'stack_cv{stack_cv:.2f}':  (stack_cv,  meta_test),
    f'A_lgbm_sbert':            (((y-a_lgbm_oof)**2).mean(), a_lgbm_test),
    f'B_lgbm_optuna':           (((y-b_lgbm_oof)**2).mean(), b_lgbm_test),
    f'B_seedavg':               (((y-b_seed_oof)**2).mean(), b_seed_test),
}

print('Tüm adaylar (CV MSE sırasıyla):')
for name, (mse, _) in sorted(candidates.items(), key=lambda x: x[1][0]):
    print(f'  {name:<30s}: {mse:.4f}')

# Submission 1: En düşük CV MSE
sub1_name, (sub1_cv, sub1_preds) = min(candidates.items(), key=lambda x: x[1][0])
# Submission 2: En düşük CV MSE ama sub1 ile FARKLI MODEL türü
# (örn: sub1 ensemble ise sub2 tek model; sub1 stack ise sub2 blend olmasın)
# Manuel seç:
print('\nSub-1 seçildi:', sub1_name, f'(CV={sub1_cv:.4f})')
print('Sub-2 için yukarıdaki listeden el ile farklı bir model seç.')
```

Submission dosyalarını oluştur:

```python
def save_sub(preds, fname):
    sub = pd.DataFrame({
        'student_id': test['student_id'],
        'career_success_score': preds,
    })
    assert sub['career_success_score'].notna().all()
    assert len(sub) == 10000
    sub.to_csv(SUBDIR / fname, index=False)
    mn, mx = preds.min(), preds.max()
    print(f'Saved: {fname}  |  min={mn:.2f} max={mx:.2f} mean={preds.mean():.2f}')


# Sub 1 — en iyi CV
save_sub(sub1_preds, f'FINAL_sub1_{sub1_name}.csv')

# Sub 2 — el ile seç (örnek: stacking ya da seed-avg)
SUB2_NAME  = 'B_seedavg'   # bunu değiştir
SUB2_PREDS = candidates[SUB2_NAME][1]
save_sub(SUB2_PREDS, f'FINAL_sub2_{SUB2_NAME}.csv')

print(f'\nFinal Özet:')
print(f'  Sub 1: {sub1_name:<30s}  CV MSE = {sub1_cv:.4f}')
print(f'  Sub 2: {SUB2_NAME:<30s}  CV MSE = {candidates[SUB2_NAME][0]:.4f}')
print(f'  Baseline:                              CV MSE = 83.83')
```

---

## C6 — PROGRESS.md Güncelle

Sonuçları kayıt defterine yaz:

```
## 📊 Submission kayıt defteri
| # | Tarih | Açıklama | Lokal CV | Public LB | Final aday? |
|---|-------|----------|----------|-----------|-------------|
| 1 | 10 Haz | LightGBM baseline (01)         | 83.83 | —    | hayır |
| 2 | 13 Haz | Ensemble (02 notebook)          | XX.XX | —    | evet  |
| 3 | 14 Haz | FINAL Sub1: <isim>              | XX.XX | XX.XX | ✅   |
| 4 | 14 Haz | FINAL Sub2: <isim>              | XX.XX | XX.XX | ✅   |
```

---

## Seçim Rehberi — Hangi 2'yi Submitle?

| Durum | Sub 1 | Sub 2 |
|---|---|---|
| Stack ≥ Blend | En iyi blend | Stack |
| Blend ≥ Stack | Stack | En iyi tek model |
| İkisi yakınsa | En düşük CV | En farklı model türü |

**Altın kural:** Sub 2 hiçbir zaman Sub 1'in kopyası olmasın.  
Blend + Stack birbirine çok yakınsa, Sub 2 olarak en iyi tekil modeli (A_lgbm_sbert veya B_seedavg) al.

---

## Beklenen Nihai Skor

| Yöntem | Beklenen CV MSE |
|---|---|
| En iyi tekil model (EXP-A/B) | ~78–80 |
| Simple blend | ~76–78 |
| Ridge stacking | **~75–77** |

Bu seviye private leaderboard'da rekabetçi olmalı.
