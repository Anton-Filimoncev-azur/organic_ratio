# organic_ratio

Pipeline для оценки доли органических установок (`organic_share`) на горизонте 7 дней когорты и разложения органики на halo-вклад paid-каналов через MMM.

В проекте две независимые модели:

| Модель | Гранулярность | Target | Что даёт |
|---|---|---|---|
| **PyMC hierarchical** (`run_train.py`) | cohort: `platform × country × install_date` | `organic_share` через `Binomial(total_installs, p)` | предсказание доли органики по фичам когорты (retention, sessions, IAP, ads, costs) |
| **Halo MMM** (`run_mmm_*.py`) | panel: `platform × country × install_date` (7d-cadence) | `organic_installs` (count) | разложение органики на вклад каждого paid-канала |

Baseline (`run_baseline.py`) — взвешенный Ridge на `logit(organic_share)`, нужен как точка отсчёта.

---

## 1. Установка

```bash
pip install -r requirements.txt
```

Python 3.10+. Для ускорения NUTS работает JAX/CUDA через `numpyro` (уже стоит дефолтом в [parameters.yml](conf/base/parameters.yml#L57)). На CPU параллелизм включается раскомментированием `XLA_FLAGS` в [run_train.py:24-27](run_train.py#L24-L27).

Создать `.env` в корне:
- `GIT_PAT` — для синка кода в jupyter (см. первая ячейка [run.ipynb](run.ipynb))
- креды для S3 (читаются через `dotenv`, используются `s3fs`)

## 2. Конфигурация

Единственный источник параметров — [conf/base/](conf/base/):
- [globals.yml](conf/base/globals.yml) — S3 bucket, пути, `project.name = kclash`
- [data.yml](conf/base/data.yml) — все parquet-файлы и куда они кладутся
- [parameters.yml](conf/base/parameters.yml) — параметры моделей, фичи, ключи когорт, split-даты

Загружается через `load_config()` ([src/organic_ratio/utils/config.py](src/organic_ratio/utils/config.py)) — OmegaConf с интерполяцией `${...}` и оверрайдом через `CONFIG_OVERRIDE_PATH`.

Ключевые поля в `parameters.yml`:
- `cohort.keys` — `[platform, media_source, country_code, install_date]`
- `cleaning.min_total_installs: 30` — порог фильтрации малых когорт
- `train_start_date / test_start_date / test_end_date / gap_days_split` — временной split
- `modeling.features` — whitelist фичей для PyMC/baseline (после cleaning остаются только эти + ключи + weight + target)
- `modeling.pymc` — параметры NUTS для иерархической модели
- `modeling.mmm` — параметры MMM (`top_n_channels`, `top_n_geos`, `cadence_days`, `adstock_l_max`, `saturation`, `geo_dim`, …)

## 3. Pipeline — порядок запуска

Каждый шаг читает выход предыдущего. Запускать из корня репо. Можно прогнать всё последовательно через ячейки в [run.ipynb](run.ipynb).

### Шаг 1. Ingest из S3
```bash
python run.py
```
Тянет `installs / ads / iap / devices / costs / sessions / personal` из `s3://az-jupyterhub-share-…/azur_ml_core/kclash/partition/`.
→ `data/raw/partition/*.parquet`

### Шаг 2. Per-source preprocessing
```bash
python run_preprocessing.py
```
Каждый датасет проходит через свой builder из [preprocesser_registry.py](src/organic_ratio/core/preprocessing/preprocesser_registry.py). На выходе — user-grain feature parquets.
→ `data/features/partition/*.parquet`

### Шаг 3. Cohort aggregation
```bash
python run_cohort_aggregation.py
```
Мерджит все per-source фичи по `[match_id, install_date]`, группирует по `cohort.keys`, агрегирует SUM/MEAN.
→ `data/features/cohort/cohort_level.parquet`

### Шаг 4. Target + features build
```bash
python run_target_build.py
```
Считает target (`organic_share`, `organic_installs`, `total_installs`) на грануляции `cohort.keys − media_source`, реагрегирует фичи на ту же грануляцию, джойнит, делит по датам.
→ `data/features/targets/targets.parquet`, `data/train/targets_train.parquet`, `data/test/targets_test.parquet`

### Шаг 5. Cleaning
```bash
python run_clean.py
```
Отрезает когорты с `total_installs < cleaning.min_total_installs`, оставляет только колонки из `modeling.{keep_keys, target, weight, features}`.
→ `data/train/targets_train_clean.parquet`, `data/test/targets_test_clean.parquet`

### Шаг 6a. Baseline (Ridge)
```bash
python run_baseline.py
```
`logit(organic_share)` → Ridge с `sample_weight = total_installs`. Numeric — median impute + StandardScaler; `platform` — OHE; `country_code` — weighted target encoding по train; `install_date` — дроп.
→ `data/predictions/baseline_{train,test}.parquet`, `data/plots/baseline_{calibration,coefficients,pe_distribution}.png`

### Шаг 6b. PyMC hierarchical
```bash
python run_train.py
```
Модель:
```
organic_installs[i] ~ Binomial(total_installs[i], p[i])
logit(p[i]) = α + X[i]·β + u_country[c] + u_platform[plat]
u_country  ~ Normal(0, σ_country),  σ_country ~ HalfNormal(1)
u_platform ~ Normal(0, σ_platform), σ_platform ~ HalfNormal(1)
```
Non-centered параметризация для country/platform — см. [pymc_model.py:50-60](src/organic_ratio/core/modeling/pymc_model.py#L50-L60). Binomial-likelihood сам учитывает вес.

Параметры сэмплирования — в `modeling.pymc` ([parameters.yml:64-72](conf/base/parameters.yml#L64-L72)).
→ `data/models/pymc/{trace.nc, prep.pkl}`, `data/predictions/pymc_{train,test}.parquet`, `data/plots/pymc_{calibration,beta}.png`

### Шаг 7. MMM data
```bash
python run_mmm_data.py
```
Собирает panel `platform × country × install_date` (cadence 7 дней):
- target: `organic_installs`
- channels: `spend_<top_n_channels>` + `spend_other_paid`
- controls: `dow_0..dow_6`
- dim: `geo = <platform>_<country>` (топ-N по installs)

→ `data/features/mmm/mmm_panel{,_train,_test}.parquet`

### Шаг 8. MMM training
```bash
python run_mmm_train.py
```
`pymc-marketing.MMM` с `GeometricAdstock(l_max=4)` + `LogisticSaturation`, `dims=("geo",)`. Параметры — `modeling.mmm` в [parameters.yml:43-62](conf/base/parameters.yml#L43-L62).

⚠ Скрипт сохраняет trace **сразу после fit**, до тяжёлых post-fit операций — флаги `mmm.compute_predict` и `mmm.save_plots` дефолтом `false`, чтобы первый прогон не висел.
→ `data/models/mmm/mmm.nc`

### Шаг 9. MMM analyze
```bash
python run_mmm_analyze.py
```
Загружает trace через ArviZ (без pymc-marketing heavy methods), считает convergence overview, posterior mean ± 94% HDI по `beta_channel`/`saturation_*`.
→ `data/predictions/mmm_summary.csv`, `data/plots/mmm_channel_coefs.png`

### Шаг 10. MMM out-of-sample eval
```bash
python run_mmm_eval.py
```
Загружает MMM через `pymc_marketing.mmm.multidimensional.MMM.load`, считает posterior predictive на test (`include_last_observations=True` для adstock warmup), выводит test метрики.
→ `data/predictions/mmm_test.parquet`, `data/plots/mmm_test_pred_vs_actual.png`

### Шаг 11. MMM attribution
```bash
python run_mmm_attribution.py
```
Берёт `channel_contribution[date, geo, channel]` из trace, калибрует per-geo scale через сопоставление с observed train, считает halo per channel в оригинальных install-units, склеивает со spend для ROAS.
→ `data/predictions/mmm_attribution.parquet`, `mmm_attribution_summary.csv`, `data/plots/mmm_halo_{per_channel,over_time}.png`

---

## 4. Что уже протестировано

Цифры — последний прогон, полный лог в [run.ipynb](run.ipynb).

### Baseline (Ridge)
- Train 14 869 когорт, test 2 069. Split: train ≤ 2025-05-01, test 2025-05-01 .. 2025-06-01, gap 14 дней.
- **Test weighted R² ≈ 0.53**, WMAPE ≈ 28%, within ±20%: 37.8%, within ±50%: 77.1%.
- Топ-фичи: `country_te`, `platform`, `max_gap_day7`, `ret_2/3`, `log1p_ads_cum_learn`, `d1_share_iap`.

### PyMC hierarchical
- 39 фичей × 14 869 obs, 97 стран, 2 платформы. NUTS via `numpyro`, ~16 мин на GPU (1500 tune + 500 draws × 2 chains).
- **Test weighted R² ≈ 0.56** (чуть лучше baseline), WMAPE ≈ 29%.
- ⚠ **Проблемы сходимости** при текущей конфигурации:
  - `R̂(sigma_country) = 1.52`, `R̂(alpha) = 1.16`, `ess_bulk(sigma_country) = 4`
  - Норматив для аналитики — `R̂ < 1.01`, `ess > 400`.
- Топ-β: `log1p_ads_cum_learn (−0.22)`, `log1p_iap_cum_learn (+0.17)`, `cvr (+0.13)`, `ret_2 (+0.11)`, `max_gap_day7 (−0.10)`.

### MMM
- Panel: 100 топ-geos × 26 недель. 11 channels (top-10 spend + `other_paid`).
- Train: 2600 строк (2024-11-01 → 2025-04-25). Test: 500 строк (2025-05-02 → 2025-05-30).
- NUTS `numpyro`, ~7 мин на GPU (2500 tune + 1000 draws × 4 chains). **13 divergences** — терпимо, но стоит поднять `target_accept` или скорректировать priors.
- **In-sample convergence хорошая**: `R̂ > 1.01: 0%`, все ESS > 200. `total_media_contribution_original_scale ≈ 637k` organic installs.
- 🚨 **Out-of-sample test сломан**: `R²_w = −80`, `WMAPE = 561%`. Причина — per-geo scale factor получается практически одинаковым (~12 900) для всех geo, то есть калибровка `actual_train / pred_scaled_train` не разделяет geos. См. п. 5.1 ниже.
- **In-sample attribution даёт осмысленные числа** — `run_mmm_attribution.py` отрабатывает, `mmm_halo_per_channel.png` и `mmm_halo_over_time.png` визуализируют halo по каналам.

### Что точно работает end-to-end
- Полный S3 → preprocessing → cohort → train/test → baseline → плоты. Воспроизводимо.
- PyMC сэмплируется, метрики и β-summary читаются. Нужно довести сходимость.
- MMM обучается, in-sample attribution выгружается.

### Что не работает / требует ревью
- **MMM out-of-sample** — см. ниже.
- **PyMC sampling diagnostics** — красные на `sigma_country` и `alpha`.
- Фичи, выкинутые из-за outliers / багов препроцессинга — см. комментарии в [parameters.yml:99-100](conf/base/parameters.yml#L99-L100) и `:117`: `acceleration_iap`, `avg_check_d1_3_iap`, `avg_check_d4_7_iap`, `time_between_iap_2_3`, `zero_session_days`.

---

## 5. Что нужно реализовать для конечной задачи

Конечная задача — достоверная per-channel halo attribution + прогноз `organic_share` на свежие когорты. Открытые блоки:

### 5.1. Починить out-of-sample MMM evaluation (P0)
В [run_mmm_eval.py](run_mmm_eval.py) `mmm.sample_posterior_predictive` возвращает значения в нормированном пространстве, а калибровка через `actual_train / pred_scaled_train` даёт практически одинаковый scale для всех geo (~12 900) → формула вырождается.

Варианты:
- использовать встроенный `mmm.predict()`, который сам де-нормализует
- разобраться с `target_transformer` / `MaxAbsScaler` внутри `pymc-marketing` и применить inverse_transform явно
- ручной forward pass через trace (`adstock_alpha`, `saturation_lam`, `saturation_beta`, `intercept_contribution`) — у [run_mmm_attribution.py](run_mmm_attribution.py) уже есть похожая логика, её можно переиспользовать

Definition of done: `R²_w > 0` на test, WMAPE сопоставима с baseline.

### 5.2. Довести сходимость PyMC модели (P0)
- Поднять `chains: 2 → 4`, `draws: 500 → 1000` в `modeling.pymc`. На GPU `chain_method: vectorized` даст ~30 мин.
- Если `sigma_country` всё ещё застревает — посмотреть, нет ли стран с 1–2 наблюдениями, объединить в `country_other`.
- Добавить prior-predictive check в [run_train.py](run_train.py) — сейчас его нет.

### 5.3. Валидация MMM attribution per-channel (P0)
- `mmm_attribution_summary.csv` уже выгружается, но без sanity-проверок. Добавить health-check'и: `halo_total` ≤ разумного множителя от `spend_total`, `adstock_alpha ∈ [0,1]`, saturation монотонна — встроить в [run_mmm_analyze.py](run_mmm_analyze.py).
- Сделать ROAS-таблицу с per-channel CI (94% HDI), а не только posterior mean.

### 5.4. Объединение двух моделей (P1)
PyMC даёт `predicted organic_share`. MMM даёт `halo_organic` per (date, geo, channel). Финальный артефакт — разделить **true organic** vs **paid-induced organic**:
```
true_organic[g, t] = organic_share[g, t] × total_installs[g, t] − Σ_channel halo_organic[g, t, channel]
```
Скрипта `run_combine.py` (или ноутбука) сейчас нет — это и есть основной бизнес-deliverable.

### 5.5. Восстановить выкинутые фичи (P1)
Починить препроцессинг в [src/organic_ratio/core/preprocessing/iap.py](src/organic_ratio/core/preprocessing/iap.py) и [.../sessions.py](src/organic_ratio/core/preprocessing/sessions.py), вернуть в `modeling.features`:
- `acceleration_iap`, `avg_check_d1_3_iap`, `avg_check_d4_7_iap`, `time_between_iap_2_3` — экстремальные outliers
- `zero_session_days` — buggy values

### 5.6. Per-cohort PE breakdown (P1)
Сейчас PE-buckets выводятся только взвешенными по installs суммарно. Разрез по `country × platform` (топ-10 наихудших) нужно добавить в [run_baseline.py](run_baseline.py) и [run_train.py](run_train.py).

### 5.7. Sweep-эксперименты (P1)
В [run.ipynb](run.ipynb) последняя ячейка — заготовка под `CONFIG_OVERRIDE_PATH`. Создать `conf/batch_training/sweep.yml` с разными `top_n_geos`, `adstock_l_max`, `saturation`, `cadence_days` и автоматическим сравнением метрик.

---

## 6. Карта репозитория

- [src/organic_ratio/core/loaders/](src/organic_ratio/core/loaders/) — S3 IO, materialize
- [src/organic_ratio/core/preprocessing/](src/organic_ratio/core/preprocessing/) — per-source feature builders, реестр [preprocesser_registry.py](src/organic_ratio/core/preprocessing/preprocesser_registry.py)
- [src/organic_ratio/core/cohort/](src/organic_ratio/core/cohort/) — merge, aggregator (SUM/MEAN policy), target build, clean
- [src/organic_ratio/core/modeling/](src/organic_ratio/core/modeling/) — baseline, pymc_model, mmm_data, mmm_model, metrics, preprocess (StandardScaler + category encoders)
- [src/organic_ratio/utils/config.py](src/organic_ratio/utils/config.py) — OmegaConf loader
- [conf/base/](conf/base/) — все параметры

Артефакты на диске:
- `data/raw/partition/` — сырьё с S3
- `data/features/partition/` — user-grain features
- `data/features/cohort/cohort_level.parquet` — cohort-grain merged
- `data/features/targets/` + `data/train/` + `data/test/` — model-ready
- `data/features/mmm/` — MMM panels (full / train / test)
- `data/models/{pymc,mmm}/` — trace.nc + prep.pkl
- `data/predictions/` — все predictions + attribution
- `data/plots/` — все PNG

---

## 7. Нюансы

- Git remote: `Anton-Filimoncev-azur/organic_ratio`, ветка `main`. На jupyter-инстансе (jovyan) синк через `GIT_PAT` — см. первая ячейка [run.ipynb](run.ipynb).
- Тяжёлое IO — `polars`. Моделинг (PyMC / sklearn) — `pandas` / `numpy`.
- `chain_method: vectorized` (numpyro) работает на 1 GPU. Для CPU-multi-chain — `parallel` + `XLA_FLAGS=--xla_force_host_platform_device_count=N` (см. закомментированный блок в [run_train.py:24-27](run_train.py#L24-L27)).
- В [parameters.yml](conf/base/parameters.yml) есть две секции фичей: `modeling.features` — реально используется PyMC/baseline (whitelist после cleaning); `numerical.columns` / `categorical.columns` / `sequence.columns` — историческая разметка user-grain, кодом не читается, можно почистить.
