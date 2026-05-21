# organic_ratio — документация для передачи проекта

## 0. Что делает проект

Конечная бизнес-задача: **посчитать долю органических установок (organic share) на горизонте первых 7 дней когорты** и оценить, какая часть «органики» на самом деле является **halo-эффектом от paid-каналов** (то есть органикой только по атрибуции AppsFlyer, но порождённой платными кампаниями).

В проекте реализованы **две параллельные модели**, каждая со своим pipeline:

| Модель | Гранулярность | Target | Назначение |
|---|---|---|---|
| **PyMC hierarchical** (`run_train.py`) | cohort: `platform × country × install_date` | `organic_share ∈ [0,1]` (Binomial(total_installs, p)) | предсказание доли органики по фичам когорты (retention, sessions, IAP, ads, costs) |
| **Halo MMM** (`run_mmm_*.py`) | panel: `platform × country × install_date` (7d-cadence) | `organic_installs` (count) | разложение organic'а на вклад каждого paid-канала → bottom-up halo attribution |

Baseline (`run_baseline.py`) — взвешенный Ridge на `logit(organic_share)`, нужен как точка отсчёта.

---

## 1. Как развернуть и запустить

### Окружение
- Python 3.10+
- `pip install -r requirements.txt`
- `.env` в корне:
  - `GIT_PAT` — для синка из jupyter (см. первая ячейка [run.ipynb](run.ipynb))
  - креды S3 (используются `s3fs` через `dotenv`)
- GPU не обязателен. Для ускорения NUTS — JAX/CUDA: PyMC сэмплер `numpyro` в [parameters.yml](conf/base/parameters.yml#L57) уже выставлен по умолчанию. На CPU параллелизм работает через раскомментирование `XLA_FLAGS` в [run_train.py:24-27](run_train.py#L24-L27).

### Конфигурация
Один источник правды — [conf/base/](conf/base/):
- [globals.yml](conf/base/globals.yml) — пути, S3 bucket, `project.name = kclash`
- [data.yml](conf/base/data.yml) — все парке-файлы и куда их класть
- [parameters.yml](conf/base/parameters.yml) — **все** параметры моделей, фичи, ключи когорт, split-даты

Загружается через [src/organic_ratio/utils/config.py](src/organic_ratio/utils/config.py) (`load_config()`) — OmegaConf, поддерживает интерполяцию `${...}` и оверрайды через переменную окружения `CONFIG_OVERRIDE_PATH` (см. закомментированная sweep-секция в конце [run.ipynb](run.ipynb)).

### Pipeline (порядок запуска)

Запускать строго по порядку — каждый шаг читает выход предыдущего. Команды из корня репо:

| # | Скрипт | Что делает | Input | Output |
|---|---|---|---|---|
| 1 | [run.py](run.py) | S3 → локальный `data/raw/partition/*.parquet` | `s3://az-jupyterhub-share-…/azur_ml_core/kclash/partition/*.parquet` | `data/raw/partition/{installs,ads,iap,devices,costs,sessions,personal}.parquet` |
| 2 | [run_preprocessing.py](run_preprocessing.py) | per-source user-grain фичи (см. [src/organic_ratio/core/preprocessing/](src/organic_ratio/core/preprocessing/)) | `data/raw/partition/*` | `data/features/partition/*.parquet` |
| 3 | [run_cohort_aggregation.py](run_cohort_aggregation.py) | merge user-grain → group by `cohort.keys` | `data/features/partition/*` | `data/features/cohort/cohort_level.parquet` |
| 4 | [run_target_build.py](run_target_build.py) | target (`organic_share`) на грануляции `cohort.keys − media_source` + фичи на той же грануляции + split по датам | installs feature + все feature parquets | `data/features/targets/targets.parquet`, `data/train/targets_train.parquet`, `data/test/targets_test.parquet` |
| 5 | [run_clean.py](run_clean.py) | фильтр когорт `total_installs < 30` (cleaning.min_total_installs) и whitelist колонок из `modeling.features` | step 4 | `data/train/targets_train_clean.parquet`, `data/test/targets_test_clean.parquet` |
| 6a | [run_baseline.py](run_baseline.py) | взвешенный Ridge baseline | step 5 | `data/predictions/baseline_{train,test}.parquet`, plots |
| 6b | [run_train.py](run_train.py) | hierarchical PyMC | step 5 | `data/models/pymc/{trace.nc, prep.pkl}`, `data/predictions/pymc_{train,test}.parquet`, plots |
| 7 | [run_mmm_data.py](run_mmm_data.py) | MMM-панель: `platform × country × date(7d)`, `spend_<top10>` + `spend_other_paid`, dow-controls | installs feature + raw costs | `data/features/mmm/mmm_panel{,_train,_test}.parquet` |
| 8 | [run_mmm_train.py](run_mmm_train.py) | `pymc-marketing.MMM` (GeometricAdstock + LogisticSaturation, `dims=("geo",)`) | mmm_panel_train | `data/models/mmm/mmm.nc` |
| 9 | [run_mmm_analyze.py](run_mmm_analyze.py) | convergence + per-channel coefs из trace (без heavy pymc-marketing методов) | `mmm.nc` | `data/predictions/mmm_summary.csv`, `data/plots/mmm_channel_coefs.png` |
| 10 | [run_mmm_eval.py](run_mmm_eval.py) | out-of-sample posterior predictive на test | `mmm.nc` + mmm_panel_test | `data/predictions/mmm_test.parquet` |
| 11 | [run_mmm_attribution.py](run_mmm_attribution.py) | per-channel halo в оригинальных install-units | `mmm.nc` + train panel | `mmm_attribution.parquet`, `mmm_attribution_summary.csv`, plots |

Для удобства все шаги собраны в виде ячеек в [run.ipynb](run.ipynb) — секции 1..14.

---

## 2. Что уже протестировано (текущие результаты)

Все цифры ниже — последний прогон, см. лог в [run.ipynb](run.ipynb).

### 2.1 Baseline (Ridge, logit-target)
- Train: 14 869 cohorts, Test: 2 069. Split по `install_date`: train ≤ 2025-05-01, test 2025-05-01 .. 2025-06-01, gap 14 дней.
- **Test weighted R² ≈ 0.53**, WMAPE ≈ 28%
- within ±20%: **37.8%** когорт; within ±50%: 77.1%
- Топ-фичи: `country_te` (target encoding страны), `platform`, `max_gap_day7`, `ret_2/3`, `log1p_ads_cum_learn`, `d1_share_iap`. Полный список — в выводе baseline.

### 2.2 PyMC hierarchical
- 39 фичей × 14 869 obs, 97 стран, 2 платформы. NUTS via `numpyro`, 1500 tune + 500 draws × 2 chains, ~16 мин на GPU.
- **Test weighted R² ≈ 0.56** (чуть лучше baseline), WMAPE ≈ 29%.
- ⚠ **Проблемы сходимости** на текущей конфигурации:
  - `R̂(sigma_country) = 1.52`, `R̂(alpha) = 1.16`, `ess_bulk(sigma_country) = 4` — для нормального anal-grade результата нужно `R̂ < 1.01`, `ess > 400`.
  - Скорее всего недохватает `chains` и `draws`; non-centered параметризация для country уже включена ([pymc_model.py:50-54](src/organic_ratio/core/modeling/pymc_model.py#L50-L54)).
- Топ-β (posterior mean): `log1p_ads_cum_learn (−0.22)`, `log1p_iap_cum_learn (+0.17)`, `cvr (+0.13)`, `ret_2 (+0.11)`, `max_gap_day7 (−0.10)`.

### 2.3 MMM (halo attribution)
- Панель: 100 топ-geos × 26 недель (cadence 7d). 11 channels (`spend_facebook ads`, `spend_applovin_int`, …, `spend_other_paid`).
- Train: 2600 строк (2024-11-01 → 2025-04-25). Test: 500 строк (2025-05-02 → 2025-05-30).
- NUTS `numpyro`, 2500 tune + 1000 draws × 4 chains ≈ 7 мин на GPU. **13 divergences** — терпимо, но стоит поднять `target_accept`.
- **In-sample convergence хорошая** (`R̂ > 1.01: 0%`, все ESS > 200).
- 🚨 **Out-of-sample test catastrophic** (см. `run_mmm_eval.py` лог): `R²_w = −80`, `WMAPE = 561%`. Причина — per-geo scale factor получается одинаковый (~12 900) для всех geo, то есть `predict()` от `pymc-marketing` отдаёт в каком-то «scaled space», а калибровка через train не переносится корректно на test. Это **известная нерешённая проблема**, см. блок «Что нужно реализовать» п. 3.
- **In-sample attribution даёт осмысленные числа**: `total_media_contribution_original_scale ≈ 637k organic installs`, доля halo в total observed organic — считается в [run_mmm_attribution.py](run_mmm_attribution.py), картинки `mmm_halo_per_channel.png`, `mmm_halo_over_time.png`.

### 2.4 Что точно работает end-to-end
- Полный S3 → preprocessing → cohort → train/test → baseline. Воспроизводимо.
- PyMC модель сэмплируется, метрики читаются, но **нужно довести сходимость**.
- MMM **обучается и даёт in-sample attribution**, attribution по каналам — главный бизнес-артефакт — выгружается.

### 2.5 Что НЕ работает / надо проверить
- **MMM out-of-sample прогноз** — см. выше. Per-geo scale factor одинаковый → train-калибровка некорректна.
- **PyMC sampling** — diagnostics красные на `sigma_country` и `alpha`.
- Закомментированные в [parameters.yml:99-100](conf/base/parameters.yml#L99-L100) фичи (`acceleration_iap`, `avg_check_d1_3_iap`, `avg_check_d4_7_iap`, `time_between_iap_2_3`) дропнуты из-за экстремальных outliers — стоит починить препроцессинг, а не выкидывать.
- `zero_session_days` помечена как «buggy» ([parameters.yml:117](conf/base/parameters.yml#L117)) — посмотреть [src/organic_ratio/core/preprocessing/sessions.py](src/organic_ratio/core/preprocessing/sessions.py).

---

## 3. Что нужно реализовать для конечной задачи

Конечная задача = **достоверная per-channel halo attribution** + **prediction organic_share на свежие когорты**. Открытые блоки:

### Приоритет P0 — без этого нельзя считать результат

1. **Починить out-of-sample MMM evaluation** ([run_mmm_eval.py](run_mmm_eval.py)).
   - Сейчас `mmm.sample_posterior_predictive` возвращает значения в нормированном пространстве, а попытка калибровки через `actual_train / pred_scaled_train` даёт почти одинаковый scale для всех geo (~12 900). Это значит train-pred тоже в scaled-space, и формула калибровки тождественна нулю.
   - Варианты: (a) использовать встроенный `mmm.predict()` который сам де-нормализует, (b) разобраться с `target_transformer` / `MaxAbsScaler` внутри `pymc-marketing` и применить inverse_transform явно, (c) перейти на ручной forward pass через trace (`adstock_alpha`, `saturation_lam`, `saturation_beta`, `intercept_contribution`) — у [run_mmm_attribution.py](run_mmm_attribution.py) уже есть похожая логика.
   - Definition of done: `R²_w > 0` на test, WMAPE сопоставима с baseline.

2. **Довести сходимость PyMC модели**.
   - Поднять `chains: 2 → 4`, `draws: 500 → 1000`. На GPU `chain_method: vectorized` это даст ~30 мин — приемлемо.
   - Если `sigma_country` всё ещё застревает — посмотреть, нет ли стран с 1–2 наблюдениями (нужно объединить в `country_other`).
   - Добавить prior-predictive check (сейчас его нет в [run_train.py](run_train.py)).

3. **Валидация MMM attribution per-channel**.
   - `mmm_attribution_summary.csv` уже выгружается, но сейчас нет sanity-проверки: не превышает ли `halo_total` для канала его spend / прочие health-check'и (saturation должна быть монотонной, adstock_alpha ∈ [0,1]). [run_mmm_analyze.py](run_mmm_analyze.py) уже грузит summary — добавить туда проверки.
   - Сделать **ROAS-таблицу** (`halo_per_1k_spend` уже считается, но без баров и без CI). Нужно построить per-channel CI (94% HDI), а не только posterior mean.

### Приоритет P1 — улучшение качества

4. **Объединение двух моделей**.
   - PyMC даёт `predicted organic_share` на cohort. MMM даёт `halo_organic` per (date, geo, channel).
   - Финальный артефакт для бизнеса — **отделить true organic (`organic_share × total_installs − halo`) от paid-induced organic**, per geo per date. Скрипта, который это делает, **сейчас нет**. Нужно сделать `run_combine.py` или ноутбук в `notebooks/`.

5. **Восстановить выкинутые фичи**.
   - `acceleration_iap`, `avg_check_d1_3_iap`, `avg_check_d4_7_iap`, `time_between_iap_2_3`, `zero_session_days` — закомментированы из-за outliers/багов. Починить в [src/organic_ratio/core/preprocessing/iap.py](src/organic_ratio/core/preprocessing/iap.py) и [.../sessions.py](src/organic_ratio/core/preprocessing/sessions.py), вернуть в `modeling.features`.

6. **Per-cohort PE breakdown**.
   - В отчётах есть PE-buckets (взвешенные по installs), но нет разреза по `country × platform`. Какие гео самые «непредсказуемые» — нужно вывести.

### Приоритет P2 — продакшен

7. **Sweep-эксперименты**.
   - В [run.ipynb](run.ipynb) последний ячейка `## (опц.) Пакетный запуск` — заготовка под `CONFIG_OVERRIDE_PATH`. Создать `conf/batch_training/sweep.yml` с разными `top_n_geos`, `adstock_l_max`, `saturation`, `cadence_days` и автоматическим сравнением метрик.

8. **Раскатка inference на свежие даты**.
   - Сейчас всё пишется в `data/predictions/`, нет save → S3. Добавить шаг 12, который выгружает predictions + attribution обратно в S3.

9. **CI/тесты**.
   - Тестов нет вообще. Минимум — smoke-test, что pipeline проходит на synthetic 100-row parquet.

---

## 4. Файлы и где что лежит

- [src/organic_ratio/core/loaders/](src/organic_ratio/core/loaders/) — S3 IO, materialize
- [src/organic_ratio/core/preprocessing/](src/organic_ratio/core/preprocessing/) — per-source feature builders, реестр [preprocesser_registry.py](src/organic_ratio/core/preprocessing/preprocesser_registry.py)
- [src/organic_ratio/core/cohort/](src/organic_ratio/core/cohort/) — merge, aggregator (SUM/MEAN policy), target build, clean
- [src/organic_ratio/core/modeling/](src/organic_ratio/core/modeling/) — baseline, pymc_model, mmm_data, mmm_model, metrics, preprocess (StandardScaler + category encoders)
- [src/organic_ratio/utils/config.py](src/organic_ratio/utils/config.py) — OmegaConf loader
- [conf/base/](conf/base/) — единственный источник параметров

Артефакты:
- `data/raw/partition/` — сырьё с S3
- `data/features/partition/` — user-grain features
- `data/features/cohort/cohort_level.parquet` — cohort-grain merged
- `data/features/targets/` + `data/train/` + `data/test/` — model-ready
- `data/features/mmm/` — MMM panels
- `data/models/{pymc,mmm}/` — trace.nc + prep.pkl
- `data/predictions/` — все predictions + attribution
- `data/plots/` — все PNG

---

## 5. Контакты и нюансы

- Git remote: `Anton-Filimoncev-azur/organic_ratio`, ветка `main`.
- На jupyter инстансе (jovyan) синк через GIT_PAT — см. первая ячейка [run.ipynb](run.ipynb).
- Все скрипты используют `polars` для тяжёлого IO и `pandas` только в моделинге (PyMC/sklearn ожидают numpy/pandas).
- `chain_method: vectorized` (numpyro) работает на 1 GPU; для CPU-multi-chain — `parallel` + `XLA_FLAGS` (см. [run_train.py:24-27](run_train.py#L24-L27)).
- В [parameters.yml](conf/base/parameters.yml) **две секции фичей**: `modeling.features` — для PyMC/baseline (whitelist после cleaning); `numerical.columns` / `categorical.columns` / `sequence.columns` — историческая разметка user-grain фичей, **сейчас не используется кодом**, можно почистить.
