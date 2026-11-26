# 05 — Search Spaces, Decoding & Evaluators

## 1. Search spaces (`src/evosynth/search_spaces/`)

All search spaces are `ConfigSpace.ConfigurationSpace` objects, built from the ranges and choices in `settings/search_spaces.toml` (`[search_space.tabular]`, `.nas`, `.toy`). Changing a range there changes the space; no Python edit is needed.

### 1.1 `get_dummy_search_space()` (`toy.py`): 4 dimensions (test function)

| Name | Type | Range |
|---|---|---|
| `learning_rate` | float, log scale | 0.001 – 0.3 |
| `num_layers` | int | 1 – 10 |
| `optimizer` | categorical | Adam, SGD, RMSprop |
| `dropout` | float | 0.0 – 0.5 |

### 1.2 `get_tabular_search_space(include_model_type=True, include_fe_strategy=True)` (`tabular.py`): the main 8-dimensional tabular space

| Name | Type | Range / choices | Used by |
|---|---|---|---|
| `model_type` | categorical | XGBoost, RandomForest, LightGBM, ExtraTrees | Chooses the model family |
| `fe_strategy` | categorical | none, stats, interactions | `GenericFeatureEngineer` (§4) |
| `learning_rate` | float, log scale | 0.01 – 0.3 | XGBoost, LightGBM |
| `n_estimators` | int | 50 – 500 | All (RF/ET capped at 150 during search) |
| `max_depth` | int | 3 – 10 | All (RF/ET use depth×3, or unlimited when depth = 10) |
| `subsample` | float | 0.5 – 1.0 | XGBoost, LightGBM |
| `colsample_bytree` | float | 0.3 – 1.0 | XGB/LGBM; used as `max_features` for RF/ET |
| `reg_alpha` | float | 0.0 – 10.0 | XGBoost, LightGBM (L1) |

Without the two flags the space is 6-dimensional. XGBoost always uses the `gbtree` booster: the v1 `booster` gene (gbtree / dart) was removed because XGBoost 3.4 deprecates `dart`. `get_xgboost_search_space` is kept as an alias of `get_tabular_search_space`.

### 1.3 `get_image_nas_search_space()` (`nas.py`): 7 dimensions (CNN NAS)

| Name | Type | Choices / range |
|---|---|---|
| `conv1_channels` | categorical | 16, 32, 64 |
| `conv2_channels` | categorical | 32, 64, 128 |
| `fc_units` | categorical | 64, 128, 256 |
| `dropout_rate` | float | 0.0 – 0.5 |
| `learning_rate` | float, log scale | 1e-4 – 1e-1 |
| `optimizer` | categorical | Adam, SGD, RMSprop |
| `activation` | categorical | relu, leaky_relu |

---

## 2. Decoding the universal latent space

Every individual is a vector `z ∈ [0,1]^d`. `evosynth.core.decoding.decode_vector_to_dict(vector, space)`, used by the evaluators, the controller and the refit, decodes it like this:

```python
valid_vec = np.copy(individual.vector)
for i, hp_name in enumerate(list(space.keys())):
    hp = space[hp_name]                     # or space.get_hyperparameter(hp_name)
    if isinstance(hp, CategoricalHyperparameter):
        K = len(hp.choices)
        valid_vec[i] = float(min(int(np.floor(valid_vec[i] * K)), K - 1))
config = Configuration(space, vector=valid_vec)
```

- **Categoricals** with K choices are split into K equal bins: `index = min(floor(z·K), K−1)`. The `min` handles `z = 1.0`. ConfigSpace expects the choice index in the vector.
  - `model_type` (K = 4): `[0, 0.25)` → XGBoost, `[0.25, 0.5)` → RandomForest, `[0.5, 0.75)` → LightGBM, `[0.75, 1]` → ExtraTrees.
- **Floats and integers** are passed through as ConfigSpace's normalised vector values, and ConfigSpace maps them to real values:
  - linear: `x = L + z·(U − L)`;
  - log scale: `x = exp(ln L + z·(ln U − ln L))`;
  - integers are rounded from the scaled value.
- **Dimension order is `list(space.keys())`.** In the ConfigSpace version used, that order is **alphabetical**, as every log's hyperparameter listing shows: `colsample_bytree, fe_strategy, learning_rate, max_depth, model_type, n_estimators, reg_alpha, subsample` (a unit test checks this order). The vector index of a hyperparameter is its alphabetical position, not its declaration order.
- **Inactive dimensions.** For RF/ET, `learning_rate`, `subsample` and `reg_alpha` have no effect. They drift neutrally through the search, which the project lists as a known limitation.

Why the latent space is needed: PSO velocities, DE differences, GA blending and CMA-ES covariance all need continuous vectors. Binning lets those continuous operators move between model families and other categorical choices without special rules.

---

## 3. Evaluators

Evaluators live in `src/evosynth/evaluators/`. All return a **score in percent** (0–100; higher is better): the dataset's metric for tabular data, accuracy for NAS. Every model's `random_state` is `run.seed` (42); data splits use `run.data_seed` (42).

### 3.1 `DummyObjectiveFunction` (`toy.py`)

A microsecond-fast synthetic objective with a known optimum: `lr = 0.05`, `layers = 4`, `optimizer = Adam`, `dropout = 0.2`.

```
error = |lr − 0.05|·500 + |layers − 4|·5 + |dropout − 0.2|·100 + (0 if Adam else 20)
score = max(0, 100 − error)
```

### 3.2 `TabularEvaluator(dataset="covertype", subsample_size=None)` (`tabular.py`), the real-world benchmark

The evaluator is generic over the datasets registered in `settings/datasets.toml` (`evosynth datasets list`). Loading and splitting live in `src/evosynth/datasets/`, models in `src/evosynth/models/factory.py`, scoring in `models/metrics.py`.

**Datasets.**

| Dataset | Source | Rows | Features | Classes | Metric |
|---|---|---|---|---|---|
| Covertype | `sklearn.fetch_covtype` (UCI) | 581,012 | 54 | 7 | accuracy |
| HIGGS (98k version) | OpenML 23512 | 98,050 | 28 | 2 | ROC AUC |
| Adult | OpenML 1590 | 48,842 | 14 (8 categorical) | 2 | ROC AUC |
| Airlines | OpenML 1169 | 539,383 | 7 (4 categorical) | 2 | ROC AUC |
| Poker Hand (v2) | OpenML 1569 | 1,025,000 | 10 | 9 | accuracy |
| MiniBooNE | OpenML 41150 | 130,064 | 50 | 2 | ROC AUC |
| Jannis | OpenML 41168 | 83,733 | 54 | 4 | accuracy |

Binary datasets use ROC AUC because several are imbalanced (Adult is 76/24), where accuracy rewards predicting the majority class.

- **Loading** (`datasets/loaders.py`): downloaded once into `.data/` and stored as `.data/<name>/<name>.npz`. Labels are re-coded to `0..K−1`; categorical columns become integer codes (missing values get their own code); missing numbers are replaced by the column median.
- **Splits** (`datasets.partition`, `run.data_seed`): the **test rows** (20,000; 12,000 for Adult) are split off first, stratified, and are never seen by any search. The **search rows** (20,000) and the **train rows** for the refit (60,000; 36,000 for Adult) are stratified samples of the rest. `subsample_size` (CLI `--subsample`) takes a stratified subset of the search rows; the Phase 5 runs use 10,000. The exact row indices are published in the EvoSynth-Bench dataset (Zenodo 10.5281/zenodo.23645119).
- **Protocol:** `StratifiedKFold(evaluation.cv_folds = 2, shuffle=True, random_state=data_seed)`. In each fold the feature engineer is **fit on the training fold and applied to the validation fold**. For XGBoost and LightGBM, 15% of the training fold (`early_stopping_fraction`) is held back as the early-stopping set; the validation fold is used only for the score.
- **Models by `model_type`:**

| Choice | Construction during search (`build_model(phase="search")`) |
|---|---|
| XGBoost | `XGBClassifier(lr, n_est, depth, subsample, colsample, reg_alpha, booster="gbtree", early_stopping_rounds=10, eval_metric=logloss or mlogloss, n_jobs=1)` fit with `eval_set` = the early-stopping slice. When a training fold lacks a class (rare Poker Hand hands), labels are re-coded for training and mapped back for prediction. |
| LightGBM | `LGBMClassifier(lr, n_est, depth, subsample, colsample, reg_alpha, n_jobs=1)` with `early_stopping(10)` on the early-stopping slice |
| ExtraTrees | `ExtraTreesClassifier(n_estimators=min(n,150), max_depth = depth·3 if depth<10 else None, max_features=colsample, n_jobs=1)` on the whole training fold |
| RandomForest | Same rules as ExtraTrees, using `RandomForestClassifier` |

- **Failure handling:** a fold whose model raises an exception scores `evaluation.failed_fold_score` = 0; the error is printed and counted (`evaluator.failed_folds`, reported in `result.json`).
- **Score:** mean of the fold scores in the dataset's metric, × 100.
- **`evaluate_config(params, folds)`:** scores decoded hyperparameters with any number of folds; used by confirm-before-select (5 folds).

### 3.3 `ImageCNNEvaluator(space=None, epochs=None, batch_size=None, device=None)` (`image.py`), NAS

Defaults come from `[experiments.nas]`: 2 epochs, batch size 64, CPU.

- **Data:** Fashion-MNIST (`experiments.nas.dataset`), 70,000 grayscale 28×28 images of clothing in 10 balanced classes, downloaded via torchvision, scaled to `[0,1]` and shaped `(N, 1, 28, 28)`. The search trains on 10,000 and validates on 2,500 images, both stratified samples of the official 60,000 training images (`data_seed`); the official 10,000 test images are held out.
- **Seeding:** each configuration trains with a seed derived from `run.seed` and its vector, and training is serialised under a lock, so the same configuration always gets the same score.
- **`DynamicCNN`** (`src/evosynth/models/cnn.py`):

```
Conv2d(1 → conv1, 3×3, pad 1) → BatchNorm2d → act → MaxPool2d(2)
Conv2d(conv1 → conv2, 3×3, pad 1) → BatchNorm2d → act → AdaptiveAvgPool2d(2×2)
Flatten → Linear(conv2·4 → fc_units) → act → Dropout(p) → Linear(fc_units → num_classes)
act = ReLU or LeakyReLU
```

- **Training:** CrossEntropyLoss for a fixed number of epochs.
  - Optimiser: Adam, RMSprop, or SGD with momentum 0.9.
  - CUDA is used only if it is requested and available.
- **Score:** validation accuracy × 100.

---

## 4. `GenericFeatureEngineer(strategy, max_interaction_features=6)` (`src/evosynth/features/generic.py`)

This is a dataset-agnostic feature engineer: it uses no column names and no domain rules. Its thresholds come from `[features]` in `settings/evaluation.toml`.

**`fit(X)`:**
- `none`: does nothing.
- Otherwise, it marks a column as **continuous** if it has more than 15 unique values in the first `min(2000, n)` rows. If no column qualifies, all columns count as continuous.
- For `interactions` (when there is more than one continuous column), it ranks continuous columns by variance and keeps the top 6 as `top_cols`.

**`transform(X)`:**
- `none` returns `X` unchanged.
- `stats` appends 4 row-level statistics over the continuous columns: mean, standard deviation, minimum and maximum.
- `interactions` appends the `stats` features, then, for every pair `(i, j)` of `top_cols`, the difference `c_i − c_j` and the Euclidean norm `sqrt(c_i² + c_j²)`.
- The output is cast to float32.

On Covertype, the 10 cartographic columns are continuous:

| Strategy | Features | Calculation |
|---|---|---|
| `none` | 54 | — |
| `stats` | 58 | 54 + 4 |
| `interactions` | 88 | 58 + 15 pairs × 2 |

**Leakage control:** the evaluator and the refit scripts call `fit_transform` on the training data and `transform` on validation or test data.

**Result:** every Covertype run (EvoSynth, all isolated optimizers, Random Search and Optuna) chose `interactions`, and the best runs paired it with ExtraTrees.

---

## 5. Early stopping and acceleration

- **Boosted models:** XGBoost and LightGBM stop after 10 rounds without improvement on an early-stopping slice of the training fold (never on the fold that is scored, Phase 3). Poor configurations are cut short. (Early stopping on the same fold that is then scored makes those scores slightly optimistic. The same protocol is used for every candidate.)
- **Bagging models:** RF and ET are capped at 150 trees during search, which keeps evaluation cost bounded.
- **Thread-pool parallelism:** see [02-architecture.md](02-architecture.md) §5.
- **Cheap search, full refit:** the search uses 20,000 rows (10,000 in the Phase 5 runs) with 2-fold CV. After confirm-before-select, the selected configuration is refit on the train rows (60,000 for most datasets) with the same early-stopping rule and scored once on the held-out test rows ([07-experiments.md](07-experiments.md)).

---

## 6. Writing a custom evaluator

The contract is a `.space` attribute (a ConfigSpace) and `.evaluate(individual) -> float`:

```python
import ConfigSpace as CS, ConfigSpace.hyperparameters as CSH
from evosynth.core.decoding import decode_vector_to_dict

class MyEvaluator:
    def __init__(self):
        self.space = CS.ConfigurationSpace()
        self.space.add_hyperparameters([
            CSH.UniformFloatHyperparameter("learning_rate", 1e-4, 1e-1, log=True),
            CSH.CategoricalHyperparameter("model", ["a", "b", "c"]),
        ])
        self.X, self.y = load_my_data()

    def evaluate(self, individual) -> float:
        cfg = decode_vector_to_dict(individual.vector, self.space)  # bins categoricals, scales the rest
        return float(train_and_score(cfg, self.X, self.y))           # higher is better, e.g. %

from evosynth.engine.controller import DynamicOptimizer
best = DynamicOptimizer(search_space=MyEvaluator().space, evaluator=MyEvaluator(),
                        budget=500, pop_size=40, t1_iters=5, k_iters=3).run()
```

**Rules for custom evaluators:**
- Decode with `decode_vector_to_dict` (it bins categorical dimensions the same way as EvoSynth's evaluators).
- Keep `evaluate` thread-safe, because it is called from a thread pool.
- Set `n_jobs=1` inside models.
- Return a bounded score.

---

## Quiz

**1.** List the 8 dimensions of the tabular space in vector order.
<details><summary>Answer</summary>colsample_bytree, fe_strategy, learning_rate, max_depth, model_type, n_estimators, reg_alpha, subsample (alphabetical).</details>

**2.** Why was the `booster` gene removed?
<details><summary>Answer</summary>XGBoost 3.4 deprecates the dart booster; XGBoost now always uses gbtree, which also removes a mostly wasted dimension.</details>

**3.** Decode `model_type` = 0.81 and 0.25.
<details><summary>Answer</summary>0.81 → ExtraTrees (bin 0.75–1); 0.25 → RandomForest (bin 0.25–0.5).</details>

**4.** Which metric is used for Covertype, Adult and Poker Hand?
<details><summary>Answer</summary>Accuracy (Covertype, multiclass), ROC AUC (Adult, binary), accuracy (Poker Hand, multiclass).</details>

**5.** In what order are a dataset's test, search and train rows created, and which may overlap?
<details><summary>Answer</summary>Test rows first; then search rows and train rows are drawn from the rest. Search and train may overlap; neither overlaps test.</details>

**6.** What does a failed fold score now, and how would you notice many failures?
<details><summary>Answer</summary>0; each failure is printed and counted in `failed_folds` in result.json.</details>

**7.** For XGBoost in the search, which data is used for early stopping and which for the score?
<details><summary>Answer</summary>Early stopping on 15% of the training fold; the score on the validation fold.</details>

**8.** Why does XGBoost sometimes need its labels re-coded?
<details><summary>Answer</summary>XGBoost requires training labels 0..K−1; when a training fold lacks a rare class (Poker Hand), labels are re-coded for training and mapped back for prediction.</details>

**9.** How are RandomForest and ExtraTrees made cheaper during the search?
<details><summary>Answer</summary>Their trees are capped at 150 (and depth is max_depth·3, or unlimited at the top value).</details>

**10.** What does `evaluate_config` do, and who uses it?
<details><summary>Answer</summary>Scores decoded hyperparameters with a chosen number of folds; confirm-before-select uses it with 5 folds.</details>

**11.** What changed in the NAS evaluator's data?
<details><summary>Answer</summary>From 1,797 8×8 digits to Fashion-MNIST (28×28, 10 classes), training on 10,000 and validating on 2,500 images, with the official test set held out.</details>

**12.** Which function should a custom evaluator use to decode vectors, and why?
<details><summary>Answer</summary>`decode_vector_to_dict`, so categoricals, integers and log scales are decoded exactly as in EvoSynth's own evaluators.</details>
