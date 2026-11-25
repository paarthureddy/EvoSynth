# 05 — Search Spaces, Decoding & Evaluators

## 1. Search spaces (`automl_engine/core/search_space.py`)

All search spaces are `ConfigSpace.ConfigurationSpace` objects.

### 1.1 `get_dummy_search_space()`: 4 dimensions (test function)

| Name | Type | Range |
|---|---|---|
| `learning_rate` | float, log scale | 0.001 – 0.3 |
| `num_layers` | int | 1 – 10 |
| `optimizer` | categorical | Adam, SGD, RMSprop |
| `dropout` | float | 0.0 – 0.5 |

### 1.2 `get_xgboost_search_space(include_model_type=True, include_fe_strategy=True)`: the main 9-dimensional tabular space

| Name | Type | Range / choices | Used by |
|---|---|---|---|
| `model_type` | categorical | XGBoost, RandomForest, LightGBM, ExtraTrees | All evaluators: chooses the model family |
| `fe_strategy` | categorical | none, stats, interactions | Covertype evaluator only |
| `learning_rate` | float, log scale | 0.01 – 0.3 | XGBoost, LightGBM |
| `n_estimators` | int | 50 – 500 | All (RF/ET capped at 150 during search) |
| `max_depth` | int | 3 – 10 | All (RF/ET on Covertype use depth×3, or unlimited when depth = 10) |
| `subsample` | float | 0.5 – 1.0 | XGBoost, LightGBM |
| `booster` | categorical | gbtree, dart | XGBoost |
| `colsample_bytree` | float | 0.3 – 1.0 | XGB/LGBM; used as `max_features` for RF/ET |
| `reg_alpha` | float | 0.0 – 10.0 | XGBoost, LightGBM (L1) |

Without the two flags the space is 7-dimensional. `get_model_switching_search_space()` is an alias for the version with `model_type`. Note that the default call includes `fe_strategy`, so `main.py` and the UPGS ablations also search a 9-dimensional space, even though their evaluators ignore `fe_strategy` (§3).

### 1.3 `get_image_nas_search_space()`: 7 dimensions (CNN NAS)

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

Every individual is a vector `z ∈ [0,1]^d`. Each evaluator, and the controller's `_decode_individual`, decodes it like this:

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
- **Dimension order is `list(space.keys())`.** In the ConfigSpace version used, that order is **alphabetical**, as every log's hyperparameter listing shows: `booster, colsample_bytree, fe_strategy, learning_rate, max_depth, model_type, n_estimators, reg_alpha, subsample`. The vector index of a hyperparameter is its alphabetical position, not its declaration order. (The Hybrid Workflow document's example puts `model_type` first for readability.)
- **Inactive dimensions.** For RF/ET, `learning_rate`, `subsample`, `booster` and `reg_alpha` have no effect. They drift neutrally through the search, which the project lists as a known limitation.

Why the latent space is needed: PSO velocities, DE differences, GA blending and CMA-ES covariance all need continuous vectors. Binning lets those continuous operators move between model families and other categorical choices without special rules.

---

## 3. Evaluators

All evaluators return **accuracy in percent** (0–100; higher is better) and set every model's `random_state = 42`.

### 3.1 `DummyObjectiveFunction` (`evaluator.py`)

A microsecond-fast synthetic objective with a known optimum: `lr = 0.05`, `layers = 4`, `optimizer = Adam`, `dropout = 0.2`.

```
error = |lr − 0.05|·500 + |layers − 4|·5 + |dropout − 0.2|·100 + (0 if Adam else 20)
score = max(0, 100 − error)
```

### 3.2 `XGBoostEvaluator` (alias `ModelSwitchingEvaluator`), Wisconsin Breast Cancer

- **Data:** `sklearn.datasets.load_breast_cancer`, 569 samples, 30 features, binary.
- **Protocol:** `StratifiedKFold(3, shuffle=True, random_state=42)`; mean fold accuracy × 100.
- **Models by `model_type`:**
  - **XGBoost:** `XGBClassifier(lr, n_estimators, max_depth, subsample, booster, colsample_bytree, reg_alpha, early_stopping_rounds=10, eval_metric="logloss", n_jobs=1)`, fit with `eval_set = validation fold`.
  - **LightGBM:** `LGBMClassifier(...)` with the `lgb.early_stopping(10)` callback. On any exception it falls back to a simple XGBoost.
  - **Anything else:** `RandomForestClassifier(n_estimators=min(n,150), max_depth=depth, max_features=colsample)`. This includes the `ExtraTrees` choice: in this evaluator, choosing ExtraTrees trains a RandomForest.
- `fe_strategy`, if present in the space, is ignored.

### 3.3 `PathologicalXGBoostEvaluator(fast_mode=False)` (`hard_evaluator.py`), the noisy stress test

- **Data:** `make_classification(n_samples=2000, n_features=20, n_informative=5, n_redundant=10, n_classes=2, flip_y=0.3, random_state=42)`. The **30% random label flips** create a deceptive landscape on which optimizers tend to stagnate. This exercises the SWAP and injection safeguards.
- **Standard mode** (used for UPGS-120 and `main.py`): the same 3-fold stratified CV and model rules as §3.2.
- **`fast_mode`** (used for UPGS-2500): one stratified 75/25 split made once in `__init__`. It is about 15× faster than 3-fold CV and brought a 2,500-evaluation study down from roughly 13 hours to roughly 90 minutes.
- As in §3.2, ExtraTrees falls into the RandomForest branch and `fe_strategy` is ignored. In this setting the "ExtraTrees" architecture reported in UPGS results therefore means a RandomForest.

### 3.4 `CovertypeEvaluator(subsample_size=20000)` (`covtype_evaluator.py`), the real-world benchmark

- **Data:** `sklearn.datasets.fetch_covtype(data_home="<repo>/.data")`, cached as `.data/covertype/samples_py3` and `targets_py3`.
  - 581,012 rows, 54 features: 10 continuous cartographic features plus 4 wilderness and 40 soil-type binary indicators. 7 classes.
  - `X` is cast to float32 and `y = target − 1` (classes 0..6).
- **Search subsample:** a stratified `train_test_split(train_size=subsample_size, random_state=42)`, giving 20,000 rows. Class counts in the subsample:
  - 0: 7,292
  - 1: 9,752
  - 2: 1,231
  - 3: 94
  - 4: 327
  - 5: 598
  - 6: 706
- **Protocol:** `StratifiedKFold(2, shuffle=True, random_state=42)`. The feature engineer is **fit on the training fold and applied to the validation fold**, which avoids leakage.
- **Models by `model_type`:**

| Choice | Construction during search |
|---|---|
| XGBoost | `XGBClassifier(lr, n_est, depth, subsample, booster, colsample, reg_alpha, early_stopping_rounds=10, eval_metric="mlogloss", n_jobs=1, use_label_encoder=False)` fit with `eval_set = validation fold`. The deprecated `use_label_encoder` argument only produces warnings in the logs. |
| LightGBM | `LGBMClassifier(lr, n_est, depth, subsample, colsample, reg_alpha, n_jobs=1)` with `early_stopping(10)` |
| ExtraTrees | `ExtraTreesClassifier(n_estimators=min(n,150), max_depth = depth·3 if depth<10 else None, max_features=colsample, n_jobs=1)` |
| RandomForest | Same rules as ExtraTrees, using `RandomForestClassifier` |

- **Failure handling:** any exception scores that fold 0.5, i.e. 50% accuracy.
- **Score:** mean of the 2 fold accuracies × 100.

### 3.5 `ImageCNNEvaluator(space=None, epochs=3, batch_size=64, device="cpu")` (`image_evaluator.py`), NAS

- **Data:** `sklearn.datasets.load_digits`, 1,797 grayscale 8×8 images in 10 classes, scaled to `[0,1]` (divided by 16) and shaped `(N, 1, 8, 8)`. A stratified 75/25 train/validation split (`random_state=42`) is wrapped in DataLoaders, with the training loader shuffled.
- **`DynamicCNN`:**

```
Conv2d(1 → conv1, 3×3, pad 1) → BatchNorm2d → act → MaxPool2d(2)
Conv2d(conv1 → conv2, 3×3, pad 1) → BatchNorm2d → act → AdaptiveAvgPool2d(2×2)
Flatten → Linear(conv2·4 → fc_units) → act → Dropout(p) → Linear(fc_units → 10)
act = ReLU or LeakyReLU
```

- **Training:** CrossEntropyLoss for a fixed number of epochs.
  - Optimiser: Adam, RMSprop, or SGD with momentum 0.9.
  - The run script uses `epochs = 2`.
  - CUDA is used only if it is requested and available.
- **Score:** validation accuracy × 100.

---

## 4. `GenericFeatureEngineer(strategy, max_interaction_features=6)`

This is a dataset-agnostic feature engineer: it uses no column names and no domain rules.

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

- **Boosted models:** XGBoost and LightGBM stop after 10 rounds without improvement on the validation fold (`early_stopping_rounds=10` or `lgb.early_stopping(10)`). Poor configurations are cut short. (Early stopping on the same fold that is then scored makes those scores slightly optimistic. The same protocol is used for every candidate.)
- **Bagging models:** RF and ET are capped at 150 trees during search, which keeps evaluation cost bounded.
- **Thread-pool parallelism:** see [02-architecture.md](02-architecture.md) §5.
- **Covertype multi-fidelity:** the search uses 20k rows with 2-fold CV, about 10k training rows per fold. The single winner is then refit on 60k rows and tested on 20k unseen rows ([07-experiments.md](07-experiments.md) §4).

---

## 6. Writing a custom evaluator

The contract is a `.space` attribute (a ConfigSpace) and `.evaluate(individual) -> float`:

```python
import numpy as np, ConfigSpace as CS, ConfigSpace.hyperparameters as CSH
from ConfigSpace import Configuration

class MyEvaluator:
    def __init__(self):
        self.space = CS.ConfigurationSpace()
        self.space.add_hyperparameters([
            CSH.UniformFloatHyperparameter("learning_rate", 1e-4, 1e-1, log=True),
            CSH.CategoricalHyperparameter("model", ["a", "b", "c"]),
        ])
        self.X, self.y = load_my_data()

    def evaluate(self, individual) -> float:
        v = np.copy(individual.vector)
        for i, name in enumerate(list(self.space.keys())):           # categorical binning
            hp = self.space.get_hyperparameter(name)
            if isinstance(hp, CSH.CategoricalHyperparameter):
                K = len(hp.choices); v[i] = float(min(int(np.floor(v[i] * K)), K - 1))
        cfg = Configuration(self.space, vector=v)
        return float(train_and_score(cfg, self.X, self.y))           # higher is better, e.g. %

from automl_engine.engine.controller import DynamicOptimizer
best = DynamicOptimizer(search_space=MyEvaluator().space, evaluator=MyEvaluator(),
                        budget=500, pop_size=40, t1_iters=5, k_iters=3).run()
```

**Rules for custom evaluators:**
- Always bin categorical dimensions before building the `Configuration`.
- Keep `evaluate` thread-safe, because it is called from a thread pool.
- Set `n_jobs=1` inside models.
- Return a bounded score.
