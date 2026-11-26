# Search Space API Reference

Search spaces define the configuration boundaries and distributions for hyperparameters. They are built with `ConfigSpace` from the ranges in [`settings/search_spaces.toml`](https://github.com/paarthureddy/EvoSynth/blob/main/settings/search_spaces.toml); edit that file to change a range or add a choice.

* **Source:** [`src/evosynth/search_spaces/`](https://github.com/paarthureddy/EvoSynth/tree/main/src/evosynth/search_spaces)

---

## 1. `get_tabular_search_space()`

Defines the multi-model tabular search space (8 dimensions):

```python
from evosynth.search_spaces.tabular import get_tabular_search_space

space = get_tabular_search_space(include_model_type=True, include_fe_strategy=True)
```

### Dimensions:
* `model_type` (Categorical): `['XGBoost', 'RandomForest', 'LightGBM', 'ExtraTrees']`
* `fe_strategy` (Categorical): `['none', 'stats', 'interactions']`
* `learning_rate` (UniformFloat, log): `[0.01, 0.3]`
* `n_estimators` (UniformInteger): `[50, 500]`
* `max_depth` (UniformInteger): `[3, 10]`
* `subsample` (UniformFloat): `[0.5, 1.0]`
* `colsample_bytree` (UniformFloat): `[0.3, 1.0]`
* `reg_alpha` (UniformFloat): `[0.0, 10.0]`

---

## 2. `get_image_nas_search_space()`

Defines the Neural Architecture Search space for PyTorch CNNs:

```python
from evosynth.search_spaces.nas import get_image_nas_search_space

space = get_image_nas_search_space()
```

### Dimensions:
* `conv1_channels` (Categorical): `[16, 32, 64]`
* `conv2_channels` (Categorical): `[32, 64, 128]`
* `fc_units` (Categorical): `[64, 128, 256]`
* `dropout_rate` (UniformFloat): `[0.0, 0.5]`
* `learning_rate` (UniformFloat, log): `[1e-4, 1e-1]`
* `optimizer` (Categorical): `['Adam', 'SGD', 'RMSprop']`
* `activation` (Categorical): `['relu', 'leaky_relu']`

---

## 3. `get_dummy_search_space()`

A small toy space (`learning_rate`, `num_layers`, `optimizer`, `dropout`) used with `DummyObjectiveFunction` for fast engine tests.

```python
from evosynth.search_spaces.toy import get_dummy_search_space
```
