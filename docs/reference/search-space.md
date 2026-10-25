# Search Space API Reference

Search spaces define the configuration boundaries, conditional relationships, and probability distributions for hyperparameters.

* **Source:** [`automl_engine/core/search_space.py`](file:///c:/Users/siddh/Desktop/College/EvoSynth/automl_engine/core/search_space.py)

---

## 1. `get_model_switching_search_space()`

Defines the multi-model architecture search space:

```python
from automl_engine.core.search_space import get_model_switching_search_space

space = get_model_switching_search_space()
```

### Dimensions:
* `model_type` (Categorical): `['XGBoost', 'RandomForest', 'LightGBM']`
* `learning_rate` (UniformFloat, log): `[0.01, 0.3]`
* `n_estimators` (UniformInteger): `[50, 500]`
* `max_depth` (UniformInteger): `[3, 10]`
* `subsample` (UniformFloat): `[0.5, 1.0]`
* `booster` (Categorical): `['gbtree', 'dart']`
* `colsample_bytree` (UniformFloat): `[0.3, 1.0]`
* `reg_alpha` (UniformFloat): `[0.0, 10.0]`

---

## 2. `get_image_nas_search_space()`

Defines the Neural Architecture Search space for PyTorch CNNs:

```python
from automl_engine.core.search_space import get_image_nas_search_space

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
