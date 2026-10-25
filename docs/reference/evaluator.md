# Evaluators API Reference

Evaluators bridge the continuous $[0, 1]^d$ latent vectors produced by the optimizers to concrete Machine Learning training and validation passes.

---

## 1. `XGBoostEvaluator` / `ModelSwitchingEvaluator`

* **Source:** [`automl_engine/core/evaluator.py`](file:///c:/Users/siddh/Desktop/College/EvoSynth/automl_engine/core/evaluator.py)
* **Functionality:** Dynamically evaluates hyperparameter configurations across multiple model architectures:
  * `XGBoost`: Gradient boosted decision trees with early stopping.
  * `LightGBM`: Fast histogram-based gradient boosting with early stopping.
  * `RandomForest`: Bagged ensemble trees via `scikit-learn`.

```python
from automl_engine.core.evaluator import XGBoostEvaluator, ModelSwitchingEvaluator

evaluator = ModelSwitchingEvaluator()
score = evaluator.evaluate(individual)
```

### Features:
* **3-Fold Stratified Cross-Validation:** Validates models across 3 balanced splits.
* **Early Stopping:** Halts tree generation after 10 stagnant rounds.

---

## 2. `ImageCNNEvaluator` (Vision NAS)

* **Source:** [`automl_engine/core/image_evaluator.py`](file:///c:/Users/siddh/Desktop/College/EvoSynth/automl_engine/core/image_evaluator.py)
* **Functionality:** Evaluates Convolutional Neural Network topologies and training hyperparameters in PyTorch.

```python
from automl_engine.core.image_evaluator import ImageCNNEvaluator

evaluator = ImageCNNEvaluator(epochs=3, batch_size=64, device="cpu")
accuracy = evaluator.evaluate(individual)
```

### Architectural Parameters Searched:
* `conv1_channels`: `[16, 32, 64]`
* `conv2_channels`: `[32, 64, 128]`
* `fc_units`: `[64, 128, 256]`
* `dropout_rate`: `[0.0, 0.5]`
* `activation`: `["relu", "leaky_relu"]`
* `optimizer`: `["Adam", "SGD", "RMSprop"]`
* `learning_rate`: `[1e-4, 1e-1]` (log scale)
