# Evaluators API Reference

Evaluators bridge the continuous $[0, 1]^d$ latent vectors produced by the optimizers to concrete Machine Learning training and validation passes. Every evaluator exposes `evaluate(individual) -> score` (higher is better).

All defaults below come from the [`settings/`](https://github.com/paarthureddy/EvoSynth/tree/main/settings) folder and can be changed there or per run with `--set section.key=value`.

---

## 1. `TabularEvaluator`

* **Source:** [`src/evosynth/evaluators/tabular.py`](https://github.com/paarthureddy/EvoSynth/blob/main/src/evosynth/evaluators/tabular.py)
* **Functionality:** Decodes a latent vector into a model family, a feature-engineering strategy and hyperparameters, then scores it with stratified cross-validation on a subsample of a registered dataset:
  * `XGBoost` and `LightGBM`: gradient boosted trees with early stopping.
  * `RandomForest` and `ExtraTrees`: bagged tree ensembles via `scikit-learn`.
  * Feature engineering: `none`, `stats` (row statistics) or `interactions` (pairwise interactions of continuous columns).

```python
from evosynth.evaluators.tabular import TabularEvaluator

evaluator = TabularEvaluator("covertype", subsample_size=20000)
score = evaluator.evaluate(individual)   # mean CV accuracy in %
```

### Features:
* **Stratified Cross-Validation:** `evaluation.cv_folds` folds (default 2).
* **Early Stopping:** boosting stops after `evaluation.early_stopping_rounds` stagnant rounds (default 10).
* **Datasets:** registered in `settings/datasets.toml` (`evosynth datasets list`); downloaded once into `.data/`.

---

## 2. `ImageCNNEvaluator` (Vision NAS)

* **Source:** [`src/evosynth/evaluators/image.py`](https://github.com/paarthureddy/EvoSynth/blob/main/src/evosynth/evaluators/image.py)
* **Functionality:** Evaluates Convolutional Neural Network topologies and training hyperparameters in PyTorch.

```python
from evosynth.evaluators.image import ImageCNNEvaluator
from evosynth.search_spaces.nas import get_image_nas_search_space

evaluator = ImageCNNEvaluator(space=get_image_nas_search_space(), epochs=2, batch_size=64, device="cpu")
accuracy = evaluator.evaluate(individual)
```

Defaults for `epochs`, `batch_size` and `device` come from `[experiments.nas]`. The architectural parameters searched are listed in the [Search Space reference](./search-space#2-get-image-nas-search-space).

---

## 3. `DummyObjectiveFunction`

* **Source:** [`src/evosynth/evaluators/toy.py`](https://github.com/paarthureddy/EvoSynth/blob/main/src/evosynth/evaluators/toy.py)
* **Functionality:** A fast synthetic objective over the toy search space, used to test the engine without training models.
