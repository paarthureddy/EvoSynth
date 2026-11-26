# Evaluators API Reference

Evaluators bridge the continuous $[0, 1]^d$ latent vectors produced by the optimizers to concrete Machine Learning training and validation passes. Every evaluator exposes `evaluate(individual) -> score` (higher is better).

All defaults below come from the [`settings/`](https://github.com/paarthureddy/EvoSynth/tree/main/settings) folder and can be changed there or per run with `--set section.key=value`.

---

## 1. `TabularEvaluator`

* **Source:** [`src/evosynth/evaluators/tabular.py`](https://github.com/paarthureddy/EvoSynth/blob/main/src/evosynth/evaluators/tabular.py)
* **Functionality:** Decodes a latent vector into a model family, a feature-engineering strategy and hyperparameters, then scores it with stratified cross-validation on the dataset's search rows (the held-out test rows are never used), with the dataset's metric (accuracy or ROC AUC):
  * `XGBoost` and `LightGBM`: gradient boosted trees, early-stopped on a slice of the training fold.
  * `RandomForest` and `ExtraTrees`: bagged tree ensembles via `scikit-learn`.
  * Feature engineering: `none`, `stats` (row statistics) or `interactions` (pairwise interactions of continuous columns).

```python
from evosynth.evaluators.tabular import TabularEvaluator

evaluator = TabularEvaluator("covertype", subsample_size=10000)   # optional: a subset of the search rows
score = evaluator.evaluate(individual)                            # mean CV score in % (dataset metric)
confirmed = evaluator.evaluate_config(params, folds=5)            # used by confirm-before-select
```

### Features:
* **Stratified Cross-Validation:** `evaluation.cv_folds` folds (default 2).
* **Early Stopping:** boosting stops after `evaluation.early_stopping_rounds` stagnant rounds (default 10) on `evaluation.early_stopping_fraction` (15%) of the training fold.
* **Failed fits:** a fold whose model raises an error scores 0; the error is printed and counted in `evaluator.failed_folds`.
* **Datasets:** Covertype, HIGGS, Adult, Airlines, Poker Hand, MiniBooNE, Jannis (registered in `settings/datasets.toml`, `evosynth datasets list`); downloaded once into `.data/`.

---

## 2. `ImageCNNEvaluator` (Vision NAS)

* **Source:** [`src/evosynth/evaluators/image.py`](https://github.com/paarthureddy/EvoSynth/blob/main/src/evosynth/evaluators/image.py)
* **Functionality:** Evaluates Convolutional Neural Network topologies and training hyperparameters in PyTorch on Fashion-MNIST (10,000 training and 2,500 validation images drawn from the official training set; the official test set is held out).

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
