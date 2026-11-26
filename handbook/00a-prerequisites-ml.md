# 00a — Prerequisites 1: Machine-Learning Basics

This file is for readers with no machine-learning background. It explains every ML idea the rest of the handbook uses, with examples taken from EvoSynth itself. Read it first, then [00b](00b-prerequisites-automl.md) (AutoML and hyperparameter search) and [00c](00c-prerequisites-evolutionary.md) (evolutionary algorithms and bandits).

---

## 1. Data

### 1.1 Rows, features and labels

A **dataset** is a table. Each **row** (also *sample* or *instance*) is one example; each **column** is one measured property, called a **feature**. One special column is the **label** (or *target*): the answer we want to predict.

Example from **Covertype**, one of EvoSynth's datasets: each row is a 30 m × 30 m patch of forest in Colorado. Its 54 features include elevation, slope, distance to water and soil type; its label is which of 7 tree species covers the patch. The dataset has 581,012 rows.

| Elevation (m) | Slope (°) | Distance to water (m) | … | Cover type (label) |
|---|---|---|---|---|
| 2596 | 3 | 258 | … | Aspen |
| 3120 | 12 | 42 | … | Spruce/Fir |

### 1.2 Feature types

- **Numeric (continuous):** a number on a scale, e.g. elevation 2596.
- **Categorical:** one of a fixed set of values with no order, e.g. an airline name in the **Airlines** dataset, or a workclass in **Adult**. Models usually need these turned into numbers; EvoSynth gives every category an integer code.
- **Binary:** a 0/1 indicator, e.g. "soil type 12 present".
- **Missing values:** empty cells. EvoSynth fills a missing number with the column's **median** and gives a missing category its own code.

### 1.3 Classification and classes

When the label is one of a fixed set of values, predicting it is **classification**. Each possible label is a **class**.

- **Binary classification:** two classes, e.g. **Adult** (income above or below $50k), **HIGGS** (signal or background particle event).
- **Multiclass classification:** more than two, e.g. Covertype (7 tree types), **Poker Hand** (9 hand types), **Jannis** (4 classes).

All of EvoSynth's tasks are classification. (Predicting a number, such as a price, would be *regression*.)

### 1.4 Balanced and imbalanced data

A dataset is **imbalanced** when some classes are much rarer than others. In Adult, 76% of people earn under $50k. A model that always answers "under $50k" is right 76% of the time while learning nothing. This matters for choosing a metric (§4).

---

## 2. Learning

### 2.1 Model, training and prediction

A **model** is a function that takes a row's features and outputs a predicted label. **Training** (or *fitting*) means adjusting the model to the data so its predictions match the labels. After training, the model makes **predictions** on rows it has not seen. This last step, in production, is called **inference**.

### 2.2 Parameters vs hyperparameters

- **Parameters** are learned *from the data* during training: the split points inside a decision tree, the weights of a neural network. Nobody sets them by hand.
- **Hyperparameters** are settings chosen *before* training that control *how* the model learns: how many trees, how deep, how fast to learn. The data cannot tell the model what they should be; they have to be searched for. Finding good hyperparameters is EvoSynth's job.

A cooking analogy: the recipe's *ingredients* are the data, the *taste you end up with* is the learned parameters, and the *oven temperature and cooking time* you pick beforehand are the hyperparameters.

### 2.3 EvoSynth's hyperparameters

EvoSynth's tabular search space has 8 dimensions (`settings/search_spaces.toml`):

| Hyperparameter | Type | Range | What it controls |
|---|---|---|---|
| `model_type` | categorical | XGBoost, RandomForest, LightGBM, ExtraTrees | Which kind of model to train (§3) |
| `fe_strategy` | categorical | none, stats, interactions | Which extra features to build (§5) |
| `n_estimators` | integer | 50–500 | Number of trees in the ensemble |
| `max_depth` | integer | 3–10 | How many questions each tree may ask in a row; deeper trees capture more detail but can memorise noise |
| `learning_rate` | float, log scale | 0.01–0.3 | Boosting only: how big a correction each new tree makes |
| `subsample` | float | 0.5–1.0 | Boosting only: share of rows each tree sees (randomness that reduces overfitting) |
| `colsample_bytree` | float | 0.3–1.0 | Share of features each tree may use (used as `max_features` by the forests) |
| `reg_alpha` | float | 0–10 | Boosting only: L1 regularisation, a penalty that keeps the model simpler |

**Log scale** means the search treats 0.01 → 0.03 as the same step size as 0.1 → 0.3. That suits values where the *ratio* matters more than the difference.

Some hyperparameters only matter for some models: `learning_rate` does nothing for a random forest. Such a value is *inactive* for that configuration.

---

## 3. The models EvoSynth chooses between

### 3.1 Decision tree

A **decision tree** asks a sequence of yes/no questions about the features, such as "elevation > 2800?" then "distance to water < 100?", until it reaches a leaf that gives a class. Training chooses the questions that best separate the classes. One deep tree tends to **overfit** (§4.2): it memorises the training rows.

### 3.2 Ensembles of trees

An **ensemble** combines many trees, which is far more accurate than one tree.

- **Random Forest:** many deep trees, each trained on a random sample of rows and allowed only a random subset of features at each question; the forest votes. Randomness makes the trees disagree in useful ways.
- **ExtraTrees (Extremely Randomised Trees):** like a random forest, but the split thresholds are also chosen at random. Even more varied trees, often faster to train.
- **Gradient boosting:** trees are built *one after another*, each one correcting the errors of the trees so far. `learning_rate` scales each correction. **XGBoost** and **LightGBM** are two popular, fast implementations with different internal tricks (LightGBM grows trees leaf by leaf, which is often faster on large data).

Forests (*bagging*) reduce variance by averaging independent trees; boosting reduces bias by fixing mistakes step by step. Which works best depends on the dataset, which is one reason to search over `model_type`.

### 3.3 Early stopping

Boosting adds trees one at a time, so we can watch a small **held-back slice** of the training data and stop when adding trees no longer improves it. In EvoSynth: XGBoost and LightGBM stop after **10** rounds without improvement on **15%** of the training rows. This saves time and prevents overfitting. Crucially, the slice used for stopping must not be the data used to *score* the model, otherwise the score looks better than it is. EvoSynth keeps them separate.

### 3.4 Neural networks and CNNs (used only for NAS)

A **neural network** is a stack of layers of simple units ("neurons"); each layer transforms its input with learned weights. A **convolutional neural network (CNN)** is built for images: its **convolution** layers slide small filters over the image to detect edges and shapes, **pooling** layers shrink the image, and **fully connected** layers at the end produce the class. Training runs for a number of **epochs** (passes over the training images) using an **optimizer** such as SGD, Adam or RMSprop, at a chosen **learning rate**. **Dropout** randomly switches off units during training to reduce overfitting. EvoSynth's NAS experiment searches over these choices on **Fashion-MNIST** (70,000 28×28 images of clothing in 10 classes).

---

## 4. Evaluating a model honestly

### 4.1 Training, validation and test data

If we score a model on the rows it was trained on, it looks excellent because it has seen the answers. So data is divided:

- **Training data:** used to fit the model.
- **Validation data:** used to compare models or hyperparameter settings *during* the search.
- **Test data (held-out):** used *once*, at the very end, to report how well the chosen model generalises. It must never influence any choice.

In EvoSynth each dataset is split once, with a fixed seed: **test rows** are taken out first (20,000 for most datasets) and never seen by any search; the search uses **search rows**; the final model is retrained on **train rows**.

### 4.2 Overfitting and generalisation

A model **overfits** when it learns noise or quirks of the training rows instead of the real pattern: high training score, lower score on new data. **Generalisation** is how well a model does on new data, which is what the test score measures. Deeper trees, more trees and weaker regularisation all raise the risk of overfitting.

### 4.3 Cross-validation

A single validation split can be lucky or unlucky. **k-fold cross-validation (CV)** splits the data into k parts ("folds"); the model is trained k times, each time validated on a different fold and trained on the rest; the k scores are averaged. **Stratified** CV keeps each fold's class proportions the same as the whole dataset's, which matters for rare classes.

EvoSynth uses **2-fold** stratified CV during the search (cheap but noisy) and **5-fold** CV to confirm the top candidates at the end (more expensive, more reliable).

### 4.4 Metrics

- **Accuracy:** the share of correct predictions. Easy to read, but misleading on imbalanced data (§1.4).
- **Balanced accuracy:** the average of the per-class recall (the share of each class predicted correctly), so every class counts equally.
- **ROC AUC (Area Under the ROC Curve):** for binary tasks. The model outputs a probability; AUC is the chance that a randomly chosen positive row gets a higher probability than a randomly chosen negative row. 0.5 = guessing, 1.0 = perfect. It does not depend on a decision threshold and is not fooled by imbalance.

EvoSynth reports **accuracy** for multiclass datasets and **ROC AUC** for binary ones, always as a percentage (e.g. AUC 0.93 is written 93%).

---

## 5. Feature engineering

**Feature engineering** creates new columns from existing ones so a model can find the pattern more easily. EvoSynth's `fe_strategy` gene chooses one of three generic strategies that need no knowledge of what the columns mean:

- **none:** use the features as they are.
- **stats:** add per-row statistics of the continuous columns (mean, standard deviation, minimum, maximum).
- **interactions:** the same row statistics, plus, for the most variable continuous columns (up to 6), pairwise differences (a − b) and distances (√(a² + b²)). In Covertype, the difference between two distance columns can be more informative than either alone.

The feature engineer is **fitted on training data only** and then applied to validation or test data, so nothing about the held-out rows leaks into the features.

---

## 6. After training: inference cost

A model that will be used in production is judged on more than its score:

- **Latency:** how long one prediction takes. EvoSynth measures the median time for a one-row prediction, in milliseconds.
- **Throughput:** predictions per second when many rows are scored at once.
- **Model size:** memory or disk needed to keep the model (EvoSynth measures the size of the saved model in MB).

An ensemble of many different models (the approach of some AutoML systems) multiplies all three; EvoSynth always delivers a single model. Forests with hundreds of deep trees are large (often 80–200 MB) and slower per row (about 4 ms); boosted models are typically small and fast (well under 1 MB and 1 ms).

---

## 7. Glossary

| Term | Meaning |
|---|---|
| Feature | An input column used for prediction |
| Label / target | The column to predict |
| Class | One possible label value |
| Imbalanced data | Some classes much rarer than others |
| Model | A function from features to a predicted label |
| Training / fitting | Adjusting a model's parameters to the data |
| Parameter | A value learned from data (e.g. a tree's split point) |
| Hyperparameter | A setting chosen before training (e.g. number of trees) |
| Inference | Using a trained model to predict |
| Overfitting | Learning noise; good on training data, worse on new data |
| Generalisation | Performance on new, unseen data |
| Validation data | Data used to compare settings during the search |
| Test data (held-out) | Data used once at the end, never for choices |
| Cross-validation (k-fold) | Train/validate k times on different folds, average the scores |
| Stratified | Keeping class proportions equal across splits |
| Accuracy | Share of correct predictions |
| ROC AUC | Probability that a positive is ranked above a negative |
| Decision tree | A model of nested yes/no questions |
| Random forest / ExtraTrees | Ensembles of randomised trees that vote |
| Gradient boosting (XGBoost, LightGBM) | Trees built in sequence, each correcting the last |
| Early stopping | Stop adding trees when a held-back slice stops improving |
| Feature engineering | Creating new features from existing ones |
| Latency / model size | Time per prediction / memory of the trained model |
| CNN | A neural network for images, built from convolution layers |
| Epoch | One full pass over the training data |

---

## Quiz

**1.** In the Adult dataset, what is the label, and is the task binary or multiclass?
<details><summary>Answer</summary>The label is whether a person's income is above $50k; two classes, so binary classification.</details>

**2.** Which of these is a hyperparameter: (a) a split threshold inside a trained tree, (b) `max_depth`, (c) the predicted class of a row?
<details><summary>Answer</summary>(b). It is chosen before training; (a) is a learned parameter and (c) is an output.</details>

**3.** A model scores 99% on its training rows and 71% on the test rows. What has most likely happened?
<details><summary>Answer</summary>Overfitting: it memorised the training data and does not generalise.</details>

**4.** True or false: it is fine to try several hyperparameter settings and keep the one with the best *test* score. Why?
<details><summary>Answer</summary>False. Choosing by the test score makes the test data part of the search, so the reported score is optimistic. Choices use validation data; the test set is used once at the end.</details>

**5.** Why does EvoSynth use ROC AUC, not accuracy, for Adult?
<details><summary>Answer</summary>Adult is imbalanced (76/24); accuracy rewards always predicting the majority class, while AUC measures how well positives are ranked above negatives regardless of imbalance.</details>

**6.** Put in order the steps of 2-fold stratified CV for one configuration: (A) average the two scores; (B) split the rows into two folds with equal class proportions; (C) train on fold 1, score on fold 2; (D) train on fold 2, score on fold 1.
<details><summary>Answer</summary>B → C → D → A (C and D can be swapped).</details>

**7.** What is the difference between a random forest and gradient boosting in how they build trees?
<details><summary>Answer</summary>A random forest builds many trees independently on random samples and lets them vote; boosting builds trees one after another, each correcting the errors of the previous ones.</details>

**8.** Why must the early-stopping slice not be the same data that the model is scored on?
<details><summary>Answer</summary>Stopping where that data looks best tunes the model to it, so its score on that same data becomes optimistic. EvoSynth stops on a slice of the training rows and scores on separate rows.</details>

**9.** A model has `learning_rate` 0.1 but `model_type` = RandomForest. What effect does the learning rate have?
<details><summary>Answer</summary>None: learning rate only applies to boosting, so it is inactive for a random forest.</details>

**10.** What does the `interactions` feature-engineering strategy add, and why is it fitted on training data only?
<details><summary>Answer</summary>Row statistics plus pairwise differences and distances of the most variable continuous columns. Fitting on training data only keeps any information about validation or test rows out of the features (no leakage).</details>

**11.** Name the three numbers EvoSynth measures to describe a model's inference cost.
<details><summary>Answer</summary>Median one-row latency (ms), batch throughput (rows per second) and model size (MB).</details>

**12.** In a CNN, what is the job of a convolution layer, and what does dropout do?
<details><summary>Answer</summary>Convolution slides small learned filters over the image to detect local patterns such as edges; dropout randomly switches off units during training to reduce overfitting.</details>
