import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_cnn_search_space():
    """
    Simplified 5-parameter CNN search space from the PDF.
    Layers: 3-10, Filters: 32-256, Kernel: {3x3, 5x5},
    LR: 0.0001-0.01 (log), Optimizer: {Adam, SGD, RMSprop}
    """
    cs = CS.ConfigurationSpace(seed=42)
    cs.add_hyperparameters([
        CSH.UniformIntegerHyperparameter("num_layers", lower=3, upper=10),
        CSH.UniformIntegerHyperparameter("filters", lower=32, upper=256),
        CSH.CategoricalHyperparameter("kernel", choices=[3, 5]),
        CSH.UniformFloatHyperparameter("learning_rate", lower=0.0001, upper=0.01, log=True),
        CSH.CategoricalHyperparameter("optimizer", choices=["Adam", "SGD", "RMSprop"]),
    ])
    return cs


def get_dummy_search_space():
    return get_cnn_search_space()


def get_xgboost_search_space():
    cs = CS.ConfigurationSpace(seed=42)
    cs.add_hyperparameters([
        CSH.UniformFloatHyperparameter("learning_rate", lower=0.01, upper=0.3, log=True),
        CSH.UniformIntegerHyperparameter("n_estimators", lower=50, upper=500),
        CSH.UniformIntegerHyperparameter("max_depth", lower=3, upper=10),
        CSH.UniformFloatHyperparameter("subsample", lower=0.5, upper=1.0),
        CSH.CategoricalHyperparameter("booster", choices=["gbtree", "dart"]),
    ])
    return cs