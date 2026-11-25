"""Helpers turning master-file range tables into ConfigSpace hyperparameters."""
import ConfigSpace.hyperparameters as CSH


def numeric(name, rng):
    """{low, high, log?, int?} -> UniformInteger/UniformFloat hyperparameter."""
    log = bool(rng.get("log", False))
    if rng.get("int", False):
        return CSH.UniformIntegerHyperparameter(name, lower=int(rng["low"]), upper=int(rng["high"]), log=log)
    return CSH.UniformFloatHyperparameter(name, lower=float(rng["low"]), upper=float(rng["high"]), log=log)


def categorical(name, choices):
    return CSH.CategoricalHyperparameter(name, choices=list(choices))
