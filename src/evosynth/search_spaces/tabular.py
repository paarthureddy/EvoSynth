"""
The tabular search space: model family + feature-engineering strategy + 7 hyperparameters.
Values come from [search_space.tabular] in settings/search_spaces.toml.
"""
import ConfigSpace as CS

from evosynth.config import get_settings
from evosynth.search_spaces._build import categorical, numeric


def get_tabular_search_space(include_model_type=True, include_fe_strategy=True, settings=None):
    """
    Build the tabular ConfigSpace. `model_type` (architecture) and `fe_strategy`
    (feature engineering) are categorical genes; the rest are numeric/categorical
    hyperparameters, all searched in the universal latent space [0, 1]^d.
    """
    t = (settings or get_settings()).search_space.tabular
    cs = CS.ConfigurationSpace()
    hps = []
    if include_model_type:
        hps.append(categorical("model_type", t.model_types))
    if include_fe_strategy:
        hps.append(categorical("fe_strategy", t.fe_strategies))
    hps.extend([
        numeric("learning_rate", t.learning_rate),
        numeric("n_estimators", t.n_estimators),
        numeric("max_depth", t.max_depth),
        numeric("subsample", t.subsample),
        categorical("booster", t.boosters),
        numeric("colsample_bytree", t.colsample_bytree),
        numeric("reg_alpha", t.reg_alpha),
    ])
    cs.add_hyperparameters(hps)
    return cs


# Historical name kept for existing callers.
get_xgboost_search_space = get_tabular_search_space
