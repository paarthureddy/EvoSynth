import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

def get_dummy_search_space():
    """
    Returns a ConfigSpace defining the boundaries for our dummy model.
    This contains a mix of continuous, integer, and categorical hyperparameters.
    """
    cs = CS.ConfigurationSpace()
    
    # 1. Continuous (e.g., learning rate)
    lr = CSH.UniformFloatHyperparameter("learning_rate", lower=0.001, upper=0.3, log=True)
    
    # 2. Integer (e.g., number of layers)
    layers = CSH.UniformIntegerHyperparameter("num_layers", lower=1, upper=10)
    
    # 3. Categorical (e.g., optimizer type)
    optimizer = CSH.CategoricalHyperparameter("optimizer", choices=["Adam", "SGD", "RMSprop"])
    
    # 4. Another continuous (e.g., dropout)
    dropout = CSH.UniformFloatHyperparameter("dropout", lower=0.0, upper=0.5)
    
    cs.add_hyperparameters([lr, layers, optimizer, dropout])
    return cs

def get_xgboost_search_space():
    """
    Returns a ConfigSpace defining the boundaries for a real XGBoost model.
    """
    cs = CS.ConfigurationSpace()
    
    lr = CSH.UniformFloatHyperparameter("learning_rate", lower=0.01, upper=0.3, log=True)
    n_est = CSH.UniformIntegerHyperparameter("n_estimators", lower=50, upper=500)
    depth = CSH.UniformIntegerHyperparameter("max_depth", lower=3, upper=10)
    sub = CSH.UniformFloatHyperparameter("subsample", lower=0.5, upper=1.0)
    booster = CSH.CategoricalHyperparameter("booster", choices=["gbtree", "dart"])
    
    cs.add_hyperparameters([lr, n_est, depth, sub, booster])
    return cs