import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_dummy_search_space():
    """
    Returns a ConfigSpace defining the boundaries for our dummy model.
    This contains a mix of continuous, integer, and categorical hyperparameters.
    """
    cs = CS.ConfigurationSpace()

    # 1. Continuous (e.g., learning rate)
    lr = CSH.UniformFloatHyperparameter(
        "learning_rate", lower=0.001, upper=0.3, log=True)

    # 2. Integer (e.g., number of layers)
    layers = CSH.UniformIntegerHyperparameter("num_layers", lower=1, upper=10)

    # 3. Categorical (e.g., optimizer type)
    optimizer = CSH.CategoricalHyperparameter(
        "optimizer", choices=["Adam", "SGD", "RMSprop"])

    # 4. Another continuous (e.g., dropout)
    dropout = CSH.UniformFloatHyperparameter("dropout", lower=0.0, upper=0.5)

    cs.add_hyperparameters([lr, layers, optimizer, dropout])
    return cs


def get_xgboost_search_space(include_model_type=True):
    """
    Returns a ConfigSpace defining the boundaries for model optimization.
    Includes an architecture-level categorical hyperparameter `model_type`
    (['XGBoost', 'RandomForest', 'LightGBM']) along with continuous and discrete
    hyperparameters mapped across the Universal Latent Space.
    """
    cs = CS.ConfigurationSpace()

    hps = []
    if include_model_type:
        model_type = CSH.CategoricalHyperparameter(
            "model_type", choices=["XGBoost", "RandomForest", "LightGBM"]
        )
        hps.append(model_type)

    lr = CSH.UniformFloatHyperparameter(
        "learning_rate", lower=0.01, upper=0.3, log=True)
    n_est = CSH.UniformIntegerHyperparameter(
        "n_estimators", lower=50, upper=500)
    depth = CSH.UniformIntegerHyperparameter("max_depth", lower=3, upper=10)
    sub = CSH.UniformFloatHyperparameter("subsample", lower=0.5, upper=1.0)
    booster = CSH.CategoricalHyperparameter(
        "booster", choices=["gbtree", "dart"])
    colsample = CSH.UniformFloatHyperparameter(
        "colsample_bytree", lower=0.3, upper=1.0)
    reg_alpha = CSH.UniformFloatHyperparameter(
        "reg_alpha", lower=0.0, upper=10.0)

    hps.extend([lr, n_est, depth, sub, booster, colsample, reg_alpha])
    cs.add_hyperparameters(hps)
    return cs


def get_model_switching_search_space():
    """
    Convenience alias explicitly defining the multi-model architecture search space.
    """
    return get_xgboost_search_space(include_model_type=True)


def get_image_nas_search_space():
    """
    Returns a ConfigSpace for Neural Architecture Search (NAS) on image data (PyTorch CNN).
    Explores both architectural parameters (channels, layers, pooling) and optimization
    hyperparameters (learning rate, optimizer, dropout, batch size).
    """
    cs = CS.ConfigurationSpace()

    conv1_channels = CSH.CategoricalHyperparameter("conv1_channels", choices=[16, 32, 64])
    conv2_channels = CSH.CategoricalHyperparameter("conv2_channels", choices=[32, 64, 128])
    fc_units = CSH.CategoricalHyperparameter("fc_units", choices=[64, 128, 256])
    dropout = CSH.UniformFloatHyperparameter("dropout_rate", lower=0.0, upper=0.5)
    lr = CSH.UniformFloatHyperparameter("learning_rate", lower=1e-4, upper=1e-1, log=True)
    optimizer = CSH.CategoricalHyperparameter("optimizer", choices=["Adam", "SGD", "RMSprop"])
    activation = CSH.CategoricalHyperparameter("activation", choices=["relu", "leaky_relu"])

    cs.add_hyperparameters([conv1_channels, conv2_channels, fc_units, dropout, lr, optimizer, activation])
    return cs

