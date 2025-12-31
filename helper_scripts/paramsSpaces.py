

def _suggest_catboost_params_first(trial, classification_type="binary"):
    """Sugestões de parâmetros para CatBoost"""
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "verbose": False,  # Silenciar logs durante otimização
        "allow_writing_files": False,  # Não escrever arquivos de log
        "thread_count": THREADS,
        "random_seed": 42,
    }
    
    # Configure loss function and weights based on classification type
    if classification_type == "multiclass":
        params["loss_function"] = "MultiClass"
        params["classes_count"] = 3  # TSG=1, Oncogene=2, Passenger=0
        # Don't use scale_pos_weight for multiclass
    else:
        params["loss_function"] = "Logloss"
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", IMBALANCE_RATIO * 0.4, IMBALANCE_RATIO * 4)
    
    return params


def _suggest_catboost_params_second(trial, classification_type="binary"):
    """Sugestões de parâmetros reduzidos para CatBoost"""
    params = {
        "iterations": trial.suggest_int("iterations", 50, 1500),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),

        "verbose": False,
        "allow_writing_files": False,
        "thread_count": THREADS,
        "random_seed": 42,
    }
    
    # Configure loss function and weights based on classification type
    if classification_type == "multiclass":
        params["loss_function"] = "MultiClass"
        params["classes_count"] = 3  # TSG=1, Oncogene=2, Passenger=0
    else:
        #params["scale_pos_weight"] = IMBALANCE_RATIO
        params["loss_function"] = "Logloss"
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 5.0,6.0)
    
    return params

def _suggest_catboost_params_third(trial, classification_type="binary"):
    """Sugestões de parâmetros reduzidos para CatBoost"""
    params = {

        "iterations": trial.suggest_int("iterations", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.02),
        "depth": trial.suggest_int("depth", 2, 4),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 150, 255),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0, 1.0),

        "verbose": False,
        "allow_writing_files": False,
        "thread_count": THREADS,
        "random_seed": 42,
    }
    
    # Configure loss function and weights based on classification type
    if classification_type == "multiclass":
        params["loss_function"] = "MultiClass"
        params["classes_count"] = 3  # TSG=1, Oncogene=2, Passenger=0
    else:
        #params["scale_pos_weight"] = IMBALANCE_RATIO
        params["loss_function"] = "Logloss"
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 5.0,6.0)
    
    return params


def _suggest_decision_tree_params_first(trial):
    """Sugestões de parâmetros para Decision Tree"""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
        "random_state": 42
    }


def _suggest_decision_tree_params_second(trial):
    """Sugestões de parâmetros reduzidos para Decision Tree"""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 40),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),

        "random_state": 42
    }

def _suggest_decision_tree_params_third(trial):
    """Sugestões de parâmetros reduzidos para Decision Tree"""
    return {

        "max_depth": trial.suggest_int("max_depth", 2, 32),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 20),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
        "random_state": 42


    }


def _suggest_gradient_boosting_params_first(trial):
    """Sugestões de parâmetros para Gradient Boosting"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "random_state": 42
    }


def _suggest_gradient_boosting_params_second(trial):
    """Sugestões de parâmetros reduzidos para Gradient Boosting"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 30)


    }

def _suggest_gradient_boosting_params_third(trial):
    """Sugestões de parâmetros reduzidos para Gradient Boosting"""
    return {


        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.009, 0.1),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.8, 1.0),
        "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
        "random_state": 42

    }


def _suggest_hist_gradient_boosting_params_first(trial):
    """Sugestões de parâmetros para Histogram Gradient Boosting"""
    return {
        "max_iter": trial.suggest_int("max_iter", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        "random_state": 42
    }


def _suggest_hist_gradient_boosting_params_second(trial):
    """Sugestões de parâmetros reduzidos para Histogram Gradient Boosting"""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 30)



    }

def _suggest_hist_gradient_boosting_params_second(trial):
    """Sugestões de parâmetros reduzidos para Histogram Gradient Boosting"""
    return {


        "max_iter": trial.suggest_int("max_iter", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.009, 0.1),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
        "random_state": 42

    }


def _suggest_knn_params_first(trial):
    """Sugestões de parâmetros para KNN"""
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "p": trial.suggest_int("p", 1, 2),  # 1 for manhattan, 2 for euclidean
        "leaf_size": trial.suggest_int("leaf_size", 20, 40)
    }


def _suggest_knn_params_second(trial):
    """Sugestões de parâmetros reduzidos para KNN"""
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 100),
        "p": trial.suggest_int("p", 1, 4),  # 1 for manhattan, 2 for euclidean


    }

def _suggest_knn_params_third(trial):
    """Sugestões de parâmetros reduzidos para KNN"""
    return {

        "n_neighbors": trial.suggest_int("n_neighbors", 25, 125, step=10),
        "weights": "distance",
        "algorithm": trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "p": trial.suggest_int("p", 1, 2),  # 1 for manhattan, 2 for euclidean
        "leaf_size": trial.suggest_int("leaf_size", 20, 40)
    }


def _suggest_mlp_params_first(trial):
    """Sugestões de parâmetros para MLP"""
    # Sugerir arquitetura da rede
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layer_sizes = []
    for i in range(n_layers):
        layer_size = trial.suggest_int(f"layer_{i}_size", 10, 200)
        hidden_layer_sizes.append(layer_size)
    
    return {
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        "activation": trial.suggest_categorical("activation", ["tanh", "relu", "logistic"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
        "solver": trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"]),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 1000),
        "random_state": 42
    }


def _suggest_mlp_params_second(trial):
    """Sugestões de parâmetros reduzidos para MLP"""
    n_layers = trial.suggest_int("n_layers", 1, 2)
    hidden_layer_sizes = []
    for i in range(n_layers):
        layer_size = trial.suggest_int(f"layer_{i}_size", 100, 200, step=25)
        hidden_layer_sizes.append(layer_size)
    
    return {
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        "activation": trial.suggest_categorical("activation", ["tanh", "logistic"]),
        "alpha": trial.suggest_float("alpha", 1e-4, 0.01, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
        "max_iter": trial.suggest_int("max_iter", 200, 1500),
        "early_stopping": True,
        "n_iter_no_change": 10,
        "tol": 1e-4,
        "random_state": 42
    }

def _suggest_mlp_params_third(trial):
    """Sugestões de parâmetros reduzidos para MLP"""
    n_layers = trial.suggest_int("n_layers", 1, 2)
    hidden_layer_sizes = []
    for i in range(n_layers):
        layer_size = trial.suggest_int(f"layer_{i}_size", 100, 200, step=25)
        hidden_layer_sizes.append(layer_size)
    
    return {
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        "activation": trial.suggest_categorical("activation", ["tanh", "logistic"]),
        "alpha": trial.suggest_float("alpha", 1e-4, 0.01, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"]),
        "solver": trial.suggest_categorical("solver", ["adam", "sgd", "lbfgs"]),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True),
        "max_iter": trial.suggest_int("max_iter", 200, 1500),
        "early_stopping": True,
        "n_iter_no_change": 10,
        "tol": 1e-4,
        "random_state": 42
    }


def _suggest_random_forest_params_first(trial):
    """Sugestões de parâmetros para Random Forest"""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "n_jobs": -1,
        "random_state": 42
    }


def _suggest_random_forest_params_second(trial):
    """Sugestões de parâmetros reduzidos para Random Forest"""
    return {
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.3, 0.5, 0.7]),
        "max_depth": trial.suggest_int("max_depth", 5, 50),


        "n_jobs": -1,
        "random_state": 42
    }

def _suggest_random_forest_params_third(trial):
    """Sugestões de parâmetros reduzidos para Random Forest"""
    return {
        

        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 5, 20, step=2),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 15, 25, step=2),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),

        "n_jobs": -1,
        "random_state": 42
    }

def _suggest_svc_params_first(trial):
    """Sugestões de parâmetros para SVC otimizadas para evitar execução infinita"""
    
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    
    params = {
        "kernel": kernel,
        "C": trial.suggest_float("C", 0.1, 100.0, log=True), 
        "probability": True, 
        "max_iter": 1000,  
        "tol": 1e-3,  
        "cache_size": 200,  
        "degree": trial.suggest_int("degree", 2, 6),
        "shrinking": trial.suggest_categorical("shrinking", [True, False]),
        "random_state": 42
    }
    
    if kernel == "rbf":
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    
    return params

def _suggest_svc_params_second(trial):
    """Sugestões de parâmetros reduzidos para SVC"""
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    
    params = {
        "kernel": kernel,
        "probability": True, 
        "max_iter": 1000,  
        "tol": 1e-3,  
        "cache_size": 300,
        "random_state": 42
    }
    
    return params


def _suggest_svc_params_third(trial):
    """Sugestões de parâmetros reduzidos para SVC"""
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    
    params = {
        "kernel": kernel,
        "C": trial.suggest_float("C", 70.0, 100.0, log=True),
        "probability": True, 
        "max_iter": 1000,  
        "tol": 1e-3,  
        "cache_size": 300,
        "random_state": 42
    }
    
    return params


def _suggest_xgboost_params_first(trial, classification_type="binary"):
    """Sugestões de parâmetros para XGBoost"""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "random_state": 42,
        "n_jobs": -1,  # Use all available cores
        "verbosity": 0,  # Silenciar logs durante otimização
    }
    
    # Configure objective and weights based on classification type
    if classification_type == "multiclass":
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
        # Don't use scale_pos_weight for multiclass
    else:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    
    return params

def _suggest_xgboost_params_second(trial, classification_type="binary"):
    """Sugestões de parâmetros reduzidos para XGBoost"""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 30),

        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    
    # Configure objective and weights based on classification type
    if classification_type == "multiclass":
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
    else:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
    
    return params

def _suggest_xgboost_params_third(trial, classification_type="binary"):
    """Sugestões de parâmetros reduzidos para XGBoost"""
    params = {

        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step = 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.009, 0.1),
        "max_depth": trial.suggest_int("max_depth", 2, 20, step = 2),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.9, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.6),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0, step=1.0),
        "random_state": 42,
        "n_jobs": -1,  # Use all available cores
        "verbosity": 0,  # Silenciar logs durante otimização

        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,

    }
    
    # Configure objective and weights based on classification type
    if classification_type == "multiclass":
        params["objective"] = "multi:softprob"
        params["eval_metric"] = "mlogloss"
    else:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss",
        
        

    
    return params