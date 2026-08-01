class Config:
    def __init__(self, **kwargs):

        # -------------------------
        # Model Hyperparameters
        # -------------------------
        self.random_state = kwargs.get("random_state", 42)
        self.n_estimators = kwargs.get("n_estimators", 100)
        self.max_depth = kwargs.get("max_depth", None)
        self.min_samples_split = kwargs.get("min_samples_split", 2)
        self.min_samples_leaf = kwargs.get("min_samples_leaf", 1)
        self.max_features = kwargs.get("max_features", "sqrt")

        # -------------------------
        # Training Hyperparameters
        # -------------------------
