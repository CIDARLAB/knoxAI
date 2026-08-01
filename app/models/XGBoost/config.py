class Config:
    def __init__(self, **kwargs):
        self.n_estimators = kwargs.get("n_estimators", 100)
        self.max_depth = kwargs.get("max_depth", 4)
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.subsample = kwargs.get("subsample", 0.8)
        self.random_state = kwargs.get("random_state", 42)
