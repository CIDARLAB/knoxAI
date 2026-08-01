class Config:
    def __init__(self, **kwargs):

        # -------------------------
        # Model Hyperparameters
        # -------------------------
        self.interactions = kwargs.get("interactions", 10)
        self.outer_bags = kwargs.get("outer_bags", 14)
        self.inner_bags = kwargs.get("inner_bags", 0)
        self.learning_rate = kwargs.get("learning_rate", 0.04)

        # -------------------------
        # Training Hyperparameters
        # -------------------------
        self.random_state = kwargs.get("random_state", 42)
        self.early_stopping_rounds = kwargs.get("early_stopping_rounds", 10)
        self.validation_size = kwargs.get("validation_size", 0.1)
