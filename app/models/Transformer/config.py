class Config:
    def __init__(self, **kwargs):

        # -------------------------
        # MLflow / Experiment
        # -------------------------
        self.seed            = kwargs.get("seed", 42)

        # -------------------------
        # Task
        # -------------------------
        self.task = kwargs.get("task", "regression")   # "regression", "classification", "multiclass_classification", "ranking"

        self.num_classes = kwargs.get("num_classes", 2) # for multiclass_classification

        # -------------------------
        # Model Hyperparameters
        # -------------------------
        self.model_dim      = kwargs.get("model_dim", 128)
        self.num_heads      = kwargs.get("num_heads", 4)
        self.num_layers     = kwargs.get("num_layers", 3)
        self.dropout        = kwargs.get("dropout", 0.1)
        
        self.vocab_size     = kwargs.get("vocab_size", 19)
        self.features_dim   = kwargs.get("features_dim", 0)
        self.out_dim        = kwargs.get("out_dim", self.num_classes) if self.task == "multiclass_classification" else kwargs.get("out_dim", 1)

        # -------------------------
        # Training Hyperparameters
        # -------------------------
        self.lr          = kwargs.get("lr", 1e-4)
        self.batch_size  = kwargs.get("batch_size", 32)
        self.max_epochs  = kwargs.get("max_epochs", 500)
        self.num_workers = kwargs.get("num_workers", 0)
