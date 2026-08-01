class Config:
    def __init__(self, **kwargs):

        # -------------------------
        # MLflow / Experiment
        # -------------------------
        self.seed            = kwargs.get("seed", 42)

        # -------------------------
        # Task
        # -------------------------
        self.task = kwargs.get("task", "regression")   # "regression", "classification", "ranking"

        # -------------------------
        # Model Hyperparameters
        # -------------------------
        self.graph_conv       = kwargs.get("graph_conv", "gat")  # "graph", "nn", "gat", "tconv"
        self.hidden_dim        = kwargs.get("hidden_dim", 128)
        self.embedding_dim     = kwargs.get("embedding_dim", 64)
        self.num_layers        = kwargs.get("num_layers", 3)
        self.aggr              = kwargs.get("aggr", "mean")      # "sum", "mean", "max", "min"
        self.dropout           = kwargs.get("dropout", 0.1)
        
        
        self.node_features_dim = kwargs.get("node_features_dim", 0)
        self.edge_dim          = kwargs.get("edge_dim", 0)
        self.vocab_size        = kwargs.get("vocab_size", 19)
        self.features_dim      = kwargs.get("features_dim", 0)
        self.out_dim           = kwargs.get("out_dim", 1)

        # -------------------------
        # Training Hyperparameters
        # -------------------------
        self.lr          = kwargs.get("lr", 1e-3)
        self.batch_size  = kwargs.get("batch_size", 32)
        self.max_epochs  = kwargs.get("max_epochs", 500)
        self.num_workers = kwargs.get("num_workers", 0)
        