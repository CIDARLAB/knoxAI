from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


##### General schemas #####
class PredictRequest(BaseModel):
    run_id : str
    samples : List[Any]
    sample_ids : List[str] | None = None
    save_predictions : bool | None = False

class PredictResponse(BaseModel): # Not Used
    predictions : List[Any]

class TrainResponse(BaseModel):
    run_id : str

class ConfigResponse(BaseModel):
    config : Dict[str, Any]

class TuneResponse(BaseModel): # Not Used
    best_params : Dict[str, Any]
    best_value  : float
    n_trials    : int

class EvaluateResponse(BaseModel):
    metrics : Dict[str, float]


##### Schemas for models (RF, EBM, XGBoost) #####
class ModelData(BaseModel):
    x_train : List[Any]
    y_train : List[float] | List[int] | List[List[int]]

    x_test  : List[Any] | None = None
    y_test  : List[float] | List[List[int]] | None = None

    feature_names : list[str] | None = None

    class Config:
        extra = "forbid"

class TrainRequest(BaseModel):
    data             : ModelData
    feature_names    : list[str] | None = None
    task             : Literal["classification", "regression", "multiclass_classification"] = "regression"
    config           : Dict = Field(default_factory=dict)
    experiment_name  : str = "model_experiment"
    run_name         : str = "model_run"
    interpret_shap   : bool = False

    class Config:
        extra = "forbid"

class TuneRequest(BaseModel):
    data             : ModelData
    feature_names    : list[str] | None = None
    task             : Literal["classification", "regression", "multiclass_classification"] = "regression"
    config           : Dict = Field(default_factory=dict)
    n_trials         : int = 50
    experiment_name  : str = "model_tuning_experiment"

    class Config:
        extra = "forbid"

class EvaluateRequest(BaseModel):
    run_id         : str
    x_test         : List[Any]
    y_test         : List[Any]
    feature_names  : list[str] | None = None



##### Transformer-specific schemas #####
class TransformerDataPoint(BaseModel):
    token_ids  : List[List[int]]
    features   : List[float] | None = None
    sequence   : str | None = None
    y          : List[float] | List[List[int]] | None = None

    class Config:
        extra = "forbid"

class TransformerData(BaseModel):
    # For Transformer model
    train : List[TransformerDataPoint]
    val   : List[TransformerDataPoint]
    test  : List[TransformerDataPoint] | None = None

    class Config:
        extra = "forbid"

class TransformerTrainRequest(BaseModel):
    data             : TransformerData
    rule_matrix      : ModelData | None = None
    vocab_size       : int = 19
    config           : Dict = Field(default_factory=dict)
    task             : Literal["classification", "regression", "multiclass_classification"] = "regression"
    experiment_name  : str = "transformer_experiment"
    run_name         : str = "transformer_run"

    class Config:
        extra = "forbid"

class TransformerTuneRequest(BaseModel):
    data             : TransformerData
    vocab_size       : int = 19
    config           : Dict = Field(default_factory=dict)
    n_trials         : int = 30
    task             : Literal["classification", "regression", "multiclass_classification"] = "regression"
    experiment_name  : str = "transformer_tuning_experiment"

    class Config:
        extra = "forbid"

class TransformerPredictRequest(BaseModel):
    run_id : str
    samples : List[TransformerDataPoint]
    sample_ids : List[str] | None = None
    save_predictions : bool | None = False


##### GNN-specific schemas #####
class GNNDataPoint(BaseModel):
    node_features : List[List[float]] | None = None
    node_labels   : List[int] | None = None
    node_sequence : List[str] | None = None
    edge_attr     : List[List[float]] | None = None
    edge_labels   : List[int] | None = None
    edge_index    : List[List[int]]
    features      : List[List[float]] | None = None
    sequence      : str | None = None
    y             : List[float] | List[List[int]] | None = None

    class Config:
        extra = "forbid"

class GNNData(BaseModel):
    train : List[GNNDataPoint]
    val   : List[GNNDataPoint]
    test  : List[GNNDataPoint] | None = None

    class Config:
        extra = "forbid"

class GNNTrainRequest(BaseModel):
    data             : GNNData
    rule_matrix      : ModelData | None = None
    vocab_size       : int = 19
    config           : Dict = Field(default_factory=dict)
    task             : Literal["classification", "regression", "multiclass_classification"] = "regression"
    experiment_name  : str = "gnn_experiment"
    run_name         : str = "gnn_run"

    class Config:
        extra = "forbid"

class GNNTuneRequest(BaseModel):
    data             : GNNData
    vocab_size       : int = 19
    config           : Dict = Field(default_factory=dict)
    task             : Literal["classification", "regression", "multiclass_classification"] = "regression"
    n_trials         : int = 30
    experiment_name  : str = "gnn_tuning_experiment"

    class Config:
        extra = "forbid"

class GNNPredictRequest(BaseModel):
    run_id : str
    samples : List[GNNDataPoint]
    sample_ids : List[str] | None = None
    save_predictions : bool | None = False


##### MLP-specific schemas #####
class MLPDataPoint(BaseModel):
    token_ids  : List[List[int]] | None = None
    features   : List[float] | None = None
    sequence   : str | None = None
    y          : List[float] | List[List[int]] | None = None

    class Config:
        extra = "forbid"

class MLPData(BaseModel):
    # For MLP model
    train : List[MLPDataPoint]
    val   : List[MLPDataPoint]
    test  : List[MLPDataPoint] | None = None

    class Config:
        extra = "forbid"

class MLPTrainRequest(BaseModel):
    data             : MLPData
    rule_matrix      : ModelData | None = None
    vocab_size       : int = 19
    config           : Dict = Field(default_factory=dict)
    task             : Literal["classification", "regression", "multiclass_classification"] = "regression"
    experiment_name  : str = "mlp_experiment"
    run_name         : str = "mlp_run"

    class Config:
        extra = "forbid"

class MLPTuneRequest(BaseModel):
    data             : MLPData
    vocab_size       : int = 19
    config           : Dict = Field(default_factory=dict)
    n_trials         : int = 30
    task             : Literal["classification", "regression", "multiclass_classification"] = "regression"
    experiment_name  : str = "mlp_tuning_experiment"

    class Config:
        extra = "forbid"

class MLPPredictRequest(BaseModel):
    run_id : str
    samples : List[MLPDataPoint]
    sample_ids : List[str] | None = None
    save_predictions : bool | None = False
    