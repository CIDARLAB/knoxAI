DEFAULTS = {
    'task': 'regression',
    'model_name': 'GraphConvRegr',
    'title': 'testing',
    'training_metric': 'mean_squared_error',

    'train_test_split': 0.7,
    'batch_size': 32,

    'pooling_method': 'mean',
    'hidden_channels': 64,
    'learning_rate': 0.001,

    'epochs': 100,
    'early_stopping_patience': 10,
}