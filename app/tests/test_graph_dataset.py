import unittest
from app.utils.graph_dataset import GraphDataset

config = {
    "data": {

        # Training data examples, List of dictionaries with design and score keys
        "training_data": [ # (Required)
            {"design": ["A", "B", "C", "D", "E"], "score": 1.0}, 
            {"design": ["F", "G", "H", "I", "J"], "score": 0.8}, # "design" is a list of part IDs (str), "score" is the target value (float)
            {"design": ["K", "L", "M", "N", "O"], "score": 0.6}
        ],

        # Interaction library example, a dictionary mapping part IDs to part IDs they interact with along with the features
        "interaction_library": { # (Optional, required for interaction edges)
            "A": {"B": {"only_adjacent": "True", "features": [1.0, 2.5]}, "C": {"only_adjacent": "False", "features": [1.0, 2.5]}},
            "E": {"A": {"only_adjacent": "True", "features": [0.1]}, "C": {"only_adjacent": "False", "features": [0.2]},"H": {"only_adjacent": "True", "features": [0.3]},"E": {"only_adjacent": "True", "features": [0.4]}},
            "I": {"L": {"only_adjacent": "True", "features": [0.1]}, "B": {"only_adjacent": "True", "features": [0.2]},"J": {"only_adjacent": "False", "features": [0.3]},"N": {"only_adjacent": "True", "features": [0.4]}},
            "O": {"A": {"only_adjacent": "False", "features": [0.1]}, "K": {"only_adjacent": "True", "features": [0.2]},"D": {"only_adjacent": "True", "features": [0.3]}}
        },

        # Part library example, a dictionary mapping part IDs to different attributes
        "part_library": { # (Optional, required for node features)
            "A": {"B": [1.0], "averageScore":1.0, "lowScore": 1.0, "highScore": 1.0, "weights": [1.0], "type": "vowel", "features": [1.0, 2.1]},
            "B": {"C": [1.0], "averageScore":1.0, "lowScore": 1.0, "highScore": 1.0, "weights": [1.0], "type": "consonant", "features": [1.0, 3.0]},
            "C": {"D": [1.0], "averageScore":1.0, "lowScore": 1.0, "highScore": 1.0, "weights": [1.0], "type": "consonant", "features": [1.0, 5.5]},
            "D": {"E": [1.0], "averageScore":1.0, "lowScore": 1.0, "highScore": 1.0, "weights": [1.0], "type": "consonant", "features": [1.0, 2.3]},
            "E": {"averageScore":1.0, "lowScore": 1.0, "highScore": 1.0, "weights": [1.0], "type": "vowel", "features": [2.0, 1.2]},

            "F": {"G": [0.8], "averageScore":0.8, "lowScore": 0.8, "highScore": 0.8, "weights": [0.8], "type": "consonant", "features": [1.0, 9.0]},
            "G": {"H": [0.8], "averageScore":0.8, "lowScore": 0.8, "highScore": 0.8, "weights": [0.8], "type": "consonant", "features": [1.0, 8.0]},
            "H": {"I": [0.8], "averageScore":0.8, "lowScore": 0.8, "highScore": 0.8, "weights": [0.8], "type": "consonant", "features": [1.0, 7.0]},
            "I": {"J": [0.8], "averageScore":0.8, "lowScore": 0.8, "highScore": 0.8, "weights": [0.8], "type": "vowel", "features": [1.0, 6.0]},
            "J": {"averageScore":0.8, "lowScore": 0.8, "highScore": 0.8, "weights": [0.8], "type": "consonant", "features": [1.0, 5.0]},

            "K": {"L": [0.6], "averageScore":0.6, "lowScore": 0.6, "highScore": 0.6, "weights": [0.6], "type": "consonant", "features": [1.0, 4.0]},
            "L": {"M": [0.6], "averageScore":0.6, "lowScore": 0.6, "highScore": 0.6, "weights": [0.6], "type": "consonant", "features": [1.0, 3.0]},
            "M": {"N": [0.6], "averageScore":0.6, "lowScore": 0.6, "highScore": 0.6, "weights": [0.6], "type": "consonant", "features": [1.0, 2.0]},
            "N": {"O": [0.6], "averageScore":0.6, "lowScore": 0.6, "highScore": 0.6, "weights": [0.6], "type": "consonant", "features": [1.0, 1.0]},
            "O": {"averageScore":0.6, "lowScore": 0.6, "highScore": 0.6, "weights": [0.6], "type": "vowel", "features": [2.0, 0.0]},

            "na": {"type": "", "features": [0.0, 0.0]}
        }
    },

    # Data configuration
    "data_config": { # (Optional)
        "edges_to_use": "all",
        "use_oneHot_parts": "False",
        "use_oneHot_types": "False",
        "use_part_data": "False",
        "norm_part_data": "False",
        "use_gta": "False",
        "use_all_part_analytics": "False",
        "part_analytics_method": "avg",
        "norm_gta": "False"
    },

    # Training configuration
    "training_config": { # (Optional)
        "task": "regression",
        "model_name": "GraphConvRegr",
        "title": "testing",
        "training_metric": "mean_squared_error",
        "train_test_split": 0.7,
        "batch_size": 32,
        "pooling_method": "mean",
        "hidden_channels": 64,
        "learning_rate": 0.001,
        "epochs": 100,
        "early_stopping_patience": 10
    }
}

