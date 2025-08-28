import os
import importlib

def get_model_registry():
    """
    Creates model_registry of the form task : {model_name : model}
    
    returns: model_registry
    """

    model_registry = {}

    tasks = [
        name for name in os.listdir("models")
        if os.path.isdir(os.path.join("models", name)) and not name.startswith("_") and not name == "base_models"
    ]

    for task in tasks:
        module = importlib.import_module(f"models.{task}")
        model_registry[task] = {
            name: getattr(module, name)
            for name in getattr(module, '__all__', [])
        }

    return model_registry

if __name__ == '__main__':
    model_registry = get_model_registry()
    print(model_registry)
    print(type(model_registry.get("regression", {}).get("NNConvRegr")))
