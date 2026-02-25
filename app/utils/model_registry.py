import importlib

def get_model_registry():
    """
    Creates a model_registry of the form {model_name: model_class}
    Only includes models from the base_models folder.

    Returns:
        dict: model_registry
    """
    model_registry = {}

    # Import the base_models module
    base_models_module = importlib.import_module("app.models")

    # Get all model names from __all__ in base_models/__init__.py
    for model_name in getattr(base_models_module, '__all__', []):
        model_registry[model_name] = getattr(base_models_module, model_name)

    return model_registry

if __name__ == '__main__':
    model_registry = get_model_registry()
    print(model_registry)
    print(list(model_registry.keys()))
