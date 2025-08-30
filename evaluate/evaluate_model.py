import torch

from ..config.training_config import DEFAULTS
from ..utils.metrics import get_available_metrics
from ..utils.losses import get_criterion

def test_model(model, test_loader, training_config, all_metrics=False):
    training_config = {**DEFAULTS, **training_config}
    criterion = get_criterion(training_config.get('task'))
    preds = []
    labels = []
    
    model.eval()
    for data in test_loader:
        with torch.no_grad():
            pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
        preds.append(pred)
        labels.append(label)
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    test_loss = criterion(preds, labels)

    if all_metrics:
        metrics = {}
        for metric_name, metric_func in get_available_metrics().get(training_config.get('task')).items():
            metrics[metric_name] = metric_func(labels.cpu(), preds.cpu())
        return test_loss, metrics
    
    else:
        metric = get_available_metrics().get(training_config.get('task')).get(training_config.get('training_metric'))(labels.cpu(), preds.cpu())
        return test_loss, metric
    
