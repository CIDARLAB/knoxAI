import torch

def predict(model, data_loader, training_config):
    model.eval()
    preds = []
    with torch.no_grad():
        for data in data_loader:
            pred = model(data)
            preds.append(pred)
    return torch.cat(preds, dim=0)
