import torch

def validation_accuracy(state_dict, model_fn, val_loader, device):
    model = model_fn().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def weight_norm(state_dict):
    return -sum(torch.norm(param).item() for param in state_dict.values())
