import torch


def evaluate(model, X, y, criterion):
    model.eval()

    with torch.no_grad():
        # Forward pass
        logits = model(X)

        # Calculate loss
        loss = criterion(logits, y)

        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Convert probabilities to 0/1 predictions
        preds = (probs >= 0.5).float()

        # Calculate accuracy
        accuracy = (preds == y).float().mean()

    return loss.item(), accuracy.item()