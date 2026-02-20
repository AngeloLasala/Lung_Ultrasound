"""
Helper for improving training stability
"""
import copy
import torch
import numpy as np
from sklearn.metrics import f1_score


class EMA:
    """
    Exponential Moving Average of model weights.

    Maintains a shadow copy of the model parameters updated as:
        shadow = decay * shadow + (1 - decay) * param

    Usage:
        ema = EMA(model, decay=0.999)
        # ... after each optimizer step:
        ema.update(model)
        # ... to evaluate with EMA weights:
        with ema.average_parameters(model):
            outputs = model(inputs)
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay

        # Deep-copy the initial weights as shadow parameters
        self.shadow = copy.deepcopy(model.state_dict())

        # Keep shadow on the same device as the model
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].float()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """
        Update shadow weights after each optimizer step.
        """

        model_state = model.state_dict()
        for k in self.shadow:
            if self.shadow[k].dtype.is_floating_point:
                self.shadow[k] = (
                    self.decay * self.shadow[k]
                    + (1.0 - self.decay) * model_state[k].float()
                )
            else:
                # Non-floating (e.g. BatchNorm num_batches_tracked): copy as-is
                self.shadow[k] = model_state[k].clone()

    def state_dict(self) -> dict:
        """
        Return the EMA shadow state dict (for saving checkpoints).
        """
        return copy.deepcopy(self.shadow)

    @torch.no_grad()
    def apply_shadow(self, model: torch.nn.Module):
        """
        Load EMA weights into model in-place.
        """
        model.load_state_dict(self.shadow)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, original_state: dict):
        """
        Restore original (non-EMA) weights into model in-place.
        """
        model.load_state_dict(original_state)

    class average_parameters:
        """
        Context manager: temporarily applies EMA weights for inference.
        """

        def __init__(self, ema: 'EMA', model: torch.nn.Module):
            self.ema = ema
            self.model = model
            self.original_state = None

        def __enter__(self):
            self.original_state = copy.deepcopy(self.model.state_dict())
            self.ema.apply_shadow(self.model)
            return self.model

        def __exit__(self, *args):
            self.ema.restore(self.model, self.original_state)

class EarlyStopping:
    """
    Stops training when the monitored metric has not improved for
    `patience` consecutive evaluation epochs.

    Args:
        patience: how many evaluations without improvement before stopping.
        min_delta: minimum change to qualify as an improvement.
        mode:      'min' (lower is better, e.g. loss) or
                   'max' (higher is better, e.g. accuracy).
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        assert mode in ('min', 'max'), "mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Call once per validation epoch.

        Returns True if training should stop, False otherwise.
        """
        improved = (
            value < self.best_value - self.min_delta
            if self.mode == 'min'
            else value > self.best_value + self.min_delta
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

def run_validation(model, valloader, criterion, device, monitor_metric):
    """
    Run one full validation pass and return a dict with all tracked metrics.

    Args:
        model:          model to evaluate (already has correct weights loaded).
        valloader:      validation DataLoader.
        criterion:      loss function.
        device:         torch device.
        monitor_metric: one of {'loss', 'weighted_f1'}.

    Returns:
        metrics (dict): {
            'loss':        float,
            'weighted_f1': float,
            'monitor':     float   ← the value of the metric to monitor
        }
    """
    model.eval()
    val_losses   = 0.0
    all_preds    = []
    all_labels   = []

    for videos, labels, subject, zones in valloader:
        videos, labels = videos.to(device), labels.to(device)

        with torch.no_grad():
            outputs    = model(videos)[0].to(device)
            loss       = criterion(outputs, labels)
            val_losses += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    avg_loss     = val_losses / len(valloader)
    weighted_f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    metrics = {
        'loss':        avg_loss,
        'weighted_f1': weighted_f1,
        'monitor':     avg_loss if monitor_metric == 'loss' else weighted_f1,
    }
    return metrics