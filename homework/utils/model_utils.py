"""Configurable fully connected models; depth includes the output layer."""
from torch import nn
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fully_connected_basics.models import FullyConnectedModel


def make_mlp(input_size, widths, dropout=0., batch_norm=False, momentum=.1):
    layers = []
    for width in widths:
        layers.append({'type': 'linear', 'size': width})
        if batch_norm:
            layers.append({'type': 'batch_norm'})
        layers.append({'type': 'relu'})
        if dropout:
            layers.append({'type': 'dropout', 'rate': dropout})
    model = FullyConnectedModel(input_size=input_size, num_classes=10, layers=layers)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = momentum
    return model


def adaptive_dropout(model, epoch, epochs):
    """Increase regularization over training, more strongly in later layers."""
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    for i, module in enumerate(drops):
        module.p = (.1 + .4 * epoch / max(epochs - 1, 1)) * (i + 1) / len(drops)
