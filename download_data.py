"""Download the two prescribed datasets into this project's data directory."""
from pathlib import Path
from torchvision.datasets import MNIST, CIFAR10

if __name__ == '__main__':
    root = Path(__file__).resolve().parent / 'data'
    for dataset in (MNIST, CIFAR10):
        for train in (True, False):
            dataset(root, train=train, download=True)
