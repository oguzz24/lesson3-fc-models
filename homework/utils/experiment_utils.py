"""Seeded experiments. Validation selects models; test curves are descriptive only."""
import argparse
import json
import logging
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from .visualization_utils import curves, weights

ROOT = Path(__file__).resolve().parents[2]
_DATA_CACHE = {}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)


def split_indices(size, seed, train_limit=0):
    """Split official training indices before any subsetting; never use test here."""
    if size < 2 or train_limit < 0:
        raise ValueError('Invalid dataset size or train limit')
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    cut = int(.9 * size)
    train_ids, val_ids = indices[:cut], indices[cut:]
    if train_limit:
        train_ids = train_ids[:train_limit]
        val_ids = val_ids[:max(100, train_limit // 5)]
    return train_ids, val_ids


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--datasets', nargs='+', choices=['mnist', 'cifar10'], default=['mnist', 'cifar10'])
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--train-limit', type=int, default=0)
    p.add_argument('--test-limit', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--smoke', action='store_true', help='Synthetic checks, never dataset evidence')
    p.add_argument('--download', action='store_true')
    return p


def loaders(args, dataset):
    seed_all(args.seed)
    cache_key = (dataset, args.seed, args.smoke, args.train_limit, args.test_limit)
    if cache_key in _DATA_CACHE:
        splits, inputs = _DATA_CACHE[cache_key]
        return tuple(DataLoader(ds, batch_size=args.batch_size, shuffle=i == 0,
                               generator=torch.Generator().manual_seed(args.seed))
                     for i, ds in enumerate(splits)), inputs
    shape = (1, 28, 28) if dataset == 'mnist' else (3, 32, 32)
    if args.smoke:
        def fake(n):
            return TensorDataset(torch.randn(n, *shape), torch.arange(n) % 10)
        train, val, test = fake(64), fake(32), fake(32)
    else:
        cls = datasets.MNIST if dataset == 'mnist' else datasets.CIFAR10
        stats = ((.1307,), (.3081,)) if dataset == 'mnist' else ((.4914,.4822,.4465), (.247,.243,.262))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
        full = cls(ROOT / 'data', train=True, transform=transform, download=args.download)
        test = cls(ROOT / 'data', train=False, transform=transform, download=args.download)
        train_ids, val_ids = split_indices(len(full), args.seed, args.train_limit)
        train, val = Subset(full, train_ids), Subset(full, val_ids)
        if args.test_limit:
            test = Subset(test, torch.randperm(len(test), generator=torch.Generator().manual_seed(args.seed)).tolist()[:args.test_limit])
    # Decode deterministic transforms once, so measured model time excludes PIL overhead.
    splits = []
    for ds in (train, val, test):
        xs, ys = [], []
        for x, y in DataLoader(ds, batch_size=256):
            xs.append(x); ys.append(y)
        splits.append(TensorDataset(torch.cat(xs), torch.cat(ys)))
    _DATA_CACHE[cache_key] = (splits, int(np.prod(shape)))
    return tuple(DataLoader(ds, batch_size=args.batch_size, shuffle=i == 0,
                            generator=torch.Generator().manual_seed(args.seed))
                 for i, ds in enumerate(splits)), int(np.prod(shape))


def evaluate(model, loader, device):
    model.eval(); total = correct = 0; loss = 0.; predictions = []; targets = []
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += nn.functional.cross_entropy(out, y, reduction='sum').item()
            pred = out.argmax(1); correct += (pred == y).sum().item(); total += len(y)
            predictions.extend(pred.cpu().tolist()); targets.extend(y.cpu().tolist())
    return {'loss': loss / total, 'accuracy': correct / total}, predictions, targets


def run(args, dataset, name, factory, output, weight_decay=0., schedule=None):
    if args.epochs < 1 or args.batch_size < 2 or args.train_limit < 0 or args.test_limit < 0:
        raise ValueError('epochs >= 1, batch-size >= 2 and nonnegative limits required')
    data, _ = loaders(args, dataset)
    if len(data[0].dataset) < 2:
        raise ValueError('At least two training examples are required')
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    destination = ROOT / 'homework' / 'results' / output / ('smoke' if args.smoke else 'real') / dataset / name
    destination.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    history = []; best = -1; best_epoch = 0; train_seconds = 0.; gradient_history = []
    for epoch in range(args.epochs):
        if schedule: schedule(model, epoch, args.epochs)
        model.train(); start = time.perf_counter(); gradients = {}
        for x, y in data[0]:
            if len(y) == 1: continue  # BatchNorm needs at least two training samples.
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y); loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    gradients.setdefault(n, []).append(p.grad.detach().norm().item())
            optimizer.step()
        if device.type == 'cuda': torch.cuda.synchronize()
        elapsed = time.perf_counter() - start; train_seconds += elapsed
        metrics = {split: evaluate(model, loader, device)[0] for split, loader in zip(('train','val','test'), data)}
        metrics['seconds'] = elapsed; history.append(metrics)
        gradient_history.append({n: float(np.mean(v)) for n, v in gradients.items()})
        if metrics['val']['accuracy'] > best:
            best = metrics['val']['accuracy']; best_epoch = epoch + 1
            torch.save(model.state_dict(), destination / 'best.pt')
        logging.info('%s %s epoch %s val=%.3f', dataset, name, epoch + 1, metrics['val']['accuracy'])
    curves(history, destination / 'curves.png'); weights(model, destination / 'weights.png')
    model.load_state_dict(torch.load(destination / 'best.pt', map_location=device, weights_only=True))
    test_metrics, predictions, targets = evaluate(model, data[2], device)
    x = next(iter(data[2]))[0].to(device)
    with torch.inference_mode():
        for _ in range(3): model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10): model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
    latency = (time.perf_counter() - start) / (10 * len(x))
    onset = next((i + 1 for i in range(1,len(history)) if history[i]['train']['loss'] < history[i-1]['train']['loss'] and history[i]['val']['loss'] > history[i-1]['val']['loss']), None)
    result = dict(name=name, dataset=dataset, synthetic=args.smoke, arguments=vars(args),
                  samples=[len(d.dataset) for d in data], parameters=sum(p.numel() for p in model.parameters()),
                  history=history, best_val=best, best_epoch=best_epoch, selected_test=test_metrics,
                  train_seconds=train_seconds, inference_seconds_per_image=latency,
                  overfit_signal_epoch=onset, gradient_history=gradient_history,
                  validation_accuracy_std=float(np.std([h['val']['accuracy'] for h in history])),
                  predictions=predictions, targets=targets, device=str(device))
    (destination / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result, model, destination


def summarize(results, output):
    path = ROOT / 'homework' / 'results' / output / ('smoke' if results[0]['synthetic'] else 'real')
    lines = ['# Experiment results', '', 'Synthetic smoke check; accuracy has no scientific meaning.' if results[0]['synthetic'] else 'Seeded subset/full-data experiment; sample counts and settings are in metrics.json.', '',
             '| Dataset | Model | Parameters | Val best | Test at selected epoch | Train seconds | Overfit signal |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['dataset']} | {r['name']} | {r['parameters']} | {r['best_val']:.4f} | {r['selected_test']['accuracy']:.4f} | {r['train_seconds']:.2f} | {r['overfit_signal_epoch']} |")
    for dataset in sorted({r['dataset'] for r in results}):
        best = max((r for r in results if r['dataset']==dataset), key=lambda r:r['best_val'])
        lines += ['', f"Validation-selected configuration for {dataset}: {best['name']}. Final train/test gap: {best['history'][-1]['train']['accuracy']-best['history'][-1]['test']['accuracy']:.4f}."]
    lines += ['', 'Overfit signal is the first epoch with falling train loss and rising validation loss, not proof of sustained overfitting. Standard deviation describes epoch variability, not uncertainty across seeds. Repeat with several seeds and more epochs before generalizing.']
    (path / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
