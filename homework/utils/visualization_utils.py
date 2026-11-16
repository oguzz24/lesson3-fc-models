import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def curves(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ('train', 'val', 'test'):
        for ax, metric in zip(axes, ('loss', 'accuracy')):
            ax.plot(range(1, len(history) + 1), [h[split][metric] for h in history], label=split)
            ax.set(xlabel='Epoch', ylabel=metric)
            ax.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def weights(model, path):
    params = [(n, p.detach().cpu().numpy().ravel()) for n, p in model.named_parameters() if p.ndim > 1]
    fig, axes = plt.subplots(len(params), 1, figsize=(7, 2 * len(params)), squeeze=False)
    for ax, (name, values) in zip(axes.flat, params):
        ax.hist(values, bins=40); ax.set_title(name)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def heatmap(values, rows, columns, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(np.asarray(values), aspect='auto')
    ax.set_xticks(range(len(columns)), columns); ax.set_yticks(range(len(rows)), rows)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f'{value:.3f}', ha='center', va='center')
    fig.colorbar(im, ax=ax, label='Validation accuracy')
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
