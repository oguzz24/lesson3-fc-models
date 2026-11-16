"""Задание 1: глубина сети и парное сравнение с Dropout + BatchNorm."""
from utils.experiment_utils import parser, run, summarize
from utils.model_utils import make_mlp


def main():
    """Train each prescribed depth with identical seeds and data splits."""
    args = parser('Depth including classifier: 1, 2, 3, 5, 7').parse_args()
    results = []
    for dataset in args.datasets:
        inputs = 784 if dataset == 'mnist' else 3072
        for depth in (1, 2, 3, 5, 7):
            for regularized in (False, True) if depth > 1 else (False,):
                name = f'depth_{depth}' + ('_dropout_bn' if regularized else '')
                result, _, _ = run(args, dataset, name, lambda: make_mlp(inputs, [128]*(depth-1), .3 if regularized else 0, regularized), 'depth_experiments')
                results.append(result)
    summarize(results, 'depth_experiments')


if __name__ == '__main__': main()
