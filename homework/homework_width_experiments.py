"""Задание 2: фиксированная глубина и grid search ширины скрытых слоёв."""
from utils.experiment_utils import parser, run, summarize
from utils.model_utils import make_mlp
from utils.visualization_utils import heatmap


def main():
    """Compare four required widths, then search three width schedules."""
    args = parser('Three hidden layers, plus output classifier').parse_args(); results = []
    for dataset in args.datasets:
        inputs = 784 if dataset == 'mnist' else 3072
        configs = {'narrow':[64,32,16], 'medium':[256,128,64], 'wide':[1024,512,256], 'very_wide':[2048,1024,512]}
        for name, widths in configs.items():
            results.append(run(args,dataset,name,lambda:make_mlp(inputs,widths),'width_experiments')[0])
        grid = []
        for scheme in ('contracting','constant','expanding'):
            row = []
            for base in (64,128,256):
                widths = [base*4,base*2,base] if scheme=='contracting' else [base,base*2,base*4] if scheme=='expanding' else [base]*3
                r, _, path = run(args,dataset,f'{scheme}_{base}',lambda:make_mlp(inputs,widths),'width_experiments')
                results.append(r); row.append(r['best_val'])
            grid.append(row)
        heatmap(grid, ['contracting','constant','expanding'], [64,128,256], path.parent / 'grid.png')
    summarize(results,'width_experiments')


if __name__ == '__main__': main()
