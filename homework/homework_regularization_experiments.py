"""Задание 3: регуляризация одной архитектуры и адаптация по слоям."""
from utils.experiment_utils import parser, run, summarize
from utils.model_utils import make_mlp, adaptive_dropout


def main():
    """Vary only regularization while retaining architecture and optimizer settings."""
    args = parser('Regularization at fixed [256,128,64] architecture').parse_args(); results = []
    configs = [('baseline',0,False,.1,0,False), *[(f'dropout_{p}',p,False,.1,0,False) for p in (.1,.3,.5)],
               ('batchnorm',0,True,.1,0,False),('dropout_bn',.3,True,.1,0,False),('l2',0,False,.1,.001,False),
               ('adaptive',.1,False,.1,0,True), *[(f'bn_momentum_{m}',0,True,m,0,False) for m in (.01,.5)],
               ('combined_adaptive',.1,True,.1,.001,True)]
    for dataset in args.datasets:
        inputs = 784 if dataset=='mnist' else 3072
        for name, drop, bn, momentum, decay, adaptive in configs:
            results.append(run(args,dataset,name,lambda:make_mlp(inputs,[256,128,64],drop,bn,momentum),
                               'regularization_experiments',decay,adaptive_dropout if adaptive else None)[0])
    summarize(results,'regularization_experiments')


if __name__ == '__main__': main()
