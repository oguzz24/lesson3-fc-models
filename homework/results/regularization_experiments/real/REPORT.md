# Experiment results

Seeded subset/full-data experiment; sample counts and settings are in metrics.json.

| Dataset | Model | Parameters | Val best | Test at selected epoch | Train seconds | Overfit signal |
|---|---|---:|---:|---:|---:|---:|
| cifar10 | baseline | 828490 | 0.4375 | 0.4280 | 9.68 | 2 |
| cifar10 | dropout_0.1 | 828490 | 0.4392 | 0.4250 | 13.01 | 6 |
| cifar10 | dropout_0.3 | 828490 | 0.4183 | 0.3985 | 9.53 | 8 |
| cifar10 | dropout_0.5 | 828490 | 0.3575 | 0.3490 | 9.60 | None |
| cifar10 | batchnorm | 829386 | 0.4425 | 0.4460 | 8.88 | 6 |
| cifar10 | dropout_bn | 829386 | 0.4258 | 0.4305 | 10.23 | 6 |
| cifar10 | l2 | 828490 | 0.4425 | 0.4115 | 10.86 | 6 |
| cifar10 | adaptive | 828490 | 0.4342 | 0.4195 | 11.38 | None |
| cifar10 | bn_momentum_0.01 | 829386 | 0.4325 | 0.4410 | 10.32 | 4 |
| cifar10 | bn_momentum_0.5 | 829386 | 0.4400 | 0.4405 | 10.10 | 6 |
| cifar10 | combined_adaptive | 829386 | 0.4375 | 0.4180 | 10.99 | 6 |

Validation-selected configuration for cifar10: batchnorm. Final train/test gap: 0.3057.

Overfit signal is the first epoch with falling train loss and rising validation loss, not proof of sustained overfitting. Standard deviation describes epoch variability, not uncertainty across seeds. Repeat with several seeds and more epochs before generalizing.