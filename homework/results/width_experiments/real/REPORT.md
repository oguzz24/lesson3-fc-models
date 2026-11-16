# Experiment results

Seeded subset/full-data experiment; sample counts and settings are in metrics.json.

| Dataset | Model | Parameters | Val best | Test at selected epoch | Train seconds | Overfit signal |
|---|---|---:|---:|---:|---:|---:|
| cifar10 | narrow | 199450 | 0.4150 | 0.4090 | 4.31 | 4 |
| cifar10 | medium | 828490 | 0.4375 | 0.4280 | 7.97 | 2 |
| cifar10 | wide | 3805450 | 0.4158 | 0.4160 | 29.30 | 4 |
| cifar10 | very_wide | 8921610 | 0.4300 | 0.4030 | 69.54 | 4 |
| cifar10 | contracting_64 | 828490 | 0.4375 | 0.4280 | 8.05 | 2 |
| cifar10 | contracting_128 | 1738890 | 0.4433 | 0.4295 | 16.17 | 4 |
| cifar10 | contracting_256 | 3805450 | 0.4158 | 0.4160 | 33.74 | 4 |
| cifar10 | constant_64 | 205642 | 0.4283 | 0.4210 | 4.76 | 5 |
| cifar10 | constant_128 | 427658 | 0.4225 | 0.4145 | 6.00 | 5 |
| cifar10 | constant_256 | 920842 | 0.4217 | 0.4250 | 8.40 | 4 |
| cifar10 | expanding_64 | 240586 | 0.4242 | 0.4065 | 4.73 | 5 |
| cifar10 | expanding_128 | 563082 | 0.4350 | 0.4270 | 8.55 | 4 |
| cifar10 | expanding_256 | 1453834 | 0.4325 | 0.4165 | 13.55 | 4 |

Validation-selected configuration for cifar10: contracting_128. Final train/test gap: 0.3483.

Overfit signal is the first epoch with falling train loss and rising validation loss, not proof of sustained overfitting. Standard deviation describes epoch variability, not uncertainty across seeds. Repeat with several seeds and more epochs before generalizing.