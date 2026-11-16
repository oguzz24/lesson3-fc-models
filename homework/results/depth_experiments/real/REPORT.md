# Experiment results

Seeded subset/full-data experiment; sample counts and settings are in metrics.json.

| Dataset | Model | Parameters | Val best | Test at selected epoch | Train seconds | Overfit signal |
|---|---|---:|---:|---:|---:|---:|
| cifar10 | depth_1 | 30730 | 0.3450 | 0.3335 | 2.60 | 2 |
| cifar10 | depth_2 | 394634 | 0.4208 | 0.3965 | 5.00 | 4 |
| cifar10 | depth_2_dropout_bn | 394890 | 0.4242 | 0.4095 | 7.63 | 7 |
| cifar10 | depth_3 | 411146 | 0.4225 | 0.4160 | 5.53 | 5 |
| cifar10 | depth_3_dropout_bn | 411658 | 0.4308 | 0.4260 | 6.55 | 7 |
| cifar10 | depth_5 | 444170 | 0.4358 | 0.4205 | 6.59 | 4 |
| cifar10 | depth_5_dropout_bn | 445194 | 0.4167 | 0.4120 | 10.55 | None |
| cifar10 | depth_7 | 477194 | 0.4033 | 0.4105 | 7.12 | 4 |
| cifar10 | depth_7_dropout_bn | 478730 | 0.3775 | 0.3810 | 11.30 | None |

Validation-selected configuration for cifar10: depth_5. Final train/test gap: 0.2855.

Overfit signal is the first epoch with falling train loss and rising validation loss, not proof of sustained overfitting. Standard deviation describes epoch variability, not uncertainty across seeds. Repeat with several seeds and more epochs before generalizing.