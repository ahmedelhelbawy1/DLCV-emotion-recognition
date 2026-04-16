# Final Model 2 Run Plan

This file explains how to produce the final Model 2 results required by the assignment.

## Official Full Run

Use this command for the official full Model 2 train / validation / test pipeline:

```bash
python model2/run_final_pipeline.py
```

This uses:

- image size: `512 x 512 x 3`
- train set: full `dataset/train`
- validation set: full `dataset/val`
- test set: full `dataset/test`
- epochs: `10`
- batch size: `8`
- optimizer: `Adam`
- learning rate: `0.001`

Outputs:

- `outputs/models/best_model2.pth`
- `outputs/plots/train_loss.png`
- `outputs/plots/val_loss.png`
- `outputs/plots/train_accuracy.png`
- `outputs/plots/val_accuracy.png`
- `outputs/plots/model2/accuracy_vs_iterations_model2.png`
- `outputs/plots/confusion_matrix_model2.png`
- `outputs/reports/classification_report.txt`
- `outputs/reports/confusion_matrix_report_model2.txt`
- `outputs/reports/test_results.txt`

Logs:

- `outputs/logs/model2/official_train_model2.log`
- `outputs/logs/model2/official_evaluate_model2.log`

## Quick CPU Proof Run

Use this only to confirm that the whole pipeline works on a CPU machine:

```bash
python model2/run_final_pipeline.py --quick-test
```

This is not the final assignment result because it uses:

- image size: `128 x 128`
- a balanced subset of the dataset
- only `2` epochs

## K-Fold Validation

Official K-fold command:

```bash
python evaluation/kfold_validation.py --k 4 --epochs 1 --image-size 512
```

If full K-fold is too slow on CPU, use a clearly justified balanced subset:

```bash
python evaluation/kfold_validation.py --k 4 --epochs 1 --image-size 128 --max-samples 400
```

The K-fold report records the exact settings and the subset justification automatically.

Outputs:

- `outputs/reports/kfold_results_model2.txt`
- `outputs/plots/model2/kfold_accuracy_model2.png`
