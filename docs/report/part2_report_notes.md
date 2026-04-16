# Part 2 Report Notes

This file summarizes the final reporting tasks required in Part 2 of the assignment.

## 1. Accuracy vs Number of Iterations

The training script saves a combined accuracy curve for Model 2:

- `outputs/plots/model2/accuracy_vs_iterations_model2.png`

The x-axis represents the training iteration/epoch number.
The y-axis represents accuracy percentage.
The plot includes:

- training accuracy
- validation accuracy

## 2. K-Fold Cross Validation

K-fold cross-validation is implemented in:

- `evaluation/kfold_validation.py`

The minimum value of K required by the assignment is 4.

Example command:

```bash
python evaluation/kfold_validation.py --k 4 --epochs 1
```

For fast local testing on CPU, a smaller sample can be used:

```bash
python evaluation/kfold_validation.py --k 4 --epochs 1 --image-size 128 --max-samples 400
```

Final outputs:

- `outputs/reports/kfold_results_model2.txt`
- `outputs/plots/model2/kfold_accuracy_model2.png`

## 3. Confusion Matrix

Model 2 evaluation saves:

- `outputs/plots/confusion_matrix_model2.png`
- `outputs/reports/confusion_matrix_report_model2.txt`

The confusion matrix is interpreted as follows:

- values on the main diagonal are correct predictions
- values outside the main diagonal are wrong predictions
- rows represent actual labels
- columns represent predicted labels

## 4. Block Diagram and Hyperparameters

The Model 2 block diagram can be generated using:

```bash
python docs/diagrams/generate_model2_architecture.py
```

Output:

- `docs/diagrams/model2_architecture.png`

Important hyperparameters:

- image size: `512 x 512 x 3`
- batch size: `8`
- epochs: `10`
- optimizer: `Adam`
- learning rate: `0.001`
- loss function: `CrossEntropyLoss`
- hidden layer activation: `Sigmoid`
- classes: `angry`, `happy`, `sad`, `surprise`

## 5. Preprocessing Steps

The dataset preparation is implemented in:

- `dataset/prepare_dataset.py`

Main preprocessing steps:

- selected 4 emotion classes
- removed exact duplicate images
- converted images to RGB
- resized images to `512 x 512`
- split data into `70% train`, `20% validation`, and `10% test`
- normalized images during training and evaluation using standard RGB mean/std

The preparation summary is saved in:

- `outputs/reports/dataset_preparation_summary.txt`

## 6. Final Report / Presentation

These notes can be copied into the final Word report or PowerPoint presentation.
The most important outputs to include are:

- accuracy-vs-iterations curve
- K-fold validation results
- confusion matrix
- Model 2 block diagram
- preprocessing summary
- final discussion comparing model performance
