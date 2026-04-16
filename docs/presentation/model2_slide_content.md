# Model 2 PowerPoint Slide Content

Use these slides as the Model 2 section in the final team presentation. The images listed under each slide can be inserted directly into PowerPoint.

## Slide 1 - Model 2 Overview

Title:
Model 2: CNN Using PyTorch

Main points:
- Model 2 is the library-based CNN implementation.
- The goal is to classify facial expression images into four emotion classes.
- Selected classes: angry, happy, sad, surprise.
- The model was trained, validated, tested, and evaluated with plots and a confusion matrix.

Visual to add:
- Optional dataset/sample image preview.
- `docs/diagrams/model2_sample_predictions.png`

Speaker note:
This model is different from Model 1 because it uses PyTorch layers instead of manually implementing convolution and pooling from scratch.

## Slide 2 - Dataset and Preprocessing

Title:
Dataset Preparation for Model 2

Main points:
- Public benchmark emotion dataset was used.
- Four classes were selected: angry, happy, sad, surprise.
- Images were organized into separate class folders.
- Duplicate images were removed.
- Images were resized to 512 x 512 x 3.
- Dataset was split into 70% training, 20% validation, and 10% testing.

Numbers to show:
- Full training set: 8410 images.
- Full validation set: 2402 images.
- Full test set: 1205 images.
- Final CPU-feasible experiment used 1000 images per class in total.

Visual to add:
- `docs/diagrams/model2_sample_predictions.png`

Speaker note:
For the final experiment, a balanced subset was used to keep the run feasible on a CPU-only machine while keeping all classes equally represented.

## Slide 3 - CNN Architecture

Title:
Model 2 CNN Architecture

Main points:
- Input: 512 x 512 x 3 image.
- Five convolution layers were used.
- First three convolution layers were followed by ReLU and 2 x 2 MaxPooling.
- Last two convolution layers were followed by ReLU only.
- The output was flattened and passed to one hidden fully connected layer.
- Hidden activation: Sigmoid.
- Output: 4 logits for the four emotion classes.

Visual to add:
- `docs/diagrams/model2_architecture.png`

Speaker note:
The architecture follows the assignment requirements. Valid convolution was used, meaning no padding was added.

## Slide 4 - Layer Dimensions

Title:
Layer Dimensions and In/Out Data

Table to add:

| Stage | Output dimensions |
|---|---:|
| Input | 512 x 512 x 3 |
| Conv1 | 510 x 510 x 32 |
| MaxPool1 | 255 x 255 x 32 |
| Conv2 | 253 x 253 x 64 |
| MaxPool2 | 126 x 126 x 64 |
| Conv3 | 124 x 124 x 64 |
| MaxPool3 | 62 x 62 x 64 |
| Conv4 | 58 x 58 x 32 |
| Conv5 | 52 x 52 x 16 |
| Flatten | 43,264 |
| Hidden layer | 256 |
| Output | 4 logits |

Speaker note:
The flatten size was computed automatically inside the model using a dummy input, so we did not hardcode it in the CNN class.

## Slide 5 - Hyperparameters

Title:
Selected Hyperparameters

Main points:
- Image size: 512 x 512
- Batch size: 8
- Epochs: 10
- Optimizer: Adam
- Learning rate: 0.0001
- Weight decay: 0.0001
- Label smoothing: 0.05
- Loss function: CrossEntropyLoss
- Hidden layer size: 256
- Device: CPU
- Number of classes: 4

Additional note:
CrossEntropyLoss was used because this is a multi-class classification problem. The model outputs logits during training, and softmax is used only for probabilities during inference/reporting.

## Slide 6 - Training and Validation Curves

Title:
Training and Validation Performance

Main points:
- The model was trained for 10 epochs.
- Validation was performed after every epoch.
- The best model was saved based on validation accuracy.
- Training accuracy improved from 38.54% to 70.75%.
- Best validation accuracy was 56.50%.

Visuals to add:
- `outputs/plots/train_accuracy.png`
- `outputs/plots/val_accuracy.png`
- `outputs/plots/train_loss.png`
- `outputs/plots/val_loss.png`
- `outputs/plots/model2/accuracy_vs_iterations_model2.png`

Speaker note:
The curves show that the model learned useful patterns and improved clearly above the 25% random baseline for four classes.

## Slide 7 - Final Test Results

Title:
Model 2 Test Results

Main points:
- The best saved model was tested on a balanced test subset.
- Test images used: 400.
- Test images per class: 100.
- Final test accuracy: 56.75%.
- Overall error rate: 43.25%.

Classification summary:
- angry recall: 60.00%
- happy recall: 62.00%
- sad recall: 33.00%
- surprise recall: 72.00%

Speaker note:
The test accuracy is close to the validation accuracy, which means the validation result was realistic.

## Slide 8 - Confusion Matrix

Title:
Confusion Matrix Analysis

Main points:
- Rows represent actual classes.
- Columns represent predicted classes.
- Diagonal values are correct predictions.
- Off-diagonal values are wrong predictions.
- The model performed best on surprise and happy.
- The model struggled most with sad.

Visual to add:
- `outputs/plots/confusion_matrix_model2.png`

Speaker note:
The confusion matrix shows that the model is no longer predicting only one class. It is making predictions across all four emotion categories.

## Slide 9 - Sample Predictions

Title:
Example Model Predictions

Main points:
- This slide shows real test images.
- Each image is annotated with the true class and predicted class.
- Green titles mean correct predictions.
- Red titles mean wrong predictions.

Visual to add:
- `docs/diagrams/model2_sample_predictions.png`

Speaker note:
This visual makes the model output easier to understand and helps connect the numerical results to actual image examples.

## Slide 10 - K-Fold Validation

Title:
K-Fold Validation

Main points:
- K-fold validation was performed with K = 4.
- A balanced subset of 4000 images was used.
- Each fold was trained for 1 epoch due to CPU runtime limitations.

Results:
- Fold 1 accuracy: 38.20%
- Fold 2 accuracy: 24.80%
- Fold 3 accuracy: 25.60%
- Fold 4 accuracy: 35.60%
- Average K-fold accuracy: 31.05%

Visual to add:
- `outputs/plots/model2/kfold_accuracy_model2.png`

Speaker note:
The K-fold score is lower than the final test result because each fold was trained for only one epoch. The final Model 2 model was trained for 10 epochs.

## Slide 11 - Model 2 Conclusion

Title:
Model 2 Conclusion

Main points:
- Model 2 was successfully implemented using PyTorch.
- The required CNN architecture was followed.
- The model was trained, validated, and tested.
- Final validation accuracy: 56.50%.
- Final test accuracy: 56.75%.
- The model performed best on surprise and happy.
- Sad was the hardest class.

Final note:
Model 2 satisfies the assignment requirements for CNN implementation, training, validation, testing, plots, confusion matrix, K-fold validation, and reporting.

## Slide 12 - Files to Mention in Submission

Title:
Model 2 Submitted Files

Code files:
- `model2/model2_emotion_cnn.py`
- `model2/train.py`
- `model2/evaluate.py`
- `model2/utils.py`
- `evaluation/kfold_validation.py`
- `evaluation/confusion_matrix.py`
- `evaluation/metrics.py`
- `evaluation/plots.py`

Report and output files:
- `docs/report/model2_member4_report.md`
- `outputs/reports/model2_results.txt`
- `outputs/reports/classification_report.txt`
- `outputs/reports/confusion_matrix_report_model2.txt`
- `outputs/reports/kfold_results_model2.txt`

Visual files:
- `docs/diagrams/model2_architecture.png`
- `docs/diagrams/model2_sample_predictions.png`
- `outputs/plots/confusion_matrix_model2.png`
- `outputs/plots/model2/accuracy_vs_iterations_model2.png`
- `outputs/plots/model2/kfold_accuracy_model2.png`
