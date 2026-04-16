# Model 2 Report Section - PyTorch CNN

## 1. Role of Model 2

Model 2 is the deep learning library-based CNN implementation for the emotion recognition assignment. Unlike Model 1, which is implemented from scratch, Model 2 uses PyTorch to build, train, validate, and test a convolutional neural network.

The model classifies face images into four emotion classes:

- angry
- happy
- sad
- surprise

The main goal of this part is to show a complete CNN training pipeline using a deep learning framework, including preprocessing, model definition, training, validation, testing, plots, confusion matrix, classification report, and K-fold validation.

## 2. Submitted Code Files

The Model 2 implementation is submitted as well-commented Python files:

- `model2/model2_emotion_cnn.py`: defines the CNN architecture.
- `model2/train.py`: trains and validates the model.
- `model2/evaluate.py`: tests the saved model and generates reports.
- `model2/utils.py`: contains helper functions such as accuracy calculation, seed setting, device selection, and folder creation.
- `evaluation/kfold_validation.py`: performs K-fold validation for Model 2.
- `evaluation/confusion_matrix.py`: writes the confusion matrix report and creates the plot.
- `evaluation/metrics.py`: contains shared metric functions.
- `evaluation/plots.py`: creates accuracy and K-fold plots.

## 3. Libraries Used

The following libraries were used:

- Python
- PyTorch
- torchvision
- matplotlib
- scikit-learn
- NumPy
- Pillow

PyTorch was used for model construction, training, and inference. torchvision was used for image loading, transforms, and ImageFolder dataset handling. matplotlib was used for plots and diagrams. scikit-learn was used for the confusion matrix and classification report.

## 4. Dataset and Preprocessing

The dataset was prepared from a public benchmark emotion dataset. Four emotion classes were selected: angry, happy, sad, and surprise.

The dataset preparation steps were:

- Download images from a public benchmark dataset.
- Store each class in a separate folder.
- Remove redundant duplicate images.
- Resize all images to 512 x 512 x 3.
- Split the cleaned data into train, validation, and test folders using a 70/20/10 ratio.

The prepared full dataset contains:

- Training images: 8410
- Validation images: 2402
- Test images: 1205

For the final CPU-feasible experiment, a balanced subset based on 1000 images per class was used:

- Training images used: 2800
- Validation images used: 800
- Test images used: 400

This keeps the experiment balanced while still satisfying the minimum data requirement.

During training, the following preprocessing was applied:

- Resize to 512 x 512.
- Convert image to tensor.
- Normalize using standard RGB mean and standard deviation.
- Apply light training-only augmentation.

The training augmentation included:

- Random horizontal flip.
- Small random rotation.
- Small brightness and contrast changes.

Validation and testing used only resizing, tensor conversion, and normalization. No random augmentation was used during validation or testing because those stages should measure stable model performance.

## 5. Model Architecture

The CNN architecture follows the required Model 2 assignment structure.

Layer-by-layer architecture:

- Input image: 512 x 512 x 3
- Conv1: 3 input channels to 32 filters, kernel size 3 x 3, valid convolution
- ReLU
- MaxPool2d: 2 x 2
- Conv2: 32 filters to 64 filters, kernel size 3 x 3, valid convolution
- ReLU
- MaxPool2d: 2 x 2
- Conv3: 64 filters to 64 filters, kernel size 3 x 3, valid convolution
- ReLU
- MaxPool2d: 2 x 2
- Conv4: 64 filters to 32 filters, kernel size 5 x 5, valid convolution
- ReLU
- Conv5: 32 filters to 16 filters, kernel size 7 x 7, valid convolution
- ReLU
- Flatten
- Fully connected hidden layer with 256 neurons
- Sigmoid activation
- Output layer with 4 logits

The layer dimensions are:

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
| Output layer | 4 logits |

The output layer gives raw logits during training. CrossEntropyLoss was used because it applies the softmax operation internally. For reporting probabilities during inference, softmax can be applied to the output logits.

Architecture diagram:

- `docs/diagrams/model2_architecture.png`

## 6. Hyperparameters

Final selected hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Image size | 512 x 512 |
| Batch size | 8 |
| Epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Label smoothing | 0.05 |
| Loss function | CrossEntropyLoss |
| Hidden layer size | 256 |
| Device | CPU |
| Number of classes | 4 |

These hyperparameters were selected to make training stable on a CPU-only machine. A smaller learning rate was used because the initial model was unstable and stayed close to the 25% random baseline. Light augmentation, weight decay, and label smoothing helped the model generalize better.

## 7. Training and Validation Results

The model was trained for 10 epochs. Validation was performed after every epoch, and the best model was saved according to validation accuracy.

| Epoch | Train Loss | Train Accuracy | Validation Loss | Validation Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 1.2741 | 38.54% | 1.2165 | 44.12% |
| 2 | 1.1674 | 48.96% | 1.1853 | 47.00% |
| 3 | 1.1003 | 53.93% | 1.1683 | 48.75% |
| 4 | 1.0590 | 56.89% | 1.1133 | 51.25% |
| 5 | 1.0076 | 60.21% | 1.0819 | 55.88% |
| 6 | 0.9709 | 62.00% | 1.0744 | 55.88% |
| 7 | 0.9469 | 61.79% | 1.0797 | 53.88% |
| 8 | 0.9104 | 64.32% | 1.0999 | 56.25% |
| 9 | 0.8637 | 68.14% | 1.1011 | 55.62% |
| 10 | 0.8245 | 70.75% | 1.0802 | 56.50% |

Best validation accuracy:

**56.50%**

The training accuracy increased from 38.54% to 70.75%, which shows that the model learned useful patterns from the training set. The validation accuracy improved from 44.12% to 56.50%, which is above the 25% random baseline for four balanced classes.

Training and validation plots:

- `outputs/plots/train_loss.png`
- `outputs/plots/val_loss.png`
- `outputs/plots/train_accuracy.png`
- `outputs/plots/val_accuracy.png`
- `outputs/plots/model2/accuracy_vs_iterations_model2.png`

## 8. Test Results

The best saved model was evaluated on the balanced test subset.

Final test accuracy:

**56.75%**

Test setup:

- Test images used: 400
- Test images per class: 100
- Classes: angry, happy, sad, surprise

The final test accuracy was close to the validation accuracy, which means the validation result was realistic and the model did not only memorize the validation set.

## 9. Classification Report

Final classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| angry | 0.5310 | 0.6000 | 0.5634 | 100 |
| happy | 0.6966 | 0.6200 | 0.6561 | 100 |
| sad | 0.3837 | 0.3300 | 0.3548 | 100 |
| surprise | 0.6429 | 0.7200 | 0.6792 | 100 |

Overall accuracy:

**56.75%**

The strongest class was surprise, with 72.00% recall. The weakest class was sad, with 33.00% recall. This suggests that sad facial expressions were more difficult for the model to separate from the other classes.

## 10. Confusion Matrix

The confusion matrix was generated to show the relationship between actual and predicted classes.

Interpretation:

- Main diagonal values are correct predictions.
- Off-diagonal values are wrong predictions.
- Rows represent actual classes.
- Columns represent predicted classes.

Per-class results from the confusion matrix:

- angry: 60.00%
- happy: 62.00%
- sad: 33.00%
- surprise: 72.00%

Overall accuracy:

**56.75%**

Overall error rate:

**43.25%**

Confusion matrix plot:

- `outputs/plots/confusion_matrix_model2.png`

## 11. K-Fold Validation

K-fold validation was performed with K = 4, which satisfies the minimum required value.

K-fold settings:

- K value: 4
- Epochs per fold: 1
- Batch size: 8
- Image size: 512 x 512
- Samples used: 4000
- Subset: balanced 1000 images per class

K-fold results:

- Fold 1 accuracy: 38.20%
- Fold 2 accuracy: 24.80%
- Fold 3 accuracy: 25.60%
- Fold 4 accuracy: 35.60%

Average K-fold accuracy:

**31.05%**

The K-fold result is lower than the final Model 2 test accuracy because each fold was trained for only one epoch to keep the experiment feasible on a CPU-only machine. The final reported Model 2 model was trained for 10 epochs.

K-fold plot:

- `outputs/plots/model2/kfold_accuracy_model2.png`

## 12. Sample Inputs and Predictions

A sample prediction sheet was created to show example input images and the model prediction for each one.

Sample prediction figure:

- `docs/diagrams/model2_sample_predictions.png`

This figure can be included in the final report or presentation to show real examples of the model output.

## 13. Final Model 2 Output Files

Important output files:

- `outputs/models/best_model2.pth`
- `outputs/reports/model2_results.txt`
- `outputs/reports/test_results.txt`
- `outputs/reports/classification_report.txt`
- `outputs/reports/confusion_matrix_report_model2.txt`
- `outputs/reports/kfold_results_model2.txt`
- `outputs/plots/confusion_matrix_model2.png`
- `outputs/plots/train_accuracy.png`
- `outputs/plots/val_accuracy.png`
- `outputs/plots/train_loss.png`
- `outputs/plots/val_loss.png`
- `outputs/plots/model2/accuracy_vs_iterations_model2.png`
- `outputs/plots/model2/kfold_accuracy_model2.png`
- `docs/diagrams/model2_architecture.png`
- `docs/diagrams/model2_sample_predictions.png`

## 14. Conclusion

Model 2 was successfully implemented using PyTorch. The CNN followed the required architecture, including valid convolutions, ReLU activations, max-pooling after the first three convolution layers, flattening, one hidden fully connected layer with Sigmoid activation, and a final four-class output layer.

The model achieved a best validation accuracy of 56.50% and a final test accuracy of 56.75%. This is clearly above the 25% random baseline for a balanced four-class classification problem. The model performed best on surprise and happy, while sad was the most difficult class.

This section can be merged directly with the Model 1 report section to form the final team submission.
