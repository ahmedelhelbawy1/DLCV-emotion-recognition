# DLCV Emotion Assignment

This repository is organized for a university Deep Learning for Computer Vision assignment
about facial emotion recognition. The system works on 4 classes only:

- angry
- happy
- sad
- surprise

All images are prepared to `512 x 512 x 3`, and the dataset is split into:

- train
- val
- test

The assignment contains two separate models:

- `Model 1`: a from-scratch CNN-like pipeline with manual convolution, pooling, activation, flattening, feature extraction, and K-means
- `Model 2`: a CNN built with PyTorch for training, validation, testing, and reporting

## Project Structure

```text
DLCV_Emotion_Assignment/
├── dataset/
├── model1/
├── model2/
├── evaluation/
├── outputs/
├── docs/
├── notebooks/
├── team/
├── requirements.txt
├── README.md
└── .gitignore
```

## Folder Purpose

- `dataset/`: cleaned and split images used by both models
- `model1/`: from-scratch implementation for the first model
- `model2/`: PyTorch implementation for the second model
- `evaluation/`: shared metrics, confusion matrix, K-fold validation, and comparison scripts
- `outputs/`: generated models, plots, logs, reports, and predictions
- `docs/`: report, slides, diagrams, and hyperparameter records
- `notebooks/`: optional experiments and quick checks
- `team/`: team members, roles, and contribution notes

## Data Preparation

The script in `dataset/prepare_dataset.py` prepares the dataset by:

- reading images from a source folder with class subfolders
- keeping the required 4 classes only
- removing exact duplicate images
- resizing each image to `512 x 512`
- splitting the data into `70/20/10`

Run it from the project root like this:

```bash
python dataset/prepare_dataset.py --source /path/to/source_dataset --clear-output
```

## Model 2 Quick Start

To train Model 2:

```bash
python model2/train.py
```

To evaluate Model 2:

```bash
python model2/evaluate.py
```

## Notes

- This repository is organized by functionality, not by team member.
- The current `model2/` code is implemented and ready to use.
- The `model1/` and `evaluation/` folders are scaffolded clearly so teammates can work without confusion.
- Generated files should go inside `outputs/`, not inside the source folders.
