"""Main pipeline runner for Model 1.

Main owners:
- Members 2 and 3

Purpose:
- connect all Model 1 parts together in one script
- load data, extract features, run K-means, and report results

Usage (from the project root):
    python -m model1.model1_pipeline
or:
    python model1/model1_pipeline.py

Performance note:
    ConvLayer.forward() uses Python loops — expect ~10-30 s per 512x512 image.
    Use max_per_class to cap the dataset size (team agreed: 1000/class max).
    Extracted features are cached to outputs/ so you only pay the cost once.

ConvLayer interface (Member 2 — conv_layer.py):
    ConvLayer(num_filters, filter_size, input_depth, filters=None)
    ConvLayer.forward(x)   x: (H, W, C) -> (H-fs+1, W-fs+1, num_filters)
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from model1.conv_layer import ConvLayer, get_predefined_filters
from model1.pooling_layer import PoolingLayer
from model1.activation import relu, ReLU
from model1.flatten import flatten, Flatten
from model1.feature_extractor import FeatureExtractor
from model1.kmeans_classifier import KMeansClassifier

# ── Hyperparameters ───────────────────────────────────────────────────────────

IMAGE_SIZE      = 512
NUM_CLASSES     = 4
NUM_FILTERS     = 5         # 5 predefined 3x3 filters per conv block
FILTER_SIZE     = 3
OUTPUT_DIM      = 128       # feature dimension after random projection
SEED            = 42
# Per-split caps (team agreed: 1000/class total = 700 train + 200 val + 100 test)
# Scale these down proportionally for quicker runs:
#   full:  TRAIN=700  VAL=200  TEST=100  (~7.7 h, leave overnight)
#   quick: TRAIN=70   VAL=20   TEST=10   (~47 min, results today)
#   smoke: TRAIN=7    VAL=2    TEST=1    (~5 min, pipeline check)
TRAIN_LIMIT = 70
VAL_LIMIT   = 20
TEST_LIMIT  = 10

CLASS_NAMES  = ['angry', 'happy', 'sad', 'surprise']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Predefined filters for block 1 — shape (5, 3, 3, 3)
PREDEFINED_FILTERS = get_predefined_filters(input_depth=3)

# Where cached feature arrays are stored
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'model1_features'
)

# ── Model1Pipeline class ──────────────────────────────────────────────────────

class Model1Pipeline:
    """3-block CNN feature extractor followed by random projection to 128-d.

    Architecture per block: ConvLayer -> PoolingLayer(MAX 2x2) -> ReLU
    Final stage:            Flatten -> FeatureExtractor (128-d random projection)
    """

    def __init__(self):
        # Block 1: RGB input (depth=3), predefined filters
        self.conv1 = ConvLayer(
            num_filters=NUM_FILTERS,
            filter_size=FILTER_SIZE,
            input_depth=3,
            filters=get_predefined_filters(input_depth=3)
        )
        # Block 2: depth = NUM_FILTERS (output channels of block 1)
        self.conv2 = ConvLayer(
            num_filters=NUM_FILTERS,
            filter_size=FILTER_SIZE,
            input_depth=NUM_FILTERS,
            filters=get_predefined_filters(input_depth=NUM_FILTERS)
        )
        # Block 3: depth = NUM_FILTERS (output channels of block 2)
        self.conv3 = ConvLayer(
            num_filters=NUM_FILTERS,
            filter_size=FILTER_SIZE,
            input_depth=NUM_FILTERS,
            filters=get_predefined_filters(input_depth=NUM_FILTERS)
        )
        self.pool      = PoolingLayer(pool_size=2, mode='MAX')
        self.extractor = FeatureExtractor(output_dim=OUTPUT_DIM, seed=SEED)

    def forward(self, image):
        """Run one image through the full feature-extraction pipeline.

        Parameters
        ----------
        image : np.ndarray  shape (H, W, 3) float32 in [0, 1]

        Returns
        -------
        np.ndarray  shape (OUTPUT_DIM,)
        """
        # Block 1: conv -> pool -> relu
        x = self.conv1.forward(image)
        x = self.pool.forward(x)
        x = relu(x)

        # Block 2: conv -> pool -> relu
        x = self.conv2.forward(x)
        x = self.pool.forward(x)
        x = relu(x)

        # Block 3: conv -> pool -> relu
        x = self.conv3.forward(x)
        x = self.pool.forward(x)
        x = relu(x)

        # Flatten -> random projection to 128-d
        x = flatten(x)
        x = self.extractor.transform(x)
        return x

# ── Dataset utilities ─────────────────────────────────────────────────────────

def _load_split(split_dir, max_per_class=None):
    """Return (paths, labels) for all images under split_dir/<class_name>/*.

    Parameters
    ----------
    split_dir     : str
    max_per_class : int or None — cap the number of images per class
    """
    paths, labels = [], []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        if max_per_class is not None:
            files = files[:max_per_class]
        for fname in files:
            paths.append(os.path.join(class_dir, fname))
            labels.append(CLASS_TO_IDX[class_name])
    return paths, labels


def _load_image(path):
    """Load image as float32 numpy array in [0, 1], shape (H, W, 3)."""
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.float32) / 255.0

# ── Feature extraction with caching ──────────────────────────────────────────

def _cache_path(split_name, limit):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tag = f"max{limit}" if limit else "full"
    return os.path.join(_CACHE_DIR, f"features_{split_name}_{tag}.npy"), \
           os.path.join(_CACHE_DIR, f"labels_{split_name}_{tag}.npy")


def extract_features(paths, labels, model, split_name,
                     limit=None, use_cache=True):
    """Extract features for a list of image paths, with disk caching.

    On first call the features are computed and saved to outputs/model1_features/.
    On subsequent calls the saved array is loaded directly — much faster.

    Returns
    -------
    X : np.ndarray  shape (N, OUTPUT_DIM)
    y : np.ndarray  shape (N,)
    """
    feat_file, lbl_file = _cache_path(split_name, limit)

    if use_cache and os.path.exists(feat_file) and os.path.exists(lbl_file):
        print(f"  Loading cached features from {feat_file}")
        return np.load(feat_file), np.load(lbl_file)

    features = []
    t0 = time.time()
    for i, path in enumerate(paths):
        img  = _load_image(path)
        feat = model.forward(img)
        features.append(feat)

        elapsed = time.time() - t0
        done    = i + 1
        remaining = (elapsed / done) * (len(paths) - done)
        print(f"\r  {done}/{len(paths)}  "
              f"elapsed {elapsed:.0f}s  "
              f"eta {remaining:.0f}s   ", end='', flush=True)

    print()
    X = np.array(features)
    y = np.array(labels)

    if use_cache:
        np.save(feat_file, X)
        np.save(lbl_file,  y)
        print(f"  Features saved to {feat_file}")

    return X, y

# ── Evaluation helpers ────────────────────────────────────────────────────────

def _report(split_name, y_true, y_pred, file=None):
    """Print and optionally write accuracy report for one split."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float(np.mean(y_true == y_pred))
    lines = [
        f"\n{split_name} accuracy: {acc:.4f}  "
        f"({int(acc * len(y_true))}/{len(y_true)})"
    ]
    for i, name in enumerate(CLASS_NAMES):
        mask = y_true == i
        if mask.any():
            cls_acc = float(np.mean(y_pred[mask] == i))
            lines.append(f"  {name:<10} {cls_acc:.4f}  ({mask.sum()} samples)")
    text = "\n".join(lines)
    print(text)
    if file:
        file.write(text + "\n")


def _save_results(dataset_dir, train_limit, val_limit, test_limit,
                  y_train, pred_train, y_val, pred_val, y_test, pred_test):
    """Write a results summary to outputs/reports/model1_results.txt."""
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'reports'
    )
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, 'model1_results.txt')

    with open(out_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Model 1 Results — CNN from scratch + K-means\n")
        f.write(f"per class  train={train_limit}  val={val_limit}  test={test_limit}\n")
        f.write(f"total images: {(train_limit+val_limit+test_limit)*NUM_CLASSES}\n")
        f.write("=" * 60 + "\n")
        _report("Train", y_train, pred_train, file=f)
        _report("Val",   y_val,   pred_val,   file=f)
        _report("Test",  y_test,  pred_test,  file=f)
        f.write("\n" + "=" * 60 + "\n")

    print(f"\nResults saved to {out_path}")

# ── Main entry point ──────────────────────────────────────────────────────────

def run(dataset_dir='dataset',
        train_limit=TRAIN_LIMIT,
        val_limit=VAL_LIMIT,
        test_limit=TEST_LIMIT,
        use_cache=True):
    """Execute the full Model 1 pipeline.

    Parameters
    ----------
    dataset_dir  : str
    train_limit  : int — max images per class from train/ (default 70)
    val_limit    : int — max images per class from val/   (default 20)
    test_limit   : int — max images per class from test/  (default 10)
    use_cache    : bool — save/load extracted features to outputs/model1_features/
    """
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir   = os.path.join(dataset_dir, 'val')
    test_dir  = os.path.join(dataset_dir, 'test')

    total = (train_limit + val_limit + test_limit) * NUM_CLASSES
    print("=" * 60)
    print("Model 1 - CNN from scratch + K-means")
    print(f"per class  train={train_limit}  val={val_limit}  test={test_limit}"
          f"  (total ~{total} images)")
    print(f"use_cache={use_cache}")
    print("=" * 60)

    # 1. Collect paths and labels
    print("\nLoading dataset paths ...")
    train_paths, train_labels = _load_split(train_dir, train_limit)
    val_paths,   val_labels   = _load_split(val_dir,   val_limit)
    test_paths,  test_labels  = _load_split(test_dir,  test_limit)
    print(f"  train: {len(train_paths)}  "
          f"val: {len(val_paths)}  "
          f"test: {len(test_paths)}")

    # 2. Build model
    model = Model1Pipeline()

    # 3. Extract features — slow first time, instant on cache reload
    print("\nExtracting training features ...")
    X_train, y_train = extract_features(
        train_paths, train_labels, model, 'train', train_limit, use_cache)
    print(f"  X_train shape: {X_train.shape}")

    print("\nExtracting validation features ...")
    X_val, y_val = extract_features(
        val_paths, val_labels, model, 'val', val_limit, use_cache)

    print("\nExtracting test features ...")
    X_test, y_test = extract_features(
        test_paths, test_labels, model, 'test', test_limit, use_cache)

    # 4. Train K-means classifier
    print("\nTraining K-means classifier ...")
    t0  = time.time()
    clf = KMeansClassifier(n_clusters=NUM_CLASSES, seed=SEED)
    clf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")

    # 5. Evaluate and save results
    pred_train = clf.predict(X_train)
    pred_val   = clf.predict(X_val)
    pred_test  = clf.predict(X_test)

    _report("Train", y_train, pred_train)
    _report("Val",   y_val,   pred_val)
    _report("Test",  y_test,  pred_test)

    _save_results(dataset_dir, train_limit, val_limit, test_limit,
                  y_train, pred_train, y_val, pred_val, y_test, pred_test)

    print("\n" + "=" * 60)
    return clf, model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Model 1 pipeline — CNN from scratch + K-means')
    parser.add_argument('--dataset',     default='dataset')
    parser.add_argument('--train_limit', type=int, default=TRAIN_LIMIT,
                        help=f'Max images/class from train/ (default {TRAIN_LIMIT})')
    parser.add_argument('--val_limit',   type=int, default=VAL_LIMIT,
                        help=f'Max images/class from val/   (default {VAL_LIMIT})')
    parser.add_argument('--test_limit',  type=int, default=TEST_LIMIT,
                        help=f'Max images/class from test/  (default {TEST_LIMIT})')
    parser.add_argument('--no_cache',    action='store_true',
                        help='Re-extract features even if cache exists')
    args = parser.parse_args()

    run(dataset_dir  = args.dataset,
        train_limit  = args.train_limit,
        val_limit    = args.val_limit,
        test_limit   = args.test_limit,
        use_cache    = not args.no_cache)
