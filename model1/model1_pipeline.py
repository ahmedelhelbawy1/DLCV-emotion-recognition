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

IMAGE_SIZE      = 512
NUM_CLASSES     = 4
NUM_FILTERS     = 5
FILTER_SIZE     = 3
OUTPUT_DIM      = 128
SEED            = 42
TRAIN_LIMIT     = 70
VAL_LIMIT       = 20
TEST_LIMIT      = 10

CLASS_NAMES  = ['angry', 'happy', 'sad', 'surprise']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

PREDEFINED_FILTERS = get_predefined_filters(input_depth=3)

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'model1_features'
)


class Model1Pipeline:
    def __init__(self):
        self.conv1 = ConvLayer(
            num_filters=NUM_FILTERS,
            filter_size=FILTER_SIZE,
            input_depth=3,
            filters=get_predefined_filters(input_depth=3)
        )
        self.conv2 = ConvLayer(
            num_filters=NUM_FILTERS,
            filter_size=FILTER_SIZE,
            input_depth=NUM_FILTERS,
            filters=get_predefined_filters(input_depth=NUM_FILTERS)
        )
        self.conv3 = ConvLayer(
            num_filters=NUM_FILTERS,
            filter_size=FILTER_SIZE,
            input_depth=NUM_FILTERS,
            filters=get_predefined_filters(input_depth=NUM_FILTERS)
        )
        self.pool      = PoolingLayer(pool_size=2, mode='MAX')
        self.extractor = FeatureExtractor(output_dim=OUTPUT_DIM, seed=SEED)

    def forward(self, image):
        x = self.conv1.forward(image)
        x = self.pool.forward(x)
        x = relu(x)

        x = self.conv2.forward(x)
        x = self.pool.forward(x)
        x = relu(x)

        x = self.conv3.forward(x)
        x = self.pool.forward(x)
        x = relu(x)

        x = flatten(x)
        x = self.extractor.transform(x)
        return x


def _load_split(split_dir, max_per_class=None):
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
    img = Image.open(path).convert('RGB')
    return np.asarray(img, dtype=np.float32) / 255.0


def _cache_path(split_name, limit):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tag = f"max{limit}" if limit else "full"
    return os.path.join(_CACHE_DIR, f"features_{split_name}_{tag}.npy"), \
           os.path.join(_CACHE_DIR, f"labels_{split_name}_{tag}.npy")


def extract_features(paths, labels, model, split_name,
                     limit=None, use_cache=True):
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


def _report(split_name, y_true, y_pred, file=None):
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
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'reports'
    )
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, 'model1_results.txt')

    with open(out_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Model 1 Results - CNN from scratch + K-means\n")
        f.write(f"per class  train={train_limit}  val={val_limit}  test={test_limit}\n")
        f.write(f"total images: {(train_limit+val_limit+test_limit)*NUM_CLASSES}\n")
        f.write("=" * 60 + "\n")
        _report("Train", y_train, pred_train, file=f)
        _report("Val",   y_val,   pred_val,   file=f)
        _report("Test",  y_test,  pred_test,  file=f)
        f.write("\n" + "=" * 60 + "\n")

    print(f"\nResults saved to {out_path}")


def run(dataset_dir='dataset',
        train_limit=TRAIN_LIMIT,
        val_limit=VAL_LIMIT,
        test_limit=TEST_LIMIT,
        use_cache=True):
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

    print("\nLoading dataset paths ...")
    train_paths, train_labels = _load_split(train_dir, train_limit)
    val_paths,   val_labels   = _load_split(val_dir,   val_limit)
    test_paths,  test_labels  = _load_split(test_dir,  test_limit)
    print(f"  train: {len(train_paths)}  "
          f"val: {len(val_paths)}  "
          f"test: {len(test_paths)}")

    model = Model1Pipeline()

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

    print("\nTraining K-means classifier ...")
    t0  = time.time()
    clf = KMeansClassifier(n_clusters=NUM_CLASSES, seed=SEED)
    clf.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")

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
    parser = argparse.ArgumentParser(description='Model 1 pipeline - CNN from scratch + K-means')
    parser.add_argument('--dataset',     default='dataset')
    parser.add_argument('--train_limit', type=int, default=TRAIN_LIMIT)
    parser.add_argument('--val_limit',   type=int, default=VAL_LIMIT)
    parser.add_argument('--test_limit',  type=int, default=TEST_LIMIT)
    parser.add_argument('--no_cache',    action='store_true')
    args = parser.parse_args()

    run(dataset_dir  = args.dataset,
        train_limit  = args.train_limit,
        val_limit    = args.val_limit,
        test_limit   = args.test_limit,
        use_cache    = not args.no_cache)
