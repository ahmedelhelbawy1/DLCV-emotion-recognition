import numpy as np

from model1.conv_layer import ConvLayer, get_predefined_filters
from model1.pooling_layer import PoolingLayer
from model1.activation import relu
from model1.flatten import flatten
from model1.kmeans_classifier import KMeansClassifier


class Model1Pipeline:
    def __init__(self):
        # Conv 1 takes RGB input => depth 3
        self.conv1 = ConvLayer(
            num_filters=5,
            filter_size=3,
            input_depth=3,
            filters=get_predefined_filters(input_depth=3)
        )

        # Conv 2 takes output of conv1 => depth 5
        self.conv2 = ConvLayer(
            num_filters=5,
            filter_size=3,
            input_depth=5,
            filters=get_predefined_filters(input_depth=5)
        )

        # Conv 3 takes output of conv2 => depth 5
        self.conv3 = ConvLayer(
            num_filters=5,
            filter_size=3,
            input_depth=5,
            filters=get_predefined_filters(input_depth=5)
        )

        self.pool = PoolingLayer()

    def forward(self, image):
        # ---- Block 1 ----
        x = self.conv1.forward(image)
        x = self.pool.forward(x)
        x = relu(x)

        # ---- Block 2 ----
        x = self.conv2.forward(x)
        x = self.pool.forward(x)
        x = relu(x)

        # ---- Block 3 ----
        x = self.conv3.forward(x)
        x = self.pool.forward(x)
        x = relu(x)

        # Flatten
        x = flatten(x)

        # Downsample to 128 features
        x = x[:128]

        return x


def run_demo():
    print("Running Model 1 Pipeline Demo...")

    image = np.random.rand(64, 64, 3)

    model = Model1Pipeline()

    features = model.forward(image)

    print("Feature vector shape:", features.shape)


if __name__ == "__main__":
    run_demo()