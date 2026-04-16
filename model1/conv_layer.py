import numpy as np


class ConvLayer:
    def __init__(self, num_filters=5, filter_size=3, input_depth=3, filters=None):
        """
        Convolution layer from scratch.
        Args:
            num_filters (int): number of filters
            filter_size (int): size of each filter (e.g. 3 for 3x3)
            input_depth (int): number of channels (3 for RGB)
            filters (np.ndarray): optional predefined filters
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_depth = input_depth

        if filters is None:
            self.filters = self._init_random_filters()
        else:
            self.filters = self._validate_filters(filters)

    def _init_random_filters(self):
        return np.random.randn(
            self.num_filters,
            self.filter_size,
            self.filter_size,
            self.input_depth
        ) * 0.1

    def _validate_filters(self, filters):
        filters = np.array(filters, dtype=np.float32)

        expected_shape = (
            self.num_filters,
            self.filter_size,
            self.filter_size,
            self.input_depth
        )

        if filters.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {filters.shape}")

        return filters

    def iterate_regions(self, image):
        h, w, d = image.shape

        out_h = h - self.filter_size + 1
        out_w = w - self.filter_size + 1

        for i in range(out_h):
            for j in range(out_w):
                region = image[i:i+self.filter_size, j:j+self.filter_size, :]
                yield region, i, j

    def forward(self, image):
        image = np.array(image, dtype=np.float32)

        h, w, d = image.shape
        out_h = h - self.filter_size + 1
        out_w = w - self.filter_size + 1

        output = np.zeros((out_h, out_w, self.num_filters))

        for region, i, j in self.iterate_regions(image):
            for f in range(self.num_filters):
                output[i, j, f] = np.sum(region * self.filters[f])

        return output


# ===== Required predefined filters (assignment) =====

def expand_filter_2d_to_3d(filter_2d, depth=3):
    filter_2d = np.array(filter_2d, dtype=np.float32)
    return np.stack([filter_2d] * depth, axis=-1)


def get_predefined_filters(input_depth=3):
    f1 = [[1, 1, 1],
          [1, 1, 1],
          [1, 1, 1]]

    f2 = [[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]

    f3 = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]

    f4 = [[-1, -2, -1],
          [0,  0,  0],
          [1,  2,  1]]

    f5 = [[0, -1, 0],
          [-1, 5, -1],
          [0, -1, 0]]

    filters = [
        expand_filter_2d_to_3d(f1, depth=input_depth),
        expand_filter_2d_to_3d(f2, depth=input_depth),
        expand_filter_2d_to_3d(f3, depth=input_depth),
        expand_filter_2d_to_3d(f4, depth=input_depth),
        expand_filter_2d_to_3d(f5, depth=input_depth),
    ]

    return np.array(filters, dtype=np.float32) 
