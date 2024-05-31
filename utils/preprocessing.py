import numpy as np
import pandas as pd
from numpy.typing import NDArray


def exponential_moving_standardize(data: NDArray, factor_new: float = 0.001, init_block_size=None,  eps: float = 1e-4):
    """
    Perform exponential moving standardization.
    :param data: (n_channels, n_times)
    :param factor_new:
    :param init_block_size: Standardize data before to this index with regular standardization.
    :param eps: Stabilizer for division by zero variance.
    :return: Standardized data.
    """
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(data[0:init_block_size], axis=i_time_axis, keepdims=True)
        init_std = np.std(data[0:init_block_size], axis=i_time_axis, keepdims=True)
        init_block_standardized = (data[0:init_block_size] - init_mean) / np.maximum(
            eps, init_std
        )
        standardized[0:init_block_size] = init_block_standardized
    return standardized.T
