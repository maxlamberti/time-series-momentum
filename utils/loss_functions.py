import numpy as np


def sharpe_loss(target_data, size_prediction, sigma_tgt=0.15):
    """Sharpe ratio inspired loss function."""

    # load parameters
    next_period_return = target_data[:, 0]
    sigma_t = target_data[:, 1]

    # calc sharpe
    ret = size_prediction * sigma_tgt * next_period_return / sigma_t
    mean_ret = np.mean(ret)
    mean_ret2 = np.mean(ret ** 2)
    sharpe = mean_ret / np.sqrt(mean_ret2 - (mean_ret ** 2))

    return -sharpe


def return_loss(target_data, size_prediction, sigma_tgt=0.15):
    """Loss function driven by mean returns."""

    # load parameters
    next_period_return = target_data[:, 0]
    sigma_t = target_data[:, 1]

    # calc mean_ret
    ret = size_prediction * sigma_tgt * next_period_return / sigma_t
    mean_ret = np.mean(ret)

    return -mean_ret
