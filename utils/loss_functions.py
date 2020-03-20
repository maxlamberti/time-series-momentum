import numpy as np
import keras.backend as K


def sharpe_loss(target_data, size_prediction, sigma_tgt=0.15):
    """Sharpe ratio inspired loss function."""

    # load parameters
    next_period_return = K.reshape(target_data[:, 0], (-1, 1))
    sigma_t = K.reshape(target_data[:, 1], (-1, 1))

    # calc sharpe
    ret = size_prediction * sigma_tgt * next_period_return / (np.sqrt(252) * sigma_t)
    mean_ret = K.mean(ret)
    mean_ret2 = K.mean(ret ** 2)
    sharpe = np.sqrt(252) * mean_ret / K.sqrt(mean_ret2 - (mean_ret ** 2))

    return -sharpe


def return_loss(target_data, size_prediction, sigma_tgt=0.15):
    """Loss function driven by mean returns."""

    # load parameters
    next_period_return = K.reshape(target_data[:, 0], (-1, 1))
    sigma_t = K.reshape(target_data[:, 1], (-1, 1))

    # calc mean_ret
    ret = size_prediction * sigma_tgt * next_period_return / (np.sqrt(252) * sigma_t)
    mean_ret = K.mean(ret)

    return -mean_ret


def sign_loss(target_data, size_prediction, sigma_tgt=0.15):
    """Loss function driven by mean returns."""

    # load parameters
    next_period_return = target_data[:, 0]
    sigma_t = target_data[:, 1]
    sign = target_data[:, 2]

    # calc mean_ret
    ret = size_prediction * sigma_tgt * next_period_return / (np.sqrt(252) * sigma_t)
    mean_ret = K.mean(ret)
    mean_ret = K.sum(size_prediction == sign)

    return -mean_ret


def sharpe_loss_dummy(target_data, size_prediction, sigma_tgt=0.15):
    """Sharpe ratio inspired loss function."""

    # load parameters
    next_period_return = target_data[:, 0]
    sigma_t = target_data[:, 1]

    # calc sharpe
    ret = size_prediction * sigma_tgt * next_period_return / (np.sqrt(252) * sigma_t)
    mean_ret = np.mean(ret)
    mean_ret2 = np.mean(ret ** 2)
    sharpe = ((1 + mean_ret)**252 - 1) / np.sqrt(252 * (mean_ret2 - (mean_ret ** 2)))

    return -sharpe


def return_loss_dummy(target_data, size_prediction, sigma_tgt=0.15):
    """Loss function driven by mean returns."""

    # load parameters
    next_period_return = target_data[:, 0]
    sigma_t = target_data[:, 1]

    # calc mean_ret
    ret = size_prediction * sigma_tgt * next_period_return / (np.sqrt(252) * sigma_t)
    mean_ret = np.mean(ret)
    mean_ret = ((1 + mean_ret) ** 252 - 1)

    return -mean_ret
