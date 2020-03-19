import numpy as np
import pandas as pd
from scipy import stats


def calc_backtest(daily_return_series, excess_return_series, weights):

    # construct backtest dataframe
    daily_return_series = daily_return_series.rename('Returns')
    weights = weights.rename('Weights')
    excess_return_series = excess_return_series.rename('Excess_Returns')
    backtest_series = pd.concat([daily_return_series, excess_return_series, weights], axis=1)
    backtest_series.Weights.ffill(inplace=True)
    backtest_series.dropna(inplace=True)
    backtest_series['Strategy_Returns'] = backtest_series['Weights'] * backtest_series['Returns']
    backtest_series['Strategy_Excess_Returns'] = backtest_series['Weights'] * backtest_series['Excess_Returns']

    # basic return stats
    num_data_per_year = backtest_series.Returns.resample('Y').count().mode().values[-1]
    num_data_total = backtest_series.shape[0]
    mean_ret_annual = ((1 + backtest_series['Strategy_Returns']).prod()) ** (num_data_per_year / num_data_total) - 1
    mean_exc_ret_annual = ((1 + backtest_series['Strategy_Excess_Returns']).prod()) ** (
                num_data_per_year / num_data_total) - 1
    vola_annual = np.std(backtest_series['Strategy_Returns']) * np.sqrt(num_data_per_year)
    sharpe = mean_exc_ret_annual / vola_annual

    # drawdown stats
    position = (1 + backtest_series['Strategy_Returns']).cumprod()
    cummax_position = position.cummax()
    drawdown = position - cummax_position
    is_in_drawdown = drawdown < 0
    longest_drawdown_duration = (~is_in_drawdown).cumsum()[is_in_drawdown].value_counts().max() / num_data_per_year
    max_drawdown = (drawdown / cummax_position).min()

    results = pd.Series({
        'E[Return]': mean_ret_annual,
        'E[Excess Return]': mean_exc_ret_annual,
        'Std[Return]': vola_annual,
        'Skew[Return]': stats.skew(backtest_series['Strategy_Returns']),
        'Exc.Kurtosis[Return]': stats.kurtosis(backtest_series['Strategy_Returns']),
        'Sharpe': sharpe,
        'Max_Drawdown': max_drawdown,
        'Max_Drawdown_Duration (Years)': longest_drawdown_duration
    })

    return results
