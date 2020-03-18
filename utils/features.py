import numpy as np


def calc_normalized_period_returns(daily_returns, daily_std, period):
    period = int(period)
    return daily_returns.rolling(period).sum() / (np.sqrt(period) * daily_std)


def calc_macd_features(price, short_period, long_period):
    short_ma = price.ewm(span=short_period, min_periods=short_period).mean()
    long_ma = price.ewm(span=long_period, min_periods=long_period).mean()
    ewmstd_63 = price.ewm(span=63).std()
    macd = short_ma - long_ma
    q = macd / ewmstd_63
    z = q / q.ewm(span=252, min_periods=252).std()

    return z


def construct_features_single_asset(df, ewmastd_span=60, inplace=False, asset_label=None):

    if not inplace:
        df = df.copy()

    if asset_label is not None:
        df['Asset'] = asset_label

    df['Returns_Daily'] = np.log(df['Settle']).diff()
    df['Next_Returns_Daily'] = df['Returns_Daily'].shift(-1)
    df['Sigma'] = df['Returns_Daily'].ewm(span=ewmastd_span, min_periods=ewmastd_span).std()
    df['Norm_Returns_Daily'] = df['Returns_Daily'] / df['Sigma']
    df['Norm_Returns_Monthly'] = calc_normalized_period_returns(df['Returns_Daily'], df['Sigma'], 252 / 12)
    df['Norm_Returns_Quarterly'] = calc_normalized_period_returns(df['Returns_Daily'], df['Sigma'], 252 / 3)
    df['Norm_Returns_Semiannually'] = calc_normalized_period_returns(df['Returns_Daily'], df['Sigma'], 252 / 2)
    df['Norm_Returns_Annually'] = calc_normalized_period_returns(df['Returns_Daily'], df['Sigma'], 252)
    df['MACD_8_24'] = calc_macd_features(df['Settle'], 8, 24)
    df['MACD_16_48'] = calc_macd_features(df['Settle'], 16, 48)
    df['MACD_32_96'] = calc_macd_features(df['Settle'], 32, 96)

    return df


def construct_features_batch(df_map):

    for asset, df in df_map.items():
        construct_features_single_asset(df, inplace=True, asset_label=asset)

    return df_map



