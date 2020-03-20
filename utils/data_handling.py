import pandas as pd


def merge_asset_data(asset_to_df_map, create_time_asset_index=True):
    "Merges multiple asset dataframes together."

    asset_dfs = [df for asset, df in asset_to_df_map.items()]
    combined_assets = pd.concat(asset_dfs, ignore_index=True)

    if create_time_asset_index:
        combined_assets.set_index(['Asset', 'Date'], inplace=True, drop=False)
        combined_assets.sort_index(inplace=True)
        combined_assets['Asset_Col'] = combined_assets['Asset']
        combined_assets['Date_Col'] = combined_assets['Date']
        del combined_assets['Date']
        del combined_assets['Asset']

    return combined_assets


def split_by_date(df, date_breakpoints):

    data_sets = []
    for idx, start in enumerate(date_breakpoints):
        if idx == len(date_breakpoints) - 1:
            break
        end = date_breakpoints[idx + 1]
        data_sets.append(df[(start <= df['Date_Col']) & (df['Date_Col'] < end)])

    return data_sets
