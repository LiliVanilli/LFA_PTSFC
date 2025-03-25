import pandas as pd
import numpy as np


# Calender Features
def calculate_hours_of_daylight(day_of_year, latitude):
    # Berechne den Neigungswinkel der Erde (kt)
    k_t = 0.4102 * np.sin(2 * np.pi * (day_of_year - 80.25) / 365)

    # Berechnung der Stunden des Tageslichts (HDL)
    HDL_t = 7.722 * np.arccos(-np.tan(2 * np.pi * latitude / 360) * np.tan(k_t))

    return HDL_t


def setCalenderFeatureBikeHOURLY(df, df_holidays):
    df_holidays['Datum'] = pd.to_datetime(df_holidays['Datum'])

    df['holiday'] = df.index.normalize().isin(df_holidays['Datum']).astype(int)

    df['bridge_day'] = 0
    df['prev_day_holiday'] = (df.index.normalize() - pd.Timedelta(days=1)).isin(df_holidays['Datum']).astype(int)
    df['next_day_holiday'] = (df.index.normalize() + pd.Timedelta(days=1)).isin(df_holidays['Datum']).astype(int)
    df['weekday'] = df.index.dayofweek

    df.loc[
        (df['holiday'] == 0) &
        (df['weekday'] < 5) &
        ((df['prev_day_holiday'] == 1) | (df['next_day_holiday'] == 1)),
        'bridge_day'
    ] = 1

    # Basic time features
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek
    df['month'] = df.index.month
    df['dayOfYear'] = df.index.dayofyear
    df['is_weekend'] = (df['weekday'] > 4).astype(int)

    # Create combined weekend/holiday flag
    # df['is_weekend_or_holiday'] = ((df['is_weekend'] == 1) | (df['holiday'] == 1)).astype(int)

    # Add weekend/holiday morning features
    # for hour in [6, 7, 8, 9]:
    #    df[f'weekend_holiday_morning_{hour}'] = ((df['is_weekend_or_holiday'] == 1) & (df['hour'] == hour)).astype(int)

    # Remaining features
    # df['time_trend_hours'] = (df.index - df.index[0]).total_seconds() / 3600
    df['HDD'] = df['dayOfYear'].apply(lambda x: calculate_hours_of_daylight(x, 50))

    # Clean up
    df.drop(['prev_day_holiday', 'next_day_holiday'], axis=1, inplace=True)

    return df


def add_fourier_features(df, period, n_terms, prefix):
    t = np.arange(len(df)) / period * 2 * np.pi  # Normalisiere Zeit
    for i in range(1, n_terms + 1):
        df[f'{prefix}_sin_{i}'] = np.sin(i * t)
        df[f'{prefix}_cos_{i}'] = np.cos(i * t)
    return df


# Extra Weather Features
def setWeatherFeatures(df):
    # 24h
    df['temp_change_24h'] = df['avg_temperatur'] - df['avg_temperatur'].shift(24)
    # 1w
    df['temp_change_1w'] = df['avg_temperatur'] - df['avg_temperatur'].shift(168)
    return df


# Load Features
def setBikeHourlyLoadFeatures(df):

    df['rolling_avg_30d'] = df['bike_count'].rolling(window=24 * 30, min_periods=1).mean()

    df['lag_1'] = df['bike_count'].shift(1)
    df['lag_24'] = df['bike_count'].shift(24)
    # Previous week, same hour
    df['lag_168'] = df['bike_count'].shift(168)

    return df


def setBikeHourlyLoadFeaturesEval(df, start_date, end_date):

    train_mask = (df.index >= start_date) & (df.index <= end_date)

    df.loc[train_mask, 'rolling_avg_30d'] = df.loc[train_mask, 'bike_count'].rolling(window=24 * 30, min_periods=1).mean()
    df.loc[train_mask, 'lag_1'] = df.loc[train_mask, 'bike_count'].shift(1)
    df.loc[train_mask, 'lag_24'] = df.loc[train_mask, 'bike_count'].shift(24)
    df.loc[train_mask, 'lag_168'] = df.loc[train_mask, 'bike_count'].shift(168)
    return df


