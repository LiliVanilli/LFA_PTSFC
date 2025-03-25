import pandas as pd
import numpy as np


# Calender Features
def calculate_hours_of_daylight(day_of_year, latitude):
    # Berechne den Neigungswinkel der Erde (kt)
    k_t = 0.4102 * np.sin(2 * np.pi * (day_of_year - 80.25) / 365)

    # Berechnung der Stunden des Tageslichts (HDL)
    HDL_t = 7.722 * np.arccos(-np.tan(2 * np.pi * latitude / 360) * np.tan(k_t))

    return HDL_t


def setCalenderFeatureBike(df, df_holidays):
    df_holidays['Date'] = pd.to_datetime(df_holidays['Date'], format="%d.%m.%y")
    df = df.reset_index()
    df['Date'] = df['date'].dt.date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.merge(df_holidays, how='left', on='Date')
    df['holiday'] = 0
    df.loc[(df['Holiday'].notna()), 'holiday'] = 1

    # Füge Brückentage hinzu
    df['bridge_day'] = 0
    df['weekday'] = df['Date'].dt.dayofweek  # 0 = Montag, ..., 6 = Sonntag

    # Brückentage definieren:
    # Bedingung: Ein Arbeitstag zwischen einem Feiertag und einem Wochenende oder einem weiteren Feiertag
    df.loc[
        (df['holiday'] == 0) &  # Kein Feiertag
        (df['weekday'] < 5) &  # Kein Wochenende
        (
            (df['Date'] - pd.Timedelta(days=1)).isin(df['Date'][df['holiday'] == 1]) |  # Vortag ist Feiertag
            (df['Date'] + pd.Timedelta(days=1)).isin(df['Date'][df['holiday'] == 1])    # Folgetag ist Feiertag
        ),
        'bridge_day'
    ] = 1

    df = df.set_index('date')

    df['weekday'] = df.index.dayofweek
    df['month'] = df.index.month

    # Create seasonal indicators: NEW
    # df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    # df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    # df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    # df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)


    df['dayOfYear'] = df.index.dayofyear
    df['is_weekend'] = 0
    df.loc[(df['weekday'] > 4), 'is_weekend'] = 1
    # Modellierung zeitlicher Trend, hilft Drift in Modell zu erkennen
    #df['time_trend_hours'] = (df.index - df.index[0]).total_seconds() / 3600
    df['HDD'] = df['dayOfYear'].apply(lambda x: calculate_hours_of_daylight(x, 50))  # weil Mitte DE ca
    df.drop(['Holiday', 'Weekday', 'Minor'], axis=1, inplace=True)

    return df


def add_fourier_features(df, period, n_terms, prefix):
    t = np.arange(len(df)) / period * 2 * np.pi  # Normalisiere Zeit
    for i in range(1, n_terms + 1):
        df[f'{prefix}_sin_{i}'] = np.sin(i * t)
        df[f'{prefix}_cos_{i}'] = np.cos(i * t)
    return df


# Weather Features
def setWeatherFeaturesBike(df):
    # 24h
    df['temp_change_24h'] = df['avg_temperatur'] - df['avg_temperatur'].shift(1)
    # 1w
    df['temp_change_1w'] = df['avg_temperatur'] - df['avg_temperatur'].shift(7)
    return df


# Holiday Features
def add_xmas_holiday_features_bike_daily(df):
    dummy_dict = {}

    # Create day variable if needed
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df['day'] = df.index.day
    else:
        df['day'] = pd.to_datetime(df.index).day

    # Christmas-New Year Period (Dec 23 - Jan 2)
    dummy_dict['holiday_period'] = (
            ((df['month'] == 12) & (df['day'] >= 23)) |
            ((df['month'] == 1) & (df['day'] <= 2))
    ).astype(int)

    # Core Holiday Days (Dec 24-26, Dec 31-Jan 1)
    dummy_dict['core_holiday'] = (
            ((df['month'] == 12) & (df['day'].isin([24, 25, 26]))) |
            ((df['month'] == 12) & (df['day'] == 31)) |
            ((df['month'] == 1) & (df['day'] == 1))
    ).astype(int)


    # Drop the temporary day column if we created it
    if 'day' in df.columns:
        df.drop('day', axis=1, inplace=True)

    # Create DataFrame with all dummies and combine with original
    dummy_df = pd.DataFrame(dummy_dict, index=df.index)
    return pd.concat([df, dummy_df], axis=1)


# Lag Features
def setBikeLoadFeatures_Eval(df, start_date, end_date):
    train_mask = (df.index >= start_date) & (df.index <= end_date)

    # Calculate features only for training period
    df.loc[train_mask, 'rolling_avg_30d'] = df.loc[train_mask, 'bike_count'].rolling(window=30, min_periods=1).mean()
    df.loc[train_mask, 'lag_1'] = df.loc[train_mask, 'bike_count'].shift(1)
    df.loc[train_mask, 'lag_1w'] = df.loc[train_mask, 'bike_count'].shift(7)

    return df

def setBikeLoadFeatures(df):
    df['rolling_avg_30d'] = df['bike_count'].rolling(window=30, min_periods=1).mean()
    df['lag_1'] = df['bike_count'].shift(1)
    df['lag_1w'] = df['bike_count'].shift(7)

    return df