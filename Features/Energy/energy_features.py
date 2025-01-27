import pandas as pd
import numpy as np


# Calender Features
def calculate_hours_of_daylight(day_of_year, latitude):
    # Check Paper
    # Berechne den Neigungswinkel der Erde (kt)
    k_t = 0.4102 * np.sin(2 * np.pi * (day_of_year - 80.25) / 365)

    # Berechnung der Stunden des Tageslichts (HDL)
    HDL_t = 7.722 * np.arccos(-np.tan(2 * np.pi * latitude / 360) * np.tan(k_t))

    return HDL_t


def setCalenderFeatureEnergy(df, df_holidays):
    df_holidays['Datum'] = pd.to_datetime(df_holidays['Datum'])

    df['holiday'] = df.index.normalize().isin(df_holidays['Datum']).astype(int)

    df['bridge_day'] = 0
    df['prev_day_holiday'] = (df.index.normalize() - pd.Timedelta(days=1)).isin(df_holidays['Datum']).astype(int)
    df['next_day_holiday'] = (df.index.normalize() + pd.Timedelta(days=1)).isin(df_holidays['Datum']).astype(int)
    df['weekday'] = df.index.dayofweek  # 0 = Montag, ..., 6 = Sonntag

    df.loc[
        (df['holiday'] == 0) &  # Kein Feiertag
        (df['weekday'] < 5) &  # Kein Wochenende
        ((df['prev_day_holiday'] == 1) | (df['next_day_holiday'] == 1)),  # Feiertag davor oder danach
        'bridge_day'
    ] = 1

    # Zusätzliche Features
    df['hour'] = df.index.hour
    df['weekday'] = df.index.dayofweek
    df['month'] = df.index.month
    df['dayOfYear'] = df.index.dayofyear
    df['is_weekend'] = (df['weekday'] > 4).astype(int)

    # Zeitlicher Trend (in Stunden)
    df['time_trend_hours'] = (df.index - df.index[0]).total_seconds() / 3600

    # Tageslichtstunden (mittlere Breitengrade, z. B. Deutschland)
    df['HDD'] = df['dayOfYear'].apply(lambda x: calculate_hours_of_daylight(x, 50))

    # Unnötige Spalten entfernen
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


# Holiday Features
def add_xmas_holiday_features_energy(df):
    """
    Create aggregated holiday features for energy data with hourly granularity.
    """
    # Christmas-New Year Period (Dec 23 - Jan 2)
    df['is_christmas_newyear_period'] = (
            ((df.index.month == 12) & (df.index.day >= 23)) |
            ((df.index.month == 1) & (df.index.day <= 2))
    ).astype(int)

    # Core Holiday Days (Dec 24-26, Dec 31-Jan 1)
    df['is_core_holiday'] = (
            ((df.index.month == 12) & (df.index.day.isin([24, 25, 26]))) |
            ((df.index.month == 12) & (df.index.day == 31)) |
            ((df.index.month == 1) & (df.index.day == 1))
    ).astype(int)

    # # Add interactions with key hours
    for hour in [6, 12, 16, 20]:  # Important hours
         df[f'holiday_period_hour_{hour}'] = df['is_christmas_newyear_period'] * (df.index.hour == hour)
         df[f'core_holiday_hour_{hour}'] = df['is_core_holiday'] * (df.index.hour == hour)

    return df


# Load Features
def setEnergyLoadFeatures_Eval(df, start_date, end_date):
    train_mask = (df.index >= start_date) & (df.index <= end_date)

    # Calculate features only for training period
    df.loc[train_mask, 'rolling_avg_30d'] = df.loc[train_mask, 'load'].rolling(window=24 * 30, min_periods=1).mean()
    df.loc[train_mask, 'lag_1'] = df.loc[train_mask, 'load'].shift(1)
    df.loc[train_mask, 'lag_24'] = df.loc[train_mask, 'load'].shift(24)

    return df


def setEnergyLoadFeatures(df):
    df['rolling_avg_30d'] = df['load'].rolling(window=24 * 30, min_periods=1).mean()
    df['lag_1'] = df['load'].shift(1)
    df['lag_24'] = df['load'].shift(24)

    return df



# Special Features
def add_peak_specific_features(df):
    # Early morning period (2-5)
    df['early_morning'] = (df['hour'].between(2, 5)).astype(int)
    df['weekend_early_morning'] = ((df['is_weekend'] == 1) & df['early_morning']).astype(int)
    df = df.drop(columns=['early_morning'], errors='ignore')

    # Morning period (5-9) - stronger distinction between weekday/weekend
    df['weekday_early_morning_ramp'] = ((df['is_weekend'] == 0) &
                                        (df['hour'].between(5, 6))).astype(int)
    df['weekday_morning_ramp'] = ((df['is_weekend'] == 0) &
                                  (df['hour'].between(7, 9))).astype(int)
    df['weekend_morning_flat'] = ((df['is_weekend'] == 1) &
                                  (df['hour'].between(5, 9))).astype(int)

    # Midday period (10-14)
    df['midday_plateau'] = (df['hour'].between(10, 14)).astype(int)
    df['weekday_midday'] = ((df['is_weekend'] == 0) & df['midday_plateau']).astype(int)
    df['weekend_midday'] = ((df['is_weekend'] == 1) & df['midday_plateau']).astype(int)

    # Afternoon transition (14-18) - more granular periods
    df['early_afternoon'] = (df['hour'].between(14, 15)).astype(int)
    df['peak_afternoon'] = (df['hour'].between(16, 17)).astype(int)
    df['late_afternoon'] = (df['hour'] == 18).astype(int)

    # Combine with day type
    df['weekday_peak_afternoon'] = ((df['is_weekend'] == 0) & df['peak_afternoon']).astype(int)
    df['weekend_peak_afternoon'] = ((df['is_weekend'] == 1) & df['peak_afternoon']).astype(int)

    # Evening decline (19-23)
    df['evening_decline'] = (df['hour'].between(19, 23)).astype(int)

    # Add transition indicators
    df['is_transition_hour'] = ((df['hour'].isin([7, 8, 14, 15, 18, 19])) &
                                (df['is_weekend'] == 0)).astype(int)

    # Add interaction with weather (temperature affects peak patterns)
    if 'avg_temperatur' in df.columns:
        df['afternoon_temp_effect'] = df['peak_afternoon'] * df['avg_temperatur']
        df['morning_temp_effect'] = df['weekday_morning_ramp'] * df['avg_temperatur']
        # df['early_morning_temp_effect'] = df['early_morning'] * df['avg_temperatur']

    return df
