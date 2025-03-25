import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def temperatur_cleaner(df_temperatur):
    df_temperatur['hour'] = df_temperatur['MESS_DATUM'].str[8:].astype(int)
    df_temperatur.rename(columns={'TT_TU': 'temperatur', 'RF_TU': 'relativeFeuchte'}, inplace=True)
    df_temperatur.drop(columns=['MESS_DATUM'], inplace=True)
    df_temperatur = df_temperatur[['STATIONS_ID', 'temperatur', 'relativeFeuchte', 'date', 'hour']]
    df_temperatur = df_temperatur.drop_duplicates()

    return df_temperatur


def wind_cleaner(df_wind):
    df_wind['hour'] = df_wind['MESS_DATUM'].str[8:].astype(int)
    df_wind.rename(columns={'   D': 'windRichtung', '   F': 'windGeschwindigkeit'}, inplace=True)
    df_wind.drop(columns=['MESS_DATUM'], inplace=True)
    df_wind = df_wind[['STATIONS_ID', 'windGeschwindigkeit', 'date', 'hour']]
    df_wind = df_wind.drop_duplicates()

    return df_wind


def sun_cleaner(df_sun):
    df_sun['hour'] = df_sun['MESS_DATUM'].str[8:].astype(int)
    df_sun.rename(columns={'SD_SO': 'sonnenscheinDauer'}, inplace=True)
    df_sun.drop(columns=['MESS_DATUM'], inplace=True)
    df_sun = df_sun[['STATIONS_ID', 'sonnenscheinDauer', 'date', 'hour']]
    df_sun = df_sun.drop_duplicates()

    return df_sun


def precipitation_cleaner(df_precipitation):
    df_precipitation['hour'] = df_precipitation['MESS_DATUM'].str[8:].astype(int)
    df_precipitation.rename(columns={'  R1': 'regenSchnee', 'RS_IND': 'niederschlagTyp'}, inplace=True)
    df_precipitation.drop(columns=['MESS_DATUM'], inplace=True)
    df_precipitation = df_precipitation[['STATIONS_ID', 'regenSchnee', 'date', 'hour']]
    df_precipitation = df_precipitation.drop_duplicates()

    return df_precipitation


def createMetrics(df, col, hourly=True):
    if hourly:
        df_agg = df.groupby(['date', 'hour']).agg(
            min=(col, 'min'),
            max=(col, 'max'),
            avg=(col, 'mean')
        ).reset_index()
    else:
        df_agg = df.groupby(['date']).agg(
            min=(col, 'min'),
            max=(col, 'max'),
            avg=(col, 'mean')
        ).reset_index()
    df_agg.rename(columns={'min': f'min_{col}', 'max': f'max_{col}', 'avg': f'avg_{col}'}, inplace=True)
    return df_agg


def preprocessHolidays(df, threshold):
    df['count'] = df.groupby(['Datum'])['Stadt'].transform('count')
    df['major'] = 0
    df.loc[(df['count'] > threshold), 'major'] = 1
    return df


def mergeWeatherData(dfList:list, cols:list):
    df_combined = pd.merge(dfList[0],dfList[1], how='left', on=cols)
    for df in dfList[2:]:
        df_combined = pd.merge(df_combined, df, how='left', on=cols)
    return df_combined


def cleanEnergy(df_energy):
    df_energy = df_energy[df_energy['Gesamt (Netzlast) [MWh] Berechnete Auflösungen'] != '-'].reset_index(drop=True)
    df_energy['load'] = (df_energy['Gesamt (Netzlast) [MWh] Berechnete Auflösungen'].str.replace('.', '', regex=False)
        .str.replace(',', '.',regex=False).astype(float))/1000
    df_energy['dateTime'] = pd.to_datetime(df_energy['Datum von'], format="%d.%m.%Y %H:%M")
    df_energy = df_energy[['dateTime', 'load']]
    return df_energy


def cet_utc(df_weather, df):

    df_weather['dateTime'] = df_weather['date'] + pd.to_timedelta(df_weather['hour'], unit='h')
    df_weather['dateTime'] = df_weather['dateTime'].dt.tz_localize('UTC')
    df_weather['dateTime'] = df_weather['dateTime'].dt.tz_convert('Europe/Paris')
    df_weather['dateTime'] = df_weather['dateTime'].apply(lambda x: x.replace(tzinfo=None))
    df_weather = df_weather.drop(['date', 'hour'], axis=1)

    df_input = pd.merge(df, df_weather, how='left', on='dateTime')
    df_input = df_input.set_index('dateTime')

    return df_input


def preprocessWeatherForecast(df_weatherForecast, df_tmp, last_index):
    df_weatherForecast = df_weatherForecast[df_weatherForecast['date'] > last_index].reset_index(drop=True)

    #from GMT to CET
    # TODO: CHECK ob mit precipitation auch bei energy dann klappt
    df_weatherForecast['date'] = df_weatherForecast['date'] + pd.Timedelta(hours=1)
    df_weatherForecast = df_weatherForecast.rename(columns={'temperature_2m': 'temperatur',
                                                      'relative_humidity_2m': 'relativeFeuchte',
                                                      'wind_speed_10m': 'windGeschwindigkeit',
                                                      'precipitation': 'regenSchnee'
                                                      })
    # TODO: in Bericht aufnehmen, wieso aggregierte Forecasts
    df_weatherForecast = df_weatherForecast.groupby(['date']).agg(
        min_temperatur=('temperatur', 'min'),
        min_relativeFeuchte=('relativeFeuchte', 'min'),
        min_windGeschwindigkeit=('windGeschwindigkeit', 'min'),
        min_regenSchnee=('regenSchnee', 'min'),
        avg_temperatur=('temperatur', 'mean'),
        avg_relativeFeuchte=('relativeFeuchte', 'mean'),
        avg_windGeschwindigkeit=('windGeschwindigkeit', 'mean'),
        avg_regenSchnee=('regenSchnee', 'mean'),
        max_temperatur=('temperatur', 'max'),
        max_relativeFeuchte=('relativeFeuchte', 'max'),
        max_windGeschwindigkeit=('windGeschwindigkeit', 'max'),
        max_regenSchnee=('regenSchnee', 'max')
    ).reset_index()

    df_weatherForecast = df_weatherForecast.merge(df_tmp, how='left', left_on='date', right_on='dateTime')
    df_weatherForecast = df_weatherForecast.drop(columns='dateTime')
    df_weatherForecast = df_weatherForecast.set_index('date')

    return df_weatherForecast


def preprocessBike(df):
    df['date'] = pd.to_datetime(df['date'], format="%m/%d/%Y")
    return df


def preprocessBikeHourly(df, start_date="2015-01-01"):
    df['date'] = pd.to_datetime(df['date'], format="%m/%d/%Y")
    df['hour'] = df.groupby('date').cumcount()
    df['dateTime'] = df.apply(lambda row: row['date'] + pd.Timedelta(hours=row['hour']), axis=1)

    df = df[df['dateTime'] >= pd.to_datetime(start_date)]

    df = df[['dateTime', 'date', 'hour', 'bike_count']]

    return df


def split_data(df_input, is_forecasting=True, forecast_horizon=14, train_start=None, train_end=None, test_start=None,
               test_end=None):
    data = df_input.sort_index()

    if train_start and train_end and test_start and test_end:
        # Manual date split
        train_data = data[train_start:train_end]
        test_data = data[test_start:test_end]

    else:
        # Automatic split with forecast horizon
        if is_forecasting:
            test_data = data[-forecast_horizon:]
            train_data = data[:-forecast_horizon]
        else:
            raise ValueError("Please provide train and test date ranges when not forecasting")

    return train_data, test_data


def split_data_xgboost(df_input, is_forecasting=False, forecast_horizon=7*24,
                       train_start='2022-01-01', train_end='2022-12-31',
                       val_start='2023-01-01', val_end='2023-12-31',
                       test_start='2024-01-01', test_end='2024-12-31'):
    if is_forecasting:
        # Find the last date with actual data
        last_actual_date = df_input[df_input['bike_count'].notna()].index[-1]

        # Get the next 7 days after the last actual date for testing
        test_df = df_input[
            (df_input.index > last_actual_date) &
            (df_input.index <= last_actual_date + pd.Timedelta(days=forecast_horizon))
            ].copy()

        # Get all data up to the last actual date
        historical_data = df_input[df_input.index <= last_actual_date].copy()

        # Split historical data into train and validation
        split_date = historical_data.index[len(historical_data) // 2].strftime('%Y-%m-%d')
        train_df = historical_data[:split_date]
        val_df = historical_data[split_date:]

        return train_df, val_df, test_df
    else:
        # Regular split for evaluation
        train_df = df_input[train_start:train_end].copy()
        val_df = df_input[val_start:val_end].copy()
        test_df = df_input[test_start:test_end].copy()

        return train_df, val_df, test_df

def remove_outliers(df, column, method='iqr'):
    # Konvertiere die Spalte in numerische Werte und entferne nicht-numerische
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])  # Entferne Zeilen mit NaN-Werten in der angegebenen Spalte

    if method == 'iqr':
        # Berechnung des IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        # lower_bound = Q1 - 2 * IQR
        upper_bound = Q3 + 0.7 * IQR

        # Filtere die Ausreißer aus
        #condition = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        condition = (df[column] <= upper_bound)

    elif method == 'zscore':
        # Berechnung der Z-Scores
        mean = df[column].mean()
        std = df[column].std()
        df['zscore'] = (df[column] - mean) / std

        # Filtere basierend auf einem Z-Score-Grenzwert (z. B. 3)
        condition = abs(df['zscore']) <= 3
        df = df.drop(columns=['zscore'])  # Entferne die zscore-Spalte nach der Filterung

    elif method == 'quantile':
        # Bereinigung basierend auf den Quantilen
        lower_bound = df[column].quantile(0.01)  # 1. Perzentil
        upper_bound = df[column].quantile(0.99)  # 99. Perzentil
        condition = (df[column] >= lower_bound) & (df[column] <= upper_bound)

    else:
        raise ValueError("Methode nicht erkannt. Wählen Sie 'iqr', 'zscore' oder 'quantile'.")

    # Behalte Zeilen mit holiday_any == 1 oder bridge_day_any == 1
    # condition |= (df.get('holiday_any', 0) == 1) | (df.get('bridge_day_any', 0) == 1)
    condition |= (df.get('holiday', 0) == 1) | (df.get('bridge_day', 0) == 1)

    # Anwenden der Bedingung auf das DataFrame
    df_cleaned = df[condition]

    return df_cleaned


def validate_and_fix_quantiles(df, quantile_cols):
    """
    Validates and fixes quantile columns in a DataFrame by ensuring
    lower quantiles are not greater than higher quantiles.
    """
    df = df.copy()  # Arbeitskopie erstellen

    # Iteriere durch die Quantil-Paare und korrigiere, falls nötig
    for i in range(len(quantile_cols) - 1):
        lower_col = quantile_cols[i]
        upper_col = quantile_cols[i + 1]

        # Switch, wenn das niedrigere Quantil größer ist als das höhere
        mask = df[lower_col] > df[upper_col]
        df.loc[mask, [lower_col, upper_col]] = df.loc[mask, [upper_col, lower_col]].values

    return df

