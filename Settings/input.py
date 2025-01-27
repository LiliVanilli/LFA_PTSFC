import pandas as pd
import requests
from Input_Data.Weather_Data.Weather import *


def importBike():
    response = requests.get(getPaths('bikeURL'))
    rawdata = response.json()
    return pd.DataFrame(rawdata, columns=['date', 'bike_count'])

def importBikehourly():
    response = requests.get(getPaths('bikeURLhourly'))
    rawdata = response.json()
    return pd.DataFrame(rawdata, columns=['date', 'bike_count'])

def importEnergy():
    df = pd.read_csv(getPaths('energyPath'), sep=';')
    return df


def importWeatherStations():
    df = pd.read_excel(getPaths('weatherStationsPath'))
    df = df[df['relevant'] == 1].reset_index(drop=True)
    df['Stations_id'] = df['Stations_id'].astype(str).str.zfill(5)
    return df


def importHolidays():
    df_germany = pd.read_excel(getPaths('holidaysPath'))
    df_karlsruhe = pd.read_csv(getPaths('holidaysKarlsruhePath'), sep=';')
    return df_germany, df_karlsruhe


def importWeather(df_stations, dataTypes):
    getWeather(df_stations)
    combineData()
    combineHistoricRecent()
    weatherData = []
    for dataType in dataTypes:
        df = pd.read_parquet(f"{getPaths("outputWeatherData")}combined/{dataType}_combined.parquet")
        weatherData.append(df)
    df_weatherForecast = forecastMeteo(df_stations)
    return weatherData, df_weatherForecast

