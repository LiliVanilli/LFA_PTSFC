from Settings.input import *
from Preprocessing.preprocessing import *
from Features.Bike.bike_hourly_features import *
from Evaluation.models_eval import *
import pandas as pd
import statsmodels.api as sm
from datetime import date, datetime

def main():
    # 1. Parquet löschen
    forecasting = False
    editor_mode = True

    current_date = date.today()
    df_bike_raw = importBikehourly()

    df_bike = preprocessBikeHourly(df_bike_raw)
    df_bike = remove_outliers(df_bike, column='bike_count')
    df_germanHolidays, df_karlsruheHolidays = importHolidays()

    ###

    if not editor_mode:
        df_weatherStations = importWeatherStations()
        df_weatherStations = df_weatherStations[df_weatherStations['Stationsname'] == 'Rheinstetten'].reset_index(drop=True)
        weatherData, df_weatherForecast = importWeather(df_weatherStations, ['air_temperature',
                                                                             'precipitation',
                                                                             'sun',
                                                                             'wind']
                                                        )
        df_temperature = temperatur_cleaner(weatherData[0])
        df_temperature = df_temperature[df_temperature['STATIONS_ID'] == 4177]
        df_precipitation = precipitation_cleaner(weatherData[1])
        df_precipitation = df_precipitation[df_precipitation['STATIONS_ID'] == 4177]
        df_sun = sun_cleaner(weatherData[2])
        df_sun = df_sun[df_sun['STATIONS_ID'] == 4177]
        df_wind = wind_cleaner(weatherData[3])
        df_wind = df_wind[df_wind['STATIONS_ID'] == 4177]
        # df_germanHolidays = preprocessHolidays(df_germanHolidays, 3)  # 7 Bundesländer (ca. 26 Stations)
        df_tempAgg = createMetrics(df_temperature, 'temperatur', hourly=True)
        df_precipitationAgg = createMetrics(df_precipitation, 'regenSchnee', hourly=True)
        df_sunAgg = createMetrics(df_sun, 'sonnenscheinDauer', hourly=True)
        df_windAgg = createMetrics(df_wind, 'windGeschwindigkeit', hourly=True)

        df_weather = mergeWeatherData([df_tempAgg,
                                       df_precipitationAgg,
                                       #df_sunAgg,
                                       df_windAgg], ['date', 'hour'])

        weather_data_zwischenspeicher = f'{getPaths("zwischenspeicherWeather")}/Bike_Zwischenspeicher/historic_recent_forecast_zwischenspeicher_{current_date}.csv'
        df_weather.to_csv(weather_data_zwischenspeicher)


    if editor_mode:
        weather_data_zwischenspeicher = "/Users/luisafaust/Desktop/PTFSC_Data/Weather/Bike_Zwischenspeicher/historic_recent_forecast_zwischenspeicher_2025-01-09.csv"
        df_weather = pd.read_csv(weather_data_zwischenspeicher)
        df_weather = df_weather.drop(df_weather.columns[0], axis=1)
        df_weather['date'] = pd.to_datetime(df_weather['date'])
        df_weather['hour'] = df_weather['hour'].astype(int)


    df_input = cet_utc(df_weather, df_bike)

    last_index = df_input[df_input['min_temperatur'].notna()].index[-1]
    df_tmp = df_input[df_input['min_temperatur'].isna()]
    df_tmp = df_tmp[['bike_count']].reset_index()

    if not editor_mode:
        df_weatherForecast = preprocessWeatherForecast(df_weatherForecast, df_tmp, last_index)
        df_input = df_input[df_input['min_temperatur'].notna()]
        df_input = pd.concat([df_input, df_weatherForecast])
    # if editor_mode:
        # df_input = df_input[df_input['min_temperatur'].notna()]

    df_input = df_input[~df_input.index.duplicated(keep='first')]
    df_input = df_input.asfreq('h')

    df_input = df_input[['bike_count', 'avg_temperatur', 'avg_windGeschwindigkeit', 'avg_regenSchnee']]
    df_input = df_input[~((df_input['avg_temperatur'].isna()) | (df_input['avg_windGeschwindigkeit'].isna())) | (df_input['avg_regenSchnee'].isna())]


    # Features
    df_karlsruheHolidays['Datum'] = pd.to_datetime(df_karlsruheHolidays['Date'], format='%d.%m.%y', errors='coerce')
    df_karlsruheHolidays['major'] = 1 - df_karlsruheHolidays['Minor']
    df_input = setCalenderFeatureBikeHOURLY(df_input, df_karlsruheHolidays)
    if forecasting:
        df_input = setBikeHourlyLoadFeatures(df_input)

    df_input = pd.get_dummies(df_input, columns=['hour', 'weekday', 'month'], drop_first=True)
    df_input = df_input.replace({True: 1, False: 0})

    df_input = df_input.apply(pd.to_numeric, errors='coerce')


    # TODO: checke, ob min, max entfernt wird
    columns_to_drop = ["is_weekend", "temp_change_1w",
                                      "temp_change_24h",
                       'weekday_hour_avg', 'dayOfYear', 'holiday_hour', 'date', 'time_trend_hours',
                       'is_weekend_or_holiday', 'Date', 'index', 'Unnamed: 0', 'lag_168'
                                      ]



    columns_to_drop = [col for col in columns_to_drop if col in df_input.columns]
    df_input = df_input.drop(columns=columns_to_drop)

    df_input = df_input.dropna(subset=[col for col in df_input.columns if col not in ['bike_count', 'lag_1', 'lag_24', 'lag_168']],
                               how='any')

    if forecasting:
        columns_to_convert = ['bike_count', 'lag_1', 'lag_24']
        for col in columns_to_convert:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
    if not forecasting:
        columns_to_convert = ['bike_count']
        for col in columns_to_convert:
            df_input[col] = pd.to_numeric(df_input[col], errors='coerce')

    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    if forecasting:
        train_df, test_df = split_data(df_input, is_forecasting=True, forecast_horizon=7*24)

    if not forecasting:
        train_start = '2021-01-01 00:00:00'
        train_end = '2023-12-31 23:00:00'
        df_input = setBikeHourlyLoadFeaturesEval(df_input, train_start, train_end)

        df_input = df_input[df_input.index >= '2022-01-01']

        train_df, test_df = split_data(df_input,
                                     is_forecasting=False,
                                     train_start='2022-01-01 00:00:00',
                                     train_end='2023-12-31 23:00:00',
                                     test_start='2024-01-01 00:00:00',
                                     test_end='2024-12-31 23:00:00')

        # For rolling_avg_30d: calculate using all data up to each point
        combined_load = pd.concat([train_df['bike_count'], test_df['bike_count']])
        rolling_mean = combined_load.rolling(window=30 * 24, min_periods=1).mean()
        test_df['rolling_avg_30d'] = rolling_mean[len(train_df):].values

        # For lag_1: set first value only
        test_df.loc[test_df.index[0], 'lag_1'] = train_df['bike_count'].iloc[-1]

        # For lag_24: get last 24 values from training
        last_24_train = train_df['bike_count'].iloc[-24:].values
        for i in range(min(24, len(test_df))):
            test_df.loc[test_df.index[i], 'lag_24'] = last_24_train[i]

        # For lag_168 (1 week): get last 168 values from training
        last_168_train = train_df['bike_count'].iloc[-168:].values
        for i in range(min(168, len(test_df))):
            test_df.loc[test_df.index[i], 'lag_168'] = last_168_train[i]

    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()

    actual_vs_pred = {}
    X_train = train_df.drop(columns=['bike_count'])
    y_train = train_df['bike_count']

    X_test = test_df.drop(columns=['bike_count'])
    y_test = test_df['bike_count']

    X_train = sm.add_constant(X_train, has_constant='add')
    X_test = sm.add_constant(X_test, has_constant='add')


    model_median = sm.QuantReg(y_train, X_train)
    res_median = model_median.fit(q=0.5)
    X_test_iter = X_test.copy()
    y_pred_median = []

    if forecasting:
        last_index_date = df_input[df_input['bike_count'].notna()].index[-1]
        count_after_timestamp = len(df_input[df_input.index > last_index_date])

    if forecasting:
        overwrite_start_1 = len(X_test) - (count_after_timestamp + 1)
        overwrite_start_2 = len(X_test) - (25 + count_after_timestamp + 1)
    if not forecasting:
        overwrite_start_1 = 1
        overwrite_start_2 = 24

    for quantile in quantiles:
        # Quantil-Regression durchführen
        model = sm.QuantReg(y_train, X_train)
        res = model.fit(q=quantile)

        # Vorhersagen
        # y_pred = res.predict(X_test)

        y_pred_quantile = []

        for t in range(len(X_test_iter)):
            # Vorhersage des 0.5-Quantils für den aktuellen Zeitpunkt (falls noch nicht gemacht)
            if quantile == 0.5:
                pred_median_t = res_median.predict(X_test_iter.iloc[t:t + 1])[0]
                y_pred_median.append(pred_median_t)  # Speichere die Medianvorhersage

            # Für alle Quantile wird das 0.5-Quantil als Lag verwendet
            pred_median_t = y_pred_median[t] if len(y_pred_median) > t else \
                res_median.predict(X_test_iter.iloc[t:t + 1])[0]

            # Vorhersage für das aktuelle Quantil basierend auf den aktualisierten Lagged Features
            pred_quantile_t = res.predict(X_test_iter.iloc[t:t + 1])[0]
            y_pred_quantile.append(pred_quantile_t)  # Speichere die Vorhersage für das aktuelle Quantil

            # Aktualisieren der Lagged Features mit der 0.5 Quantil-Vorhersage
            if 'lag_1' in X_test_iter.columns and t + 1 >= overwrite_start_1:
                if t + 1 < len(X_test_iter):  # Vermeide Out-of-Bounds-Fehler
                    X_test_iter.at[X_test_iter.index[t + 1], 'lag_1'] = pred_median_t

            # Aktualisieren von lag_24 mit der 0.5-Quantil-Vorhersage ab der 25. Vorhersage
            if 'lag_24' in X_test_iter.columns and t + 1 >= overwrite_start_2:
                if t + 1 < len(X_test_iter):  # Vermeide Out-of-Bounds-Fehler
                    X_test_iter.at[X_test_iter.index[t + 1], 'lag_24'] = pred_median_t

        y_pred_quantile = np.array(y_pred_quantile)

        # Actual vs Prediction für dieses Quantil speichern
        if forecasting:
            actual_vs_pred[quantile] = pd.DataFrame({
                'prediction': y_pred_quantile
            })
        else:
            actual_vs_pred[quantile] = pd.DataFrame({
                'actual': y_test,
                'prediction': y_pred_quantile
            })


    pred_1 = actual_vs_pred[0.025]
    pred_1 = pred_1.rename(columns={'prediction': 'q0.025'})
    if not forecasting:
        pred_1 = pred_1.rename(columns={'actual': 'actual_bike_count'})

    pred_2 = actual_vs_pred[0.25]
    pred_2 = pred_2.rename(columns={'prediction': 'q0.25'})

    pred_3 = actual_vs_pred[0.5]
    pred_3 = pred_3.rename(columns={'prediction': 'q0.5'})

    pred_4 = actual_vs_pred[0.75]
    pred_4 = pred_4.rename(columns={'prediction': 'q0.75'})

    pred_5 = actual_vs_pred[0.975]
    pred_5 = pred_5.rename(columns={'prediction': 'q0.975'})

    total_pred = pd.concat([pred_1, pred_2, pred_3, pred_4, pred_5], axis=1)
    if not forecasting:
        total_pred = total_pred.loc[:, ~total_pred.columns.duplicated(keep=False)].drop(columns='actual',
                                                                                        errors='ignore')

    tst_tmp = test_df[['bike_count']].reset_index()

    if forecasting:
        total_df = pd.concat([tst_tmp, total_pred], axis=1)
    else:
        total_df = total_pred


    if not forecasting:
    # negative Werte in dataframe auf 0 setzen, da es keine negativen Fahrrad Counts gibts
        total_df[total_df < 0] = 0

    if forecasting:
        # bei forecasting geändert
        quantile_columns = ['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975']

        # Setze negative Werte in diesen quantile columns auf 0
        total_df[quantile_columns] = total_df[quantile_columns].clip(lower=0)

        # bei forecasting geändert
        total_df = total_df.set_index('index')


    quantile_columns = ["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]
    total_df = validate_and_fix_quantiles(total_df, quantile_columns)

    # Resample auf tägliche Basis und summiere alle Spalten
    daily_sums = total_df.resample('D').sum()

    if forecasting:
       total_df.to_excel(f"{getPaths("outputEnergy")}/Forecast_Bike_hourly_{current_date}.xlsx")
       daily_sums.to_excel(f"{getPaths("outputEnergy")}/Forecast_Bike_daily_{current_date}.xlsx")
    else:
       total_df.to_excel(f"{getPaths("outputVal")}/Validation_Bike_hourly_8_{current_date}.xlsx")
       daily_sums.to_excel(f"{getPaths("outputEnergy")}/Validation_Bike_daily_8_{current_date}.xlsx")

    if not forecasting:
        save_dir = f'{getPaths("quantile_reg_eval")}/Bike_hourly'
        all_results = []

        # Evaluate Hourly Bike Model
        hourly_bike_results = evaluate_probabilistic_forecast(
            total_df,
            actual_col="actual_bike_count",
            quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"],
            model_name="Hourly Bike Model"
        )
        save_evaluation_plots(
            total_df,
            actual_col="actual_bike_count",
            quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"],
            results=hourly_bike_results,
            model_name="Hourly Bike Model",
            save_dir=save_dir
        )
        all_results.append(hourly_bike_results)

        # Evaluate Daily Summed Bike Model
        daily_summed_results = evaluate_probabilistic_forecast(
            daily_sums,
            actual_col="actual_bike_count",
            quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"],
            model_name="Daily Summed Bike Model"
        )
        save_evaluation_plots(
            daily_sums,
            actual_col="actual_bike_count",
            quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"],
            results=daily_summed_results,
            model_name="Daily Summed Bike Model",
            save_dir=save_dir
        )
        all_results.append(daily_summed_results)
        results_df = create_results_dataframe(all_results, save_dir, 'bike_hourly')

        fig = create_static_forecast_plot(
            total_df,
            actual_col='actual_bike_count',
            quantile_cols=['q0.025', 'q0.25', 'q0.5', 'q0.75', 'q0.975'],
            start_date='2024-12-01',
            end_date='2024-12-31',
            title='Bike Count Forecast',
            save_path=f'{save_dir}/forecast_december_2024.pdf'
        )

        # For interactive plot
        fig = create_interactive_forecast_plot_bike_hourly(
            total_df,
            actual_col="actual_bike_count",
            quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]
        )
        save_interactive_plot(fig, os.path.join(save_dir, 'interactive_forecast.html'))


        # Plot nur mit diesem Subset erstellen
        fig = create_interactive_forecast_plot_bike_hourly(
            daily_sums,
            actual_col="actual_bike_count",
            quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]
        )
        save_interactive_plot(fig, os.path.join(save_dir, 'interactive_forecast_dailysums.html'))



if __name__ == '__main__':
    main()
