from Settings.input import *
from Preprocessing.preprocessing import *
from Features.Bike.bike_daily_features import *
from Evaluation.models_eval import *
import pandas as pd
import statsmodels.api as sm
from datetime import date, datetime
from Evaluation.feature_eval import *


def main():
    """
    MERKE, wenn keine Angaben, dann kann sein, dass Wetter Station in Karlsruhe an manchen Tagen keine Daten oder,
    dass keine Bike Daten
    1. In Recent die Parquets vorher immer löschen, sodass Daten aktualisiert werden auf neues Wetter
    Muss noch: Fixen, sodass Parquets alte gelöscht und neue hinzugefügt werden jedes mal

    Editor Mode nutzt ein zuvor gespeichertes Weather Datenset, sodass nicht jedes mal neu die Daten gezogen werden und
    man Zeit spart, außerdem ist hier Evaluation mit enthalten und Daten Split ist anders

    """
    forecasting = True

    editor_mode = False

    current_date = date.today()
    df_bike = importBike()
    df_bike = preprocessBike(df_bike)
    df_bike = remove_outliers(df_bike, column='bike_count')

    df_germanHolidays, df_karlsruheHolidays = importHolidays()

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
        # Skalierung der Sonnenscheindauer, damit gleich zu historischen Daten
        df_sun['sonnenscheinDauer'] = df_sun['sonnenscheinDauer'] / 100
        df_wind = wind_cleaner(weatherData[3])
        df_wind = df_wind[df_wind['STATIONS_ID'] == 4177]
        df_tempAgg = createMetrics(df_temperature, 'temperatur', hourly=False)
        df_precipitationAgg = createMetrics(df_precipitation, 'regenSchnee', hourly=False)
        df_sunAgg = createMetrics(df_sun, 'sonnenscheinDauer', hourly=False)
        df_windAgg = createMetrics(df_wind, 'windGeschwindigkeit', hourly=False)

        df_weather = mergeWeatherData([df_tempAgg,
                                       df_precipitationAgg,
                                       df_sunAgg,
                                       df_windAgg], ['date'])
        weather_data_zwischenspeicher = f'{getPaths("zwischenspeicherWeather")}/Bike_Zwischenspeicher/historic_recent_forecast_zwischenspeicher_DAILY_{current_date}.csv'
        df_weather.to_csv(weather_data_zwischenspeicher)

    if editor_mode:
        weather_data_zwischenspeicher = f'{getPaths("zwischenspeicherWeather")}/Bike_Zwischenspeicher/WithoutMissingData_Bike.csv'
        df_weather = pd.read_csv(weather_data_zwischenspeicher)
        # df_weather = df_weather.drop(df_weather.columns[0], axis=1)
        df_weather['date'] = pd.to_datetime(df_weather['date'])

    # First entries are NaNs because Bike Count starts 2014 while weather starts 2015
    df_input = pd.merge(df_bike, df_weather, how='left', on='date')
    df_input = df_input.set_index('date')

    last_index = df_input[df_input['min_temperatur'].notna()].index[-1]

    if not editor_mode:
        df_weatherForecast = df_weatherForecast[df_weatherForecast['date'] > last_index].reset_index(drop=True)


        # from GMT to CET
        df_weatherForecast['date'] = df_weatherForecast['date'] + pd.Timedelta(hours=1)

        df_weatherForecast = df_weatherForecast.rename(columns={'temperature_2m': 'temperatur',
                                                                #'relative_humidity_2m': 'relativeFeuchte',
                                                                'wind_speed_10m': 'windGeschwindigkeit',
                                                                'precipitation': 'regenSchnee',
                                                                'sunshine_duration': 'sonnenscheinDauer'
                                                                })

        df_weatherForecast['date'] = df_weatherForecast['date'].dt.date
        df_weatherForecast['date'] = pd.to_datetime(df_weatherForecast['date'])

        df_weatherForecast = df_weatherForecast.groupby(['date']).agg(
            min_temperatur=('temperatur', 'min'),
            #min_relativeFeuchte=('relativeFeuchte', 'min'),
            min_windGeschwindigkeit=('windGeschwindigkeit', 'min'),
            min_regenSchnee=('regenSchnee', 'min'),
            min_sonnenscheinDauer=('sonnenscheinDauer', 'min'),
            avg_temperatur=('temperatur', 'mean'),
            #avg_relativeFeuchte=('relativeFeuchte', 'mean'),
            avg_windGeschwindigkeit=('windGeschwindigkeit', 'mean'),
            avg_regenSchnee=('regenSchnee', 'mean'),
            avg_sonnenscheinDauer=('sonnenscheinDauer', 'mean'),
            max_temperatur=('temperatur', 'max'),
            #max_relativeFeuchte=('relativeFeuchte', 'max'),
            max_windGeschwindigkeit=('windGeschwindigkeit', 'max'),
            max_regenSchnee=('regenSchnee', 'max'),
            max_sonnenscheinDauer=('sonnenscheinDauer', 'max')
        ).reset_index()

        df_weatherForecast = df_weatherForecast.set_index('date')
        df_input = pd.concat([df_input, df_weatherForecast])

    df_input = df_input[~df_input.index.duplicated(keep='first')]
    df_input = df_input.asfreq('d')

    # Features
    df_input = setCalenderFeatureBike(df_input, df_karlsruheHolidays)
    df_input = add_xmas_holiday_features_bike_daily(df_input)
    # df_input = setWeatherFeaturesBike(df_input)

    if forecasting:
        df_input = setBikeLoadFeatures(df_input)

    df_input = pd.get_dummies(df_input, columns=['weekday', 'month'], drop_first=True)

    columns_to_drop = ['dayOfYear', 'weekday_hour_avg', 'Date', 'index', 'Date', 'is_weekend', 'Unnamed: 0',
                       'avg_relativeFeuchte', 'min_relativeFeuchte', 'max_relativeFeuchte',

                       #  # After Feature Engineering
                       #  high correlation
                       # 'avg_temperatur'
                       #   'min_temperatur',
                            'max_temperatur',
                       #   'min_regenSchnee',
                       #   'max_regenSchnee',
                       # ACHTUNG!: Sonnenschein hat Modell zerschossen, da Daten im Forecast oft fehlerhaft sind
                           'min_sonnenscheinDauer',
                           'max_sonnenscheinDauer',
                       #   'min_windGeschwindigkeit',
                       #   'max_windGeschwindigkeit',
                       #   'HDD',
                       #  # no significance
                           'min_temperatur',  # + high corr
                           'min_regenSchnee',  # + high corr
                           'avg_sonnenscheinDauer',  # + high corr
                           'month_3',
                           'month_6',
                           'month_9',
                           'bridge_day',
                           'min_windGeschwindigkeit',  # + high corr
                           'month_5',
                           'month_4',
                           'avg_windGeschwindigkeit',  # + high corr
                           'month_7',
                            #'lag_1',
                            #'lag_1w'
                           #'month_10',
                           #'month_11',
                           'HDD'  # + high corr
                       ]
    columns_to_drop = [col for col in columns_to_drop if col in df_input.columns]
    df_input = df_input.drop(columns=columns_to_drop)

    df_input = df_input.apply(pd.to_numeric, errors='coerce')

    df_input = df_input.dropna(subset=[col for col in df_input.columns if col not in ['bike_count', 'lag_1', 'lag_1w']],
                                how='any')

    df_input = df_input.astype({col: 'int' for col in df_input.select_dtypes(include=['bool']).columns})

    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    if forecasting:
        df_input = df_input[df_input.index >= '2022-01-01']
        train_df, test_df = split_data(df_input, is_forecasting=True)

    if not forecasting:
        train_start = '2021-01-01'
        train_end = '2023-12-31'
        # wenn nicht Forecasting muss erst hier Load Funktion sein, damit kein Daten Leak
        df_input = setBikeLoadFeatures_Eval(df_input, train_start, train_end)

        df_input = df_input[df_input.index >= '2022-01-01']

        train_df, test_df = split_data(df_input,
                                         is_forecasting=False,
                                         train_start='2022-01-01',
                                         train_end='2023-12-31',
                                         test_start='2024-01-01',
                                         test_end='2024-12-31')

        # For rolling_avg_30d: calculate using all data up to each point
        combined_bike_count = pd.concat([train_df['bike_count'], test_df['bike_count']])
        rolling_mean = combined_bike_count.rolling(window=30, min_periods=1).mean()
        test_df['rolling_avg_30d'] = rolling_mean[len(train_df):].values

        # For lag_1: set first value only (1 day lag)
        test_df.loc[test_df.index[0], 'lag_1'] = train_df['bike_count'].iloc[-1]

        # For lag_1w: get last 7 values from training
        last_7_train = train_df['bike_count'].iloc[-7:].values
        for i in range(min(7, len(test_df))):
            test_df.loc[test_df.index[i], 'lag_1w'] = last_7_train[i]

        print(df_input.columns)


    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Dictionary zur Speicherung von Actual vs Prediction für jedes Quantil
    actual_vs_pred = {}

    # Unabhängige Variablen (X) und Zielvariable (y) für Training und Test
    X_train = train_df.drop(columns=['bike_count'])
    y_train = train_df['bike_count']

    X_test = test_df.drop(columns=['bike_count'])
    y_test = test_df['bike_count']

    # Optional: Konstanten hinzufügen, falls benötigt
    X_train = sm.add_constant(X_train, has_constant='add')
    X_test = sm.add_constant(X_test, has_constant='add')

    model_median = sm.QuantReg(y_train, X_train)
    res_median = model_median.fit(q=0.5)
    X_test_iter = X_test.copy()
    y_pred_median = []

    if forecasting:
        last_index_date = df_input[df_input['bike_count'].notna()].index[-1]
        count_after_timestamp = len(df_input[df_input.index > last_index_date])
        overwrite_start_1 = len(test_df) - (count_after_timestamp + 1)
        overwrite_start_2 = len(test_df) - (8 + count_after_timestamp + 1)


    if not forecasting:
        overwrite_start_1 = 1
        overwrite_start_2 = 7

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
            if 'lag_1w' in X_test_iter.columns and t + 1 >= overwrite_start_2:
                if t + 1 < len(X_test_iter):  # Vermeide Out-of-Bounds-Fehler
                    X_test_iter.at[X_test_iter.index[t + 1], 'lag_1w'] = pred_median_t

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

    if forecasting:
        # TODO: CHANGE
        total_df.to_excel(f"/Users/luisafaust/Desktop/PTFSC_Data/Forecast/Forecast_Bike_daily_2022_{current_date}.xlsx")
        #total_df.to_excel(f"{getPaths("outputEnergy")}/Forecast_Bike_daily_2022_{current_date}.xlsx")
    else:
        total_df.to_excel(f"{getPaths("outputVal")}/Validation_Bike_daily_2022_{current_date}.xlsx")

    #test_df.to_excel("/Users/luisafaust/Desktop/PTFSC_Data/check/test.xlsx")
    #train_df.to_excel("/Users/luisafaust/Desktop/PTFSC_Data/check/train.xlsx")


    if not forecasting:
            save_dir = f'{getPaths("quantile_reg_eval")}/Bike_daily'
            all_results = []
            daily_direct_results = evaluate_probabilistic_forecast(
                total_df,
                actual_col="actual_bike_count",
                quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"],
                model_name="Direct Daily Bike Model"
            )
            save_evaluation_plots(
                total_df,
                actual_col="actual_bike_count",
                quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"],
                results=daily_direct_results,
                model_name="Direct Daily Bike Model",
                save_dir=save_dir
            )
            all_results.append(daily_direct_results)
            results_df = create_results_dataframe(all_results, save_dir, 'bike_daily')

            # For interactive plot
            fig = create_interactive_forecast_plot_bike(
                total_df,
                actual_col="actual_bike_count",
                quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]
            )
            save_interactive_plot(fig, os.path.join(save_dir, 'interactive_forecast.html'))

            feature_analysis_path = f"{save_dir}/bike_{current_date}"
            feature_analysis_results = analyze_features(
                df_input,
                'bike_count',
                feature_analysis_path
            )

            # Plot feature coefficients
            plot_feature_coefficients(
                feature_analysis_results['importance'],
                feature_analysis_path
            )


if __name__ == '__main__':
    main()

