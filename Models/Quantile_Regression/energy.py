from Settings.input import *
from Preprocessing.preprocessing import *
from Features.Energy.energy_features import *
from Evaluation.models_eval import *
import pandas as pd
import statsmodels.api as sm
from datetime import date
from Evaluation.feature_eval import *

def main():
    """
    1. Parquets löschen
    2. neue Energie Daten von Smard runter laden
    Muss noch: Fixen, sodass Parquets alte gelöscht und neue hinzugefügt werden jedes mal
    Editor Mode nutzt ein zuvor gespeichertes Weather Datenset, sodass nicht jedes mal neu die Daten gezogen werden und
    man Zeit spart, außerdem ist hier Evaluation mit enthalten und Daten Split ist anders

    """
    forecasting = True
    editor_mode = False

    current_date = date.today()

    # Importiere Energie Load Daten
    df_energy = cleanEnergy(importEnergy())
    df_germanHolidays, df_karlsruheHolidays = importHolidays()

    if not editor_mode:
        # Importiere die Wetter Daten (für schnelleren Ablauf unten auskommentieren und csv laden
        df_weatherStations = importWeatherStations()
        weatherData, df_weatherForecast = importWeather(df_weatherStations, ['air_temperature',
                                                                             'precipitation',
                                                                             'sun',
                                                                             'wind']
                                                        )
        df_temperature = temperatur_cleaner(weatherData[0])
        df_precipitation = precipitation_cleaner(weatherData[1])
        df_sun = sun_cleaner(weatherData[2])
        df_wind = wind_cleaner(weatherData[3])
        df_germanHolidays = preprocessHolidays(df_germanHolidays, 3)  # 7 Bundesländer (ca. 26 Stations)
        df_tempAgg = createMetrics(df_temperature, 'temperatur')
        df_precipitationAgg = createMetrics(df_precipitation, 'regenSchnee')
        df_sunAgg = createMetrics(df_sun, 'sonnenscheinDauer')
        df_windAgg = createMetrics(df_wind, 'windGeschwindigkeit')

        df_weather = mergeWeatherData([df_tempAgg,
                                       df_precipitationAgg,
                                       df_sunAgg,
                                       df_windAgg], ['date', 'hour'])
        weather_data_zwischenspeicher = f'{getPaths("zwischenspeicherWeather")}/Energy_Zwischenspeicher/historic_recent_forecast_zwischenspeicher_{current_date}.csv'
        df_weather.to_csv(weather_data_zwischenspeicher)


    if editor_mode:
        # weather_data_zwischenspeicher = f'{getPaths("zwischenspeicherWeather")}/Energy_Zwischenspeicher/historic_recent_forecast_zwischenspeicher_{current_date}.csv'
        weather_data_zwischenspeicher = "/Users/luisafaust/Desktop/PTFSC_Data/Weather/Energy_Zwischenspeicher/historic_recent_forecast_zwischenspeicher_2025-01-08.csv"
        df_weather = pd.read_csv(weather_data_zwischenspeicher)
        df_weather['date'] = pd.to_datetime(df_weather['date'])
        df_weather['hour'] = df_weather['hour'].astype(int)

    df_input = cet_utc(df_weather, df_energy)

    last_index = df_input[df_input['min_temperatur'].notna()].index[-1]
    df_tmp = df_input[df_input['min_temperatur'].isna()]
    df_tmp = df_tmp[['load']].reset_index()
    if not editor_mode:
        df_weatherForecast = preprocessWeatherForecast(df_weatherForecast, df_tmp, last_index)
        df_input = df_input[df_input['min_temperatur'].notna()]
        df_input = pd.concat([df_input, df_weatherForecast])

    df_input = df_input[~df_input.index.duplicated(keep='first')]
    df_input = df_input.asfreq('h')

    #df_input = df_input[['load', 'avg_temperatur', 'avg_windGeschwindigkeit']]
    df_input = df_input[~((df_input['avg_temperatur'].isna()) | (df_input['avg_windGeschwindigkeit'].isna()))]

    # Feature setzen
    df_input = setCalenderFeatureEnergy(df_input, df_germanHolidays)
    df_input = add_xmas_holiday_features_energy(df_input)

    #if forecasting:
    #    df_input = setEnergyLoadFeatures(df_input)

    df_input = add_peak_specific_features(df_input)

    df_input = setWeatherFeatures(df_input)


# NEW END
    # TODO: CHANGE
    columns_to_drop = ['typical_load', 'hour_error_range', 'dayOfYear', 'weekday_hour_avg', 'Date', 'index',
                       'is_weekend', 'time_trend_hours', 'Unnamed: 0']
    columns_to_drop = [col for col in columns_to_drop if col in df_input.columns]
    df_input = df_input.drop(columns=columns_to_drop)


    df_input = pd.get_dummies(df_input, columns=['hour', 'weekday', 'month'], drop_first=True)


    df_input = df_input.apply(pd.to_numeric, errors='coerce')

    columns_to_drop = [
        # Basic metadata only
        "is_weekend", "Unnamed: 0",

        # Non-influential weather features
        "max_relativeFeuchte", "min_relativeFeuchte", "avg_relativeFeuchte",
        "max_sonnenscheinDauer", "min_sonnenscheinDauer", "avg_sonnenscheinDauer",
        "min_regenSchnee", "max_regenSchnee", "avg_regenSchnee",
        "min_windGeschwindigkeit", "max_windGeschwindigkeit", "avg_windGeschwindigkeit",

        "min_temperatur", "max_temperatur", #"HDD",
        #"avg_temperatur",
        # Non-significant features (p > 0.05)
        # "min_temperatur",            # p=0.721
        # "core_holiday_hour_6",       # p=0.609
        # "core_holiday_hour_12",      # p=0.360
        # "holiday_period_hour_20",    # p=0.317
        # "core_holiday_hour_20",      # p=0.232
        # "core_holiday_hour_16",      # p=0.212
        # "holiday_period_hour_12",    # p=0.145
        # "month_6",                   # p=0.110
        # "weekend_morning_flat",      # p=0.084
        # "is_core_holiday",          # p=0.067


    ]

    df_input = df_input.drop(columns=[col for col in columns_to_drop if col in df_input.columns])

    df_input = df_input.dropna(subset=[col for col in df_input.columns if col not in ['load', 'lag_1', 'lag_24']], how='any')

    df_input = df_input.astype({col: 'int' for col in df_input.select_dtypes(include=['bool']).columns})

    print(df_input.columns)

    # Quantile Reg
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    if forecasting:
        df_input = df_input[df_input.index >= '2022-01-01']
        train_df, test_df = split_data(df_input, forecast_horizon=7*24, is_forecasting=True)

    if not forecasting:
        train_start = '2021-01-01 00:00:00'  # Changed to match split
        train_end = '2023-12-31 23:00:00'  # Changed to match split
        df_input = setEnergyLoadFeatures_Eval(df_input, train_start, train_end)

    df_input = df_input[df_input.index >= '2022-01-01']

    if not forecasting:
        train_df, test_df = split_data(df_input,
                                         is_forecasting=False,
                                         train_start='2022-01-01 00:00:00',
                                         train_end='2023-12-31 23:00:00',
                                         test_start='2024-01-01 00:00:00',
                                         test_end='2024-12-31 23:00:00')
        lag = False
        if lag:
            # For rolling_avg_30d: calculate using all data up to each point
            combined_load = pd.concat([train_df['load'], test_df['load']])
            rolling_mean = combined_load.rolling(window=30 * 24, min_periods=1).mean()
            test_df['rolling_avg_30d'] = rolling_mean[len(train_df):].values

            # For lag_1: set first value only
            test_df.loc[test_df.index[0], 'lag_1'] = train_df['load'].iloc[-1]

            # For lag_24: get last 24 values from training
            last_24_train = train_df['load'].iloc[-24:].values
            for i in range(min(24, len(test_df))):
                test_df.loc[test_df.index[i], 'lag_24'] = last_24_train[i]

    train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna()

    # Dictionary zur Speicherung von Actual vs Prediction für jedes Quantil
    actual_vs_pred = {}

    # Unabhängige Variablen (X) und Zielvariable (y) für Training und Test
    X_train = train_df.drop(columns=['load'])
    y_train = train_df['load']

    X_test = test_df.drop(columns=['load'])
    y_test = test_df['load']


    # Optional: Konstanten hinzufügen, falls benötigt
    X_train = sm.add_constant(X_train, has_constant='add')
    X_test = sm.add_constant(X_test, has_constant='add')

    model_median = sm.QuantReg(y_train, X_train)
    res_median = model_median.fit(q=0.5)
    X_test_iter = X_test.copy()
    y_pred_median = []

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)
    if forecasting:
        last_index_date = df_input[df_input['load'].notna()].index[-1]
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
        # actual_vs_pred ist ein Dictionary, das für jedes Quantil (quantile) einen Eintrag enthält
        # Schlüssel ist das aktuelle Quantil (quantile)
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
        pred_1 = pred_1.rename(columns={'actual': 'actual_load'})

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
        total_pred = total_pred.loc[:, ~total_pred.columns.duplicated(keep=False)].drop(columns='actual', errors='ignore')

    tst_tmp = test_df[['load']].reset_index()

    if forecasting:
        total_df = pd.concat([tst_tmp, total_pred], axis=1)
    else:
        total_df = total_pred

    if forecasting:
        total_df.to_excel(f"/Users/luisafaust/Desktop/PTFSC_Data/Forecast/Forecast_Energy_{current_date}.xlsx")
        #total_df.to_excel(f"{getPaths("outputEnergy")}/Forecast_Energy_{current_date}.xlsx")
    else:
        total_df.to_excel(f"{getPaths("outputVal")}/Validation_Energy_2022_{current_date}.xlsx")

    if not forecasting:
    # total_df: Timestamp als Index, actual_load column, quantile q0.025... vorhersagen
        # Define save directory
        #TODO: CHANGE
        save_dir = "/Users/luisafaust/Desktop/PTFSC_Data/Graphs/Evaluation/08_FeatureAnalysis_Energy"

        # List to store all results
        all_results = []

        # Evaluate Energy Model
        energy_results = evaluate_probabilistic_forecast(
            total_df,
            actual_col="actual_load",
            quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"],
            model_name="Energy Model"
        )
        save_evaluation_plots(
            total_df,
            actual_col="actual_load",
            quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"],
            results=energy_results,
            model_name="Energy Model",
            save_dir=save_dir
        )

        all_results.append(energy_results)
        results_df = create_results_dataframe(all_results, save_dir, 'energy')


        # For interactive plot
        fig = create_interactive_forecast_plot(
            total_df,
            actual_col="actual_load",
            quantile_cols=["q0.025", "q0.25", "q0.5", "q0.75", "q0.975"]
        )
        save_interactive_plot(fig, os.path.join(save_dir, 'interactive_forecast.html'))

        # FEATURE ANALYSIS
        # features_to_drop = [col for col in df_input.columns if any(pattern in col for pattern in [
        #     # High correlation with avg_temperatur, redundant information
        # ])]
        #
        # df_input = df_input.drop(columns=features_to_drop)

        feature_analysis_path = f"{save_dir}/Features"
        feature_analysis_results = analyze_features(
            df_input,
            'load',
            feature_analysis_path
        )

        # Plot feature coefficients
        plot_feature_coefficients(
            feature_analysis_results['importance'],
            feature_analysis_path
        )

if __name__ == '__main__':
    main()