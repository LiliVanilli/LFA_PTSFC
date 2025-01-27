from Settings.input import *
from Preprocessing.preprocessing import *
from Features.Bike.bike_daily_features import *
from Evaluation.feature_eval import *
from Evaluation.models_eval import *
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import date
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


def perform_time_series_cv_bike(X_train_full, y_train_full, df_input, n_splits=5):
    """
    Perform time series cross-validation for bike traffic forecasting
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = {
        'fold_metrics': [],
        'quantile_pinball_losses': {q: [] for q in [0.025, 0.25, 0.5, 0.75, 0.975]},
        'mean_pinball_losses': [],
        'crps_scores': [],
        'coverage_metrics': []
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full), 1):
        X_train_cv = X_train_full.iloc[train_idx]
        y_train_cv = y_train_full.iloc[train_idx]
        X_val_cv = X_train_full.iloc[val_idx]
        y_val_cv = y_train_full.iloc[val_idx]

        predictions, _ = train_quantile_xgboost_bike(
            X_train_cv, y_train_cv,
            X_val_cv, y_val_cv,
            X_val_cv,
            df_input,
            quantiles=[0.025, 0.25, 0.5, 0.75, 0.975],
            forecasting=False
        )

        results_df = pd.DataFrame(index=X_val_cv.index)
        results_df['actual_bike_count'] = y_val_cv
        for q, pred in predictions.items():
            results_df[f'q{q}'] = pred

        eval_results = evaluate_probabilistic_forecast(
            results_df,
            actual_col="actual_bike_count",
            quantile_cols=[f"q{q}" for q in [0.025, 0.25, 0.5, 0.75, 0.975]],
            model_name=f"CV Fold {fold}"
        )

        for q, loss in eval_results['pinball_losses'].items():
            cv_results['quantile_pinball_losses'][float(q.replace('q', ''))].append(loss)

        cv_results['mean_pinball_losses'].append(eval_results['mean_pinball_loss'])
        cv_results['crps_scores'].append(eval_results['crps'])
        cv_results['coverage_metrics'].append(eval_results['coverage'])

        fold_summary = {
            'fold': fold,
            'start_date': X_val_cv.index[0],
            'end_date': X_val_cv.index[-1],
            'n_samples': len(X_val_cv),
            **{f'pinball_loss_q{q}': eval_results['pinball_losses'][f'q{q}']
               for q in [0.025, 0.25, 0.5, 0.75, 0.975]},
            'mean_pinball_loss': eval_results['mean_pinball_loss'],
            'crps': eval_results['crps']
        }
        cv_results['fold_metrics'].append(fold_summary)

    return cv_results


def save_cv_results_bike(cv_results, save_dir, current_date):
    """
    Save cross-validation results for bike model
    """
    os.makedirs(save_dir, exist_ok=True)

    fold_metrics_df = pd.DataFrame(cv_results['fold_metrics'])
    fold_metrics_df.to_csv(f"{save_dir}/cv_detailed_metrics_bike_{current_date}.csv", index=False)

    summary_stats = {
        'metric': ['Mean Pinball Loss', 'CRPS'] +
                  [f'Pinball Loss Q{q}' for q in [0.025, 0.25, 0.5, 0.75, 0.975]],
        'mean': [np.mean(cv_results['mean_pinball_losses']),
                 np.mean(cv_results['crps_scores'])] +
                [np.mean(cv_results['quantile_pinball_losses'][q])
                 for q in [0.025, 0.25, 0.5, 0.75, 0.975]],
        'std': [np.std(cv_results['mean_pinball_losses']),
                np.std(cv_results['crps_scores'])] +
               [np.std(cv_results['quantile_pinball_losses'][q])
                for q in [0.025, 0.25, 0.5, 0.75, 0.975]]
    }

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(f"{save_dir}/cv_summary_stats_bike_{current_date}.csv", index=False)

    coverage_df = pd.DataFrame(cv_results['coverage_metrics'])
    coverage_df.to_csv(f"{save_dir}/cv_coverage_metrics_bike_{current_date}.csv", index=False)


def train_quantile_xgboost_bike(X_train, y_train, X_val, y_val, X_test, df_input, quantiles, forecasting=False):
    predictions = {}
    models = {}
    base_params = {
        'n_estimators': 1000,
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist',
        'early_stopping_rounds': 50
    }

    # Train median model first
    median_model = XGBRegressor(**base_params, objective='reg:quantileerror', quantile_alpha=0.5)
    median_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    models[0.5] = median_model

    # Initialize prediction storage
    y_pred_median = []
    X_test_iter = X_test.copy()

    # Determine overwrite points
    if forecasting:
        last_index_date = df_input[df_input['bike_count'].notna()].index[-1]
        count_after_timestamp = len(df_input[df_input.index > last_index_date])
        overwrite_start_1 = len(X_test) - (count_after_timestamp + 1)
        overwrite_start_2 = len(X_test) - (8 + count_after_timestamp + 1)
    else:
        overwrite_start_1 = 1
        overwrite_start_2 = 7

    # Iterative prediction for median
    for t in range(len(X_test_iter)):
        pred_median_t = median_model.predict(X_test_iter.iloc[t:t + 1])[0]
        y_pred_median.append(pred_median_t)

        if t + 1 < len(X_test_iter):
            if 'lag_1' in X_test_iter.columns and t + 1 >= overwrite_start_1:
                X_test_iter.at[X_test_iter.index[t + 1], 'lag_1'] = pred_median_t
            if 'lag_1w' in X_test_iter.columns and t + 1 >= overwrite_start_2:
                X_test_iter.at[X_test_iter.index[t + 1], 'lag_1w'] = pred_median_t

    # Train and predict for all quantiles
    for q in quantiles:
        model = XGBRegressor(**base_params, objective='reg:quantileerror', quantile_alpha=q)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        if q == 0.5:
            predictions[q] = y_pred_median
        else:
            X_test_q = X_test_iter.copy()
            y_pred_quantile = []

            for t in range(len(X_test_q)):
                pred_t = model.predict(X_test_q.iloc[t:t + 1])[0]
                y_pred_quantile.append(pred_t)

            predictions[q] = y_pred_quantile

    return predictions, models[0.5]


def main():
    forecasting = False
    editor_mode = True
    lag = False
    current_date = date.today()

    # Import and preprocess data
    df_bike = importBike()
    df_bike = preprocessBike(df_bike)
    df_bike = remove_outliers(df_bike, column='bike_count')

    df_germanHolidays, df_karlsruheHolidays = importHolidays()
    # Load or process weather data
    if not editor_mode:
        df_weatherStations = importWeatherStations()
        df_weatherStations = df_weatherStations[df_weatherStations['Stationsname'] == 'Rheinstetten'].reset_index(
            drop=True)
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
                                                                # 'relative_humidity_2m': 'relativeFeuchte',
                                                                'wind_speed_10m': 'windGeschwindigkeit',
                                                                'precipitation': 'regenSchnee',
                                                                'sunshine_duration': 'sonnenscheinDauer'
                                                                })

        df_weatherForecast['date'] = df_weatherForecast['date'].dt.date
        df_weatherForecast['date'] = pd.to_datetime(df_weatherForecast['date'])

        df_weatherForecast = df_weatherForecast.groupby(['date']).agg(
            min_temperatur=('temperatur', 'min'),
            # min_relativeFeuchte=('relativeFeuchte', 'min'),
            min_windGeschwindigkeit=('windGeschwindigkeit', 'min'),
            min_regenSchnee=('regenSchnee', 'min'),
            min_sonnenscheinDauer=('sonnenscheinDauer', 'min'),
            avg_temperatur=('temperatur', 'mean'),
            # avg_relativeFeuchte=('relativeFeuchte', 'mean'),
            avg_windGeschwindigkeit=('windGeschwindigkeit', 'mean'),
            avg_regenSchnee=('regenSchnee', 'mean'),
            avg_sonnenscheinDauer=('sonnenscheinDauer', 'mean'),
            max_temperatur=('temperatur', 'max'),
            # max_relativeFeuchte=('relativeFeuchte', 'max'),
            max_windGeschwindigkeit=('windGeschwindigkeit', 'max'),
            max_regenSchnee=('regenSchnee', 'max'),
            max_sonnenscheinDauer=('sonnenscheinDauer', 'max')
        ).reset_index()

        df_weatherForecast = df_weatherForecast.set_index('date')
        df_input = pd.concat([df_input, df_weatherForecast])

    df_input = df_input[~df_input.index.duplicated(keep='first')]
    df_input = df_input.asfreq('d')

    # Feature engineering
    df_input = setCalenderFeatureBike(df_input, df_karlsruheHolidays)
    df_input = add_xmas_holiday_features_bike_daily(df_input)

    columns_to_drop = ['dayOfYear', 'weekday_hour_avg', 'Date', 'index', 'Date', 'is_weekend', 'Unnamed: 0',
                       'avg_relativeFeuchte', 'min_relativeFeuchte', 'max_relativeFeuchte',

                       # #  # After Feature Engineering
                       # #  high correlation
                       # #   'min_temperatur',
                       #      'max_temperatur',
                       # #   'min_regenSchnee',
                       # #   'max_regenSchnee',
                       #     'min_sonnenscheinDauer',
                       # #   'max_sonnenscheinDauer',
                       # #   'min_windGeschwindigkeit',
                       # #   'max_windGeschwindigkeit',
                       # #   'HDD',
                       # #  # no significance
                       #     'min_temperatur',  # + high corr
                       #     'min_regenSchnee',  # + high corr
                       #     'avg_sonnenscheinDauer',  # + high corr
                       #     'month_3',
                       #     'month_6',
                       #     'month_9',
                       #     'bridge_day',
                       #     'min_windGeschwindigkeit',  # + high corr
                       #     'month_5',
                       #     'month_4',
                       #     'avg_windGeschwindigkeit',  # + high corr
                       #     'month_7',
                       #      #'lag_1',
                       #      #'lag_1w'
                       #     #'month_10',
                       #     #'month_11',
                       #     'HDD'  # + high corr
                       ]
    columns_to_drop = [col for col in columns_to_drop if col in df_input.columns]
    df_input = df_input.drop(columns=columns_to_drop)

    df_input = df_input.dropna()


    # Split data
    if forecasting:
        df_input = setBikeLoadFeatures(df_input)  # Apply rolling and lag features
        train_df, val_df, test_df = split_data_xgboost(
            df_input=df_input,
            is_forecasting=True,
            forecast_horizon=7  # Use 7 days for forecasting
        )
    else:
        train_df, val_df, test_df = split_data_xgboost(
            df_input=df_input,
            is_forecasting=False,
            train_start='2022-01-01',
            train_end='2023-09-30',
            val_start='2023-10-01',
            val_end='2023-12-31',
            test_start='2024-01-01',
            test_end='2024-12-31'
        )
        if lag:
            train_df = setBikeLoadFeatures_Eval(train_df, train_df.index[0], train_df.index[-1])
            val_df = setBikeLoadFeatures_Eval(val_df, val_df.index[0], val_df.index[-1])
            test_df['rolling_avg_30d'] = 0
            test_df['lag_1'] = 0
            test_df['lag_1w'] = 0

    # Prepare data
    X_train = train_df.drop(columns=['bike_count'])
    y_train = train_df['bike_count']
    X_val = val_df.drop(columns=['bike_count'])
    y_val = val_df['bike_count']
    X_test = test_df.drop(columns=['bike_count'])
    y_test = test_df['bike_count']

    # Train model
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]
    predictions, median_model = train_quantile_xgboost_bike(
        X_train, y_train, X_val, y_val, X_test, df_input, quantiles, forecasting=forecasting
    )

    # Save results
    results = pd.DataFrame(index=X_test.index)
    results['actual_bike_count'] = y_test
    for q, pred in predictions.items():
        results[f'q{q}'] = pred

    if editor_mode:
        results.to_excel(f"{getPaths('outputVal')}/Validation_Bike_XGBoost_Eval_CV_{current_date}.xlsx")
    elif forecasting:
        results.to_excel(f"{getPaths('outputForecast')}/Bike_XGBoost_Forecasting_CV_{current_date}.xlsx")

    # Evaluation
    save_dir = getPaths('xg_CV_bike')
    all_results = []

    if not forecasting:
        cv_results = perform_time_series_cv_bike(X_train, y_train, df_input)
        save_cv_results_bike(cv_results, getPaths('xg_CV_bike'), current_date)

        feature_stats = analyze_xgboost_features_statistical(
            median_model,
            X_train,
            y_train,
            save_dir
        )

        bike_results = evaluate_probabilistic_forecast(
            results,
            actual_col="actual_bike_count",
            quantile_cols=[f"q{q}" for q in quantiles],
            model_name="Bike Model XGBoost"
        )

        save_evaluation_plots(
            results,
            actual_col="actual_bike_count",
            quantile_cols=[f"q{q}" for q in quantiles],
            results=bike_results,
            model_name="Bike Model XGBoost",
            save_dir=save_dir
        )

        all_results.append(bike_results)
        results_df = create_results_dataframe(all_results, save_dir, 'bike')

        # Interaktive Visualisierung
        fig = create_interactive_forecast_plot(
            results,
            actual_col="actual_bike_count",
            quantile_cols=[f"q{q}" for q in quantiles]
        )
        save_interactive_plot(fig, os.path.join(save_dir, 'interactive_forecast.html'))


if __name__ == '__main__':
    main()