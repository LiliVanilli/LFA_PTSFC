from Settings.input import *
from Preprocessing.preprocessing import *
from Features.Energy.energy_features import *
from Evaluation.feature_eval import *
from Evaluation.models_eval import *
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import date
from Evaluation.feature_eval import *
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def perform_time_series_cv(X_train_full, y_train_full, df_input, n_splits=5):
    """
    Perform time series cross-validation with detailed metrics per fold.
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

        predictions, _ = train_quantile_xgboost(
            X_train_cv, y_train_cv,
            X_val_cv, y_val_cv,
            X_val_cv,
            df_input,
            quantiles=[0.025, 0.25, 0.5, 0.75, 0.975],
            forecasting=False
        )

        results_df = pd.DataFrame(index=X_val_cv.index)
        results_df['actual_load'] = y_val_cv
        for q, pred in predictions.items():
            results_df[f'q{q}'] = pred

        eval_results = evaluate_probabilistic_forecast(
            results_df,
            actual_col="actual_load",
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


def save_cv_results(cv_results, save_dir, current_date):
    """
    Save detailed cross-validation results.
    """
    # Detailed metrics per fold
    fold_metrics_df = pd.DataFrame(cv_results['fold_metrics'])
    fold_metrics_df.to_csv(f"{save_dir}/cv_detailed_metrics_{current_date}.csv", index=False)

    # Summary statistics
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
    summary_df.to_csv(f"{save_dir}/cv_summary_stats_{current_date}.csv", index=False)

    # Coverage metrics
    coverage_df = pd.DataFrame(cv_results['coverage_metrics'])
    coverage_df.to_csv(f"{save_dir}/cv_coverage_metrics_{current_date}.csv", index=False)


def train_quantile_xgboost(X_train, y_train, X_val, y_val, X_test, df_input, quantiles, forecasting=False):
    predictions = {}
    models = {}  # Dictionary to store trained models
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
    models[0.5] = median_model  # Store median model

    # Initialize prediction storage
    y_pred_median = []
    X_test_iter = X_test.copy()

    # Determine overwrite points
    if forecasting:
        last_index_date = df_input[df_input['load'].notna()].index[-1]
        count_after_timestamp = len(df_input[df_input.index > last_index_date])
        overwrite_start_1 = len(X_test) - (count_after_timestamp + 1)
        overwrite_start_2 = len(X_test) - (25 + count_after_timestamp + 1)
    else:
        overwrite_start_1 = 1
        overwrite_start_2 = 24

    # Iterative prediction for median
    for t in range(len(X_test_iter)):
        pred_median_t = median_model.predict(X_test_iter.iloc[t:t + 1])[0]
        y_pred_median.append(pred_median_t)

        if t + 1 < len(X_test_iter):
            if 'lag_1' in X_test_iter.columns and t + 1 >= overwrite_start_1:
                X_test_iter.at[X_test_iter.index[t + 1], 'lag_1'] = pred_median_t
            if 'lag_24' in X_test_iter.columns and t + 1 >= overwrite_start_2:
                X_test_iter.at[X_test_iter.index[t + 1], 'lag_24'] = pred_median_t

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

    # Import data
    df_energy = cleanEnergy(importEnergy())
    df_germanHolidays, df_karlsruheHolidays = importHolidays()

    if not editor_mode:
        # Import weather data
        df_weatherStations = importWeatherStations()
        weatherData, df_weatherForecast = importWeather(df_weatherStations,
                                                        ['air_temperature', 'precipitation', 'sun', 'wind'])

        df_temperature = temperatur_cleaner(weatherData[0])
        df_precipitation = precipitation_cleaner(weatherData[1])
        df_sun = sun_cleaner(weatherData[2])
        df_wind = wind_cleaner(weatherData[3])

        df_germanHolidays = preprocessHolidays(df_germanHolidays, 3)

        # Create weather metrics
        df_tempAgg = createMetrics(df_temperature, 'temperatur')
        df_precipitationAgg = createMetrics(df_precipitation, 'regenSchnee')
        df_sunAgg = createMetrics(df_sun, 'sonnenscheinDauer')
        df_windAgg = createMetrics(df_wind, 'windGeschwindigkeit')

        df_weather = mergeWeatherData([df_tempAgg, df_precipitationAgg,
                                       df_sunAgg, df_windAgg], ['date', 'hour'])

        weather_data_zwischenspeicher = f'{getPaths("zwischenspeicherWeather")}/Energy_Zwischenspeicher/historic_recent_forecast_zwischenspeicher_{current_date}.csv'
        df_weather.to_csv(weather_data_zwischenspeicher)
    else:
        weather_data_zwischenspeicher = "/Users/luisafaust/Desktop/PTFSC_Data/Weather/Energy_Zwischenspeicher/historic_recent_forecast_zwischenspeicher_2025-01-08.csv"
        df_weather = pd.read_csv(weather_data_zwischenspeicher)
        df_weather['date'] = pd.to_datetime(df_weather['date'])
        df_weather['hour'] = df_weather['hour'].astype(int)

    # Process and merge data
    df_input = cet_utc(df_weather, df_energy)

    if not editor_mode:
        last_index = df_input[df_input['min_temperatur'].notna()].index[-1]
        df_tmp = df_input[df_input['min_temperatur'].isna()]
        df_tmp = df_tmp[['load']].reset_index()
        df_weatherForecast = preprocessWeatherForecast(df_weatherForecast, df_tmp, last_index)
        df_input = df_input[df_input['min_temperatur'].notna()]
        df_input = pd.concat([df_input, df_weatherForecast])

    df_input = df_input[~df_input.index.duplicated(keep='first')]
    df_input = df_input.asfreq('h')
    df_input = df_input[~((df_input['avg_temperatur'].isna()) |
                          (df_input['avg_windGeschwindigkeit'].isna()))]

    df_input = setCalenderFeatureEnergy(df_input, df_germanHolidays)
    df_input = add_xmas_holiday_features_energy(df_input)
    df_input = add_peak_specific_features(df_input)
    df_input = setWeatherFeatures(df_input)

    if forecasting and lag:
        df_input = setEnergyLoadFeatures(df_input)

    # Drop unnecessary columns
    columns_to_drop = ['typical_load', 'hour_error_range', 'dayOfYear',
                       'weekday_hour_avg', 'Date', 'index', 'is_weekend',
                       'time_trend_hours', 'Unnamed: 0']
    df_input = df_input.drop(columns=[col for col in columns_to_drop if col in df_input.columns])

    # Prepare features
    df_input = pd.get_dummies(df_input, columns=['hour', 'weekday', 'month'],
                              drop_first=True)
    df_input = df_input.apply(pd.to_numeric, errors='coerce')

    columns_to_drop = [
        "max_relativeFeuchte", "min_relativeFeuchte", "avg_relativeFeuchte",
        "max_sonnenscheinDauer", "min_sonnenscheinDauer", "avg_sonnenscheinDauer",
        "min_regenSchnee", "max_regenSchnee", "avg_regenSchnee",
        "min_windGeschwindigkeit", "max_windGeschwindigkeit",
        "avg_windGeschwindigkeit", "min_temperatur", "max_temperatur"
    ]
    df_input = df_input.drop(columns=[col for col in columns_to_drop
                                      if col in df_input.columns])

    # Handle missing values
    df_input = df_input.dropna(subset=[col for col in df_input.columns
                                       if col not in ['load', 'lag_1', 'lag_24']], how='any')
    df_input = df_input.astype({col: 'int' for col in
                                df_input.select_dtypes(include=['bool']).columns})

    # Split data
    quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    if forecasting:
        df_input = df_input[df_input.index >= '2022-01-01']
        train_df, val_df, test_df = split_data_xgboost(df_input, forecast_horizon=7 * 24, is_forecasting=True)
    else:
        # Split data for evaluation mode
        train_df, val_df, test_df = split_data_xgboost(
            df_input,
            is_forecasting=False,
            train_start='2022-01-01',
            train_end='2023-09-30',
            val_start='2023-10-01',
            val_end='2023-12-31',
            test_start='2024-01-01',
            test_end='2024-12-31'
        )

        if lag:
            train_df = setEnergyLoadFeatures_Eval(train_df, train_df.index[0], train_df.index[-1])
            val_df = setEnergyLoadFeatures_Eval(val_df, val_df.index[0], val_df.index[-1])

            test_df['rolling_avg_30d'] = 0
            test_df['lag_1'] = 0
            test_df['lag_24'] = 0

    # Prepare model inputs
    X_train = train_df.drop(columns=['load'])
    y_train = train_df['load']
    X_val = val_df.drop(columns=['load'])
    y_val = val_df['load']
    X_test = test_df.drop(columns=['load'])
    y_test = test_df['load']

    # Ensure consistent features
    all_features = X_train.columns
    X_val = X_val[all_features]
    X_test = X_test[all_features]

    # Clean data
    X_train = X_train[~y_train.isna()]
    y_train = y_train[~y_train.isna()]
    X_train = X_train[np.isfinite(y_train)]
    y_train = y_train[np.isfinite(y_train)]

    if not y_val.empty:
        X_val = X_val[~y_val.isna()]
        y_val = y_val[~y_val.isna()]
        X_val = X_val[np.isfinite(y_val)]
        y_val = y_val[np.isfinite(y_val)]

    if not forecasting:
        cv_results = perform_time_series_cv(X_train, y_train, df_input)
        save_cv_results(cv_results, getPaths('xg_CV'), current_date)

    # Train final model and make predictions
    predictions, median_model = train_quantile_xgboost(
        X_train, y_train,
        X_val, y_val,
        X_test, df_input,
        quantiles=[0.025, 0.25, 0.5, 0.75, 0.975],
        forecasting=forecasting
    )

    # Format results
    results = pd.DataFrame(index=X_test.index)  # Nutze den Index von X_test für die Ergebnisse
    if not forecasting:
        results['actual_load'] = y_test

    for q, pred in predictions.items():
        results[f'q{q}'] = pred

    # Save results
    if forecasting:
        results.to_excel(f"{getPaths('outputForecast')}/Forecast_Energy_XGBoost_CV_{current_date}.xlsx")
    else:
        results.to_excel(f"{getPaths('outputVal')}/Validation_Energy_XGBoost_CV_{current_date}.xlsx")

        # Evaluation
        save_dir = getPaths("xg_CV")
        all_results = []

        feature_stats = analyze_xgboost_features_statistical(
            median_model,
            X_train,
            y_train,
            save_dir
        )

        energy_results = evaluate_probabilistic_forecast(
            results,
            actual_col="actual_load",
            quantile_cols=[f"q{q}" for q in quantiles],
            model_name="Energy Model XGBoost"
        )

        save_evaluation_plots(
            results,
            actual_col="actual_load",
            quantile_cols=[f"q{q}" for q in quantiles],
            results=energy_results,
            model_name="Energy Model XGBoost",
            save_dir=save_dir
        )

        all_results.append(energy_results)
        results_df = create_results_dataframe(all_results, save_dir, 'energy')

        # Interactive plot
        fig = create_interactive_forecast_plot(
            results,
            actual_col="actual_load",
            quantile_cols=[f"q{q}" for q in quantiles]
        )
        save_interactive_plot(fig, os.path.join(save_dir, 'interactive_forecast.html'))


if __name__ == '__main__':
    main()