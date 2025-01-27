import pandas as pd
import numpy as np
import plotly.graph_objects as go
import Evaluation.models_eval as evaluation
from Evaluation.models_eval import *
import os


class RollingWindowModel:
    def __init__(self, window_sizes):
        self.window_sizes = window_sizes
        self.quantiles = [0.025, 0.25, 0.5, 0.75, 0.975]

    def forecast(self, data, target_col, start_date, end_date):
        all_forecasts = {}

        # Ensure data is sorted by date
        if 'dateTime' in data.columns:
            data = data.set_index('dateTime').sort_index()
            freq = 'h'
        else:
            data = data.set_index('date').sort_index()
            freq = 'D'

        for window in self.window_sizes:
            forecasts = self._generate_forecast(data, target_col, window, start_date, end_date, freq)
            all_forecasts[f'window_{window}'] = forecasts

        return all_forecasts

    def _generate_forecast(self, data, target_col, window_size, start_date, end_date, freq):
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        forecasts = pd.DataFrame(index=forecast_dates)

        for quantile in self.quantiles:
            col_name = f'q{quantile}'
            forecasts[col_name] = np.nan

        for timestamp in forecast_dates:
            historical_window = data[:timestamp].tail(window_size)[target_col]

            if len(historical_window) > 0:
                for quantile in self.quantiles:
                    col_name = f'q{quantile}'
                    forecasts.loc[timestamp, col_name] = historical_window.quantile(quantile)

        return forecasts


def evaluate_model(data, forecasts, target_col, save_dir, dataset_name):
    evaluation_results = []
    plots_dir = os.path.join(save_dir, 'plots', dataset_name)
    os.makedirs(plots_dir, exist_ok=True)

    # Handle datetime index and remove duplicates
    if 'dateTime' in data.columns:
        data = data.set_index('dateTime')
    elif 'date' in data.columns:
        data = data.set_index('date')

    # Remove duplicates by keeping the first occurrence
    data = data[~data.index.duplicated(keep='first')]

    for window_size, forecast_df in forecasts.items():
        overlapping_idx = data.index.intersection(forecast_df.index)
        combined_df = forecast_df.loc[overlapping_idx].copy()
        combined_df[target_col] = data.loc[overlapping_idx, target_col]

        quantile_cols = [col for col in forecast_df.columns if col.startswith('q')]

        results = evaluation.evaluate_probabilistic_forecast(
            combined_df, target_col, quantile_cols,
            model_name=f'{dataset_name}_{window_size}'
        )

        fig = create_interactive_forecast_plot(combined_df, target_col, quantile_cols)
        fig.write_html(os.path.join(plots_dir, f'interactive_forecast_{dataset_name}_{window_size}.html'))

        evaluation.save_evaluation_plots(
            combined_df, target_col, quantile_cols,
            results, f'{dataset_name}_{window_size}',
            plots_dir
        )

        evaluation_results.append(results)

    results_df = evaluation.create_results_dataframe(evaluation_results, save_dir, dataset_name)

    for window_size, forecast_df in forecasts.items():
        forecast_df.to_excel(os.path.join(save_dir, f'forecast_{dataset_name}_{window_size}.xlsx'))

    return results_df

def run_evaluation(df_energy, df_bike_daily, save_dir, forecast_start, forecast_end):
    energy_model = RollingWindowModel(window_sizes=[2400, 336, 720])
    energy_forecasts = energy_model.forecast(
        df_energy, 'load', forecast_start, forecast_end
    )
    energy_results = evaluate_model(
        df_energy, energy_forecasts, 'load', save_dir, 'energy'
    )

    bike_model = RollingWindowModel(window_sizes=[100, 14, 30])
    bike_forecasts = bike_model.forecast(
        df_bike_daily, 'bike_count', forecast_start, forecast_end
    )
    bike_results = evaluate_model(
        df_bike_daily, bike_forecasts, 'bike_count', save_dir, 'bike_daily'
    )

    return energy_results, bike_results