import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import pyplot as plt, dates as mdates

# Define KIT color palette
KIT_COLORS = {
    'KIT_GRUEN': (0/255, 150/255, 130/255, 1.0),      # Primary
    'KIT_BLAU': (70/255, 100/255, 170/255, 1.0),      # Primary
    'SCHWARZ': (140/255, 182/255, 60/255, 1.0),       # Primary
    'BRAUN': (167/255, 130/255, 46/255, 1.0),         # Accent
    'LILA': (163/255, 16/255, 124/255, 1.0),          # Accent
    'CYAN': (35/255, 161/255, 224/255, 1.0),          # Accent
    'GRAU': (64/255, 64/255, 64/255, 1.0),            # Accent
    'ROT': (162/255, 34/255, 35/255, 1.0)             # Accent
}

def set_kit_style():
    """Set the general style for all plots using KIT colors"""
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#cccccc'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        KIT_COLORS['KIT_GRUEN'],
        KIT_COLORS['KIT_BLAU'],
        KIT_COLORS['SCHWARZ'],
        KIT_COLORS['ROT'],
        KIT_COLORS['BRAUN'],
        KIT_COLORS['LILA'],
        KIT_COLORS['CYAN'],
        KIT_COLORS['GRAU']
    ])


def evaluate_probabilistic_forecast(df, actual_col, quantile_cols, model_name=""):
    """
    Comprehensive evaluation of probabilistic forecasts, excluding NaN values
    """
    results = {}
    results['model_name'] = model_name

    # Drop rows where either actual values or predictions are NaN
    columns_to_check = [actual_col] + quantile_cols
    df_clean = df.dropna(subset=columns_to_check)

    # 1. Point Prediction Metrics (using median/q0.5)
    actuals = df_clean[actual_col].values
    predictions = df_clean['q0.5'].values

    results['mae'] = mean_absolute_error(actuals, predictions)
    results['rmse'] = np.sqrt(mean_squared_error(actuals, predictions))

    # 2. Pinball Loss for each quantile
    quantile_levels = [float(q.replace('q', '')) for q in quantile_cols]
    pinball_losses = {}

    for q, q_level in zip(quantile_cols, quantile_levels):
        pinball_loss = calculate_pinball_loss(actuals, df_clean[q].values, q_level)
        pinball_losses[q] = pinball_loss

    results['pinball_losses'] = pinball_losses
    results['mean_pinball_loss'] = np.mean(list(pinball_losses.values()))

    # 3. Calculate Coverage
    results['coverage'] = calculate_coverage(df_clean, actual_col, quantile_cols)

    # 4. Calculate CRPS
    results['crps'] = calculate_approximate_crps(df_clean, actual_col, quantile_cols)

    # 5. Calculate PIT values
    results['pit_values'] = calculate_pit_values(df_clean, actual_col, quantile_cols)

    # Store number of valid observations
    results['n_observations'] = len(df_clean)
    results['n_missing'] = len(df) - len(df_clean)

    return results

def calculate_pinball_loss(actuals, forecasts, quantile):
    """Calculate pinball loss for a specific quantile"""
    errors = actuals - forecasts
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def calculate_coverage(df, actual_col, quantile_cols):
    """Calculate coverage for each prediction interval"""
    coverage_dict = {}

    quantile_levels = [float(q.replace('q', '')) for q in quantile_cols]
    sorted_indices = np.argsort(quantile_levels)
    sorted_quantiles = [quantile_cols[i] for i in sorted_indices]

    for i in range(len(sorted_quantiles) - 1):
        lower_q = sorted_quantiles[i]
        upper_q = sorted_quantiles[i + 1]

        lower_prob = float(lower_q.replace('q', ''))
        upper_prob = float(upper_q.replace('q', ''))
        theoretical_coverage = upper_prob - lower_prob

        within_interval = (df[actual_col] >= df[lower_q]) & (df[actual_col] <= df[upper_q])
        actual_coverage = within_interval.mean()

        interval_name = f"{lower_q}-{upper_q}"
        coverage_dict[interval_name] = {
            'theoretical': theoretical_coverage,
            'actual': actual_coverage,
            'difference': actual_coverage - theoretical_coverage
        }

    return coverage_dict


def calculate_approximate_crps(df, actual_col, quantile_cols):
    """Calculate approximate CRPS using quantile predictions"""
    quantile_levels = np.array([float(q.replace('q', '')) for q in quantile_cols])
    predictions = df[quantile_cols].values
    actuals = df[actual_col].values

    crps_values = []

    for i in range(len(actuals)):
        sorted_preds = predictions[i]
        actual = actuals[i]

        crps = 0
        for j in range(len(quantile_levels) - 1):
            width = quantile_levels[j + 1] - quantile_levels[j]
            left_pred = sorted_preds[j]
            right_pred = sorted_preds[j + 1]

            if actual < left_pred:
                crps += width * (right_pred - left_pred)
            elif actual > right_pred:
                crps += width * (right_pred - left_pred)
            else:
                # Actual value lies between left_pred and right_pred
                height = right_pred - left_pred
                if height > 0:
                    prop = (actual - left_pred) / height
                else:
                    prop = 0
                crps += width * height * (1 - prop)

        crps_values.append(crps)

    return np.mean(crps_values)


def calculate_pit_values(df, actual_col, quantile_cols):
    """Calculate PIT (Probability Integral Transform) values"""
    quantile_levels = np.array([float(q.replace('q', '')) for q in quantile_cols])
    predictions = df[quantile_cols].values
    actuals = df[actual_col].values

    pit_values = []
    for i in range(len(actuals)):
        actual = actuals[i]
        pred_quantiles = predictions[i]

        # Linear interpolation between quantiles
        if actual <= pred_quantiles[0]:
            pit = 0
        elif actual >= pred_quantiles[-1]:
            pit = 1
        else:
            for j in range(len(quantile_levels) - 1):
                if pred_quantiles[j] <= actual <= pred_quantiles[j + 1]:
                    # Linear interpolation
                    pit = quantile_levels[j] + (quantile_levels[j + 1] - quantile_levels[j]) * \
                          (actual - pred_quantiles[j]) / (pred_quantiles[j + 1] - pred_quantiles[j])
                    break

        pit_values.append(pit)

    return np.array(pit_values)


def plot_quantile_errors(df, actual_col, quantile_cols, model_name, save_dir):
    """
    Create box plots for errors across all quantiles, excluding NaN values
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate errors for each quantile, excluding NaN values
    errors_data = []
    labels = []

    for q in quantile_cols:
        # Calculate errors and drop NaN values
        q_error = df[actual_col] - df[q]
        q_error = q_error.dropna()
        errors_data.append(q_error)
        q_label = f"{float(q.replace('q', '')) * 100}%"
        labels.append(q_label)

    # Custom box plot with KIT colors
    box_props = dict(facecolor=KIT_COLORS['KIT_BLAU'], alpha=0.7)
    whisker_props = dict(color=KIT_COLORS['SCHWARZ'])
    cap_props = dict(color=KIT_COLORS['SCHWARZ'])
    flier_props = dict(marker='o',
                       markerfacecolor=KIT_COLORS['ROT'],
                       markersize=4,
                       linestyle='none')

    # Create box plot
    bp = ax.boxplot(errors_data,
                    patch_artist=True,
                    boxprops=box_props,
                    whiskerprops=whisker_props,
                    capprops=cap_props,
                    flierprops=flier_props)

    ax.set_xticklabels(labels, rotation=45)
    ax.axhline(y=0, color=KIT_COLORS['ROT'], linestyle='--', alpha=0.5)

    ax.set_title(f'Forecast Errors Across Quantiles - {model_name}')
    ax.set_xlabel('Quantile')
    ax.set_ylabel('Error (Actual - Predicted)')
    ax.grid(True, alpha=0.3)

    text_str = "Positive errors: Underprediction\nNegative errors: Overprediction"
    ax.text(0.02, 0.98, text_str,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.savefig(os.path.join(save_dir, f'quantile_errors_{model_name.lower().replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_quantile_calibration(df, actual_col, quantile_cols, model_name, save_dir):
    """
    Create a focused analysis of calibration at specific quantiles with consistent label positioning
    """
    # Create figure with extra top margin for labels
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(top=0.85)  # Adjust top margin to ensure space for labels

    # Calculate theoretical and observed probabilities
    quantile_levels = [float(q.replace('q', '')) for q in quantile_cols]
    observed_frequencies = [(df[actual_col] <= df[q]).mean() for q in quantile_cols]

    # Bar width and positions
    width = 0.35
    x_positions = np.arange(len(quantile_levels))

    # Create bars
    bars = ax.bar(x_positions, observed_frequencies, width,
                  color=KIT_COLORS['KIT_BLAU'],
                  alpha=0.7,
                  label='Observed frequency')

    # Add target points
    ax.scatter(x_positions, quantile_levels,
               color=KIT_COLORS['ROT'],
               marker='o',
               s=100,
               label='Target probability')

    # Add lines connecting observed to target
    for i, (obs, target) in enumerate(zip(observed_frequencies, quantile_levels)):
        ax.plot([i, i], [obs, target],
                color=KIT_COLORS['ROT'],
                linestyle='--',
                alpha=0.5)

    # Find maximum y-value for setting ylim
    max_val = max(max(observed_frequencies), max(quantile_levels))

    # Add value labels consistently above bars
    label_offset = 0.02  # Consistent offset for all labels
    for i, (obs, target) in enumerate(zip(observed_frequencies, quantile_levels)):
        difference = obs - target
        y_pos = max(obs, target) + label_offset
        ax.text(i, y_pos,
                f'Δ = {difference:.3f}',
                ha='center',
                va='bottom')

    # Set y-axis limit to accommodate all labels
    ax.set_ylim(0, max_val + 3 * label_offset)

    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{q * 100:.1f}%' for q in quantile_levels])
    ax.set_ylabel('Probability')
    ax.set_title(f'Quantile Calibration Analysis - {model_name}\n(n={len(df)} observations)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add explanatory text in top left
    text_str = "Δ = Observed - Target\n" + \
               "Positive Δ: Model underestimates\n" + \
               "Negative Δ: Model overestimates"
    ax.text(0.02, 0.98, text_str,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'quantile_calibration_{model_name.lower().replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_evaluation_plots(df, actual_col, quantile_cols, results, model_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    set_kit_style()

    # Drop rows with NaN values for plotting
    columns_to_check = [actual_col] + quantile_cols
    df_clean = df.dropna(subset=columns_to_check)

    # 1. Calibration Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    quantile_levels = [float(q.replace('q', '')) for q in quantile_cols]
    empirical_freqs = [(df_clean[actual_col] <= df_clean[q]).mean() for q in quantile_cols]

    ax.plot([0, 1], [0, 1], '--', color=KIT_COLORS['GRAU'],
            label='Perfect calibration', linewidth=1.5)
    ax.plot(quantile_levels, empirical_freqs, 'o-',
            color=KIT_COLORS['KIT_GRUEN'],
            label='Model calibration',
            linewidth=2,
            markersize=6)

    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title(f'Calibration Plot - {model_name}\n(n={len(df_clean)} observations)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, f'calibration_plot_{model_name.lower().replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. Box Plot of Errors
    plot_quantile_errors(df_clean, actual_col, quantile_cols, model_name, save_dir)

    # 3. PIT Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    pit_values = results['pit_values']

    n, bins, patches = ax.hist(pit_values, bins=20, density=True, alpha=0.7)

    for patch in patches:
        patch.set_facecolor(KIT_COLORS['KIT_BLAU'])
        patch.set_edgecolor(KIT_COLORS['SCHWARZ'])

    ax.axhline(y=1, color=KIT_COLORS['ROT'],
               linestyle='--',
               label='Uniform distribution',
               linewidth=2)

    ax.set_title(f'PIT Histogram - {model_name}\n(n={len(df_clean)} observations)')
    ax.set_xlabel('Probability Integral Transform')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir, f'pit_histogram_{model_name.lower().replace(" ", "_")}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 4. New Quantile Calibration Plot
    plot_quantile_calibration(df_clean, actual_col, quantile_cols, model_name, save_dir)

def create_results_dataframe(all_results, save_dir, model):
    """Create and save a DataFrame with all evaluation metrics"""
    rows = []

    for results in all_results:
        row = {
            'Model': results['model_name'],
            'MAE': results['mae'],
            'RMSE': results['rmse'],
            'CRPS': results['crps'],
            'Mean Pinball Loss': results['mean_pinball_loss']
        }

        # Add coverage metrics
        for interval, coverage_data in results['coverage'].items():
            row[f'Coverage_{interval}_actual'] = coverage_data['actual']
            row[f'Coverage_{interval}_theoretical'] = coverage_data['theoretical']
            row[f'Coverage_{interval}_difference'] = coverage_data['difference']

        # Add individual pinball losses
        for quantile, loss in results['pinball_losses'].items():
            row[f'Pinball_Loss_{quantile}'] = loss

        rows.append(row)

    results_df = pd.DataFrame(rows)

    # Save to CSV
    results_df.to_csv(os.path.join(save_dir, f'evaluation_results_{model}.csv'), index=False)

    return results_df


# Interactive


def create_interactive_forecast_plot(df, actual_col, quantile_cols):
    """
    Create an interactive plot using Plotly with weekend highlighting.
    """
    # Define KIT green with opacity
    KIT_GREEN = "rgba(0, 150, 130, 0.2)"

    fig = go.Figure()

    # Add trace for weekend legend (single entry)
    fig.add_trace(
        go.Scatter(
            x=[df.index[0]],
            y=[df[actual_col].max()],
            mode='lines',
            fill='none',
            name='Weekend',
            fillcolor=KIT_GREEN,
            line=dict(color=KIT_GREEN),
            showlegend=True
        )
    )

    # Add weekend highlighting (Saturday and Sunday)
    for idx in df.index:
        if idx.weekday() in [5, 5]:  # Saturday (5) and Sunday (6)
            fig.add_vrect(
                x0=idx,
                x1=idx + pd.Timedelta(days=1),
                fillcolor=KIT_GREEN,
                layer="below",
                line_width=0,
                showlegend=False
            )

    # Add prediction intervals
    quantile_levels = [float(q.replace('q', '')) for q in quantile_cols]
    sorted_indices = np.argsort(quantile_levels)
    sorted_quantiles = [quantile_cols[i] for i in sorted_indices]

    for i in range(len(sorted_quantiles) // 2):
        lower_q = sorted_quantiles[i]
        upper_q = sorted_quantiles[-(i + 1)]

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[upper_q],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[lower_q],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                name=f'PI [{lower_q}-{upper_q}]',
                fillcolor='rgba(70, 100, 170, 0.3)'
            )
        )

    # Add median forecast
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['q0.5'],
            mode='lines',
            name='Forecast (Median)',
            line=dict(color='rgb(70, 100, 170)', dash='dash')
        )
    )

    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[actual_col],
            mode='lines',
            name='Actual',
            line=dict(color='black')
        )
    )

    # Update layout
    fig.update_layout(
        title='Interactive Forecast Comparison',
        xaxis_title='Time',
        yaxis_title='Load',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        margin=dict(r=150)
    )

    # Add range slider and selector
    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(
                         buttons=list([
                             dict(count=1, label="1m", step="month", stepmode="backward"),
                             dict(count=3, label="3m", step="month", stepmode="backward"),
                             dict(step="all")
                         ])
                     ))

    return fig

def create_interactive_forecast_plot_bike(df, actual_col, quantile_cols):
    """
    Create an interactive plot using Plotly with weekend highlighting.
    """
    # Define KIT green with opacity
    KIT_GREEN = "rgba(0, 150, 130, 0.2)"

    fig = go.Figure()

    # Add trace for weekend legend (single entry)
    fig.add_trace(
        go.Scatter(
            x=[df.index[0]],
            y=[df[actual_col].max()],
            mode='lines',
            fill='none',
            name='Weekend',
            fillcolor=KIT_GREEN,
            line=dict(color=KIT_GREEN),
            showlegend=True
        )
    )

    # Add weekend highlighting (Saturday and Sunday)
    for idx in df.index:
        if idx.weekday() in [5, 6]:  # Saturday (5) and Sunday (6)
            fig.add_vrect(
                x0=idx,
                x1=idx + pd.Timedelta(days=1),
                fillcolor=KIT_GREEN,
                layer="below",
                line_width=0,
                showlegend=False
            )

    # Add prediction intervals
    quantile_levels = [float(q.replace('q', '')) for q in quantile_cols]
    sorted_indices = np.argsort(quantile_levels)
    sorted_quantiles = [quantile_cols[i] for i in sorted_indices]

    for i in range(len(sorted_quantiles) // 2):
        lower_q = sorted_quantiles[i]
        upper_q = sorted_quantiles[-(i + 1)]

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[upper_q],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[lower_q],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                name=f'PI [{lower_q}-{upper_q}]',
                fillcolor='rgba(70, 100, 170, 0.3)'
            )
        )

    # Add median forecast
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['q0.5'],
            mode='lines',
            name='Forecast (Median)',
            line=dict(color='rgb(70, 100, 170)', dash='dash')
        )
    )

    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[actual_col],
            mode='lines',
            name='Actual',
            line=dict(color='black')
        )
    )

    # Update layout
    fig.update_layout(
        title='Interactive Forecast Comparison',
        xaxis_title='Time',
        yaxis_title='Bike Count',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        margin=dict(r=150)
    )

    # Add range slider and selector
    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(
                         buttons=list([
                             dict(count=1, label="1m", step="month", stepmode="backward"),
                             dict(count=3, label="3m", step="month", stepmode="backward"),
                             dict(step="all")
                         ])
                     ))

    return fig

def create_interactive_forecast_plot_bike_hourly(df, actual_col, quantile_cols):
    """
    Create an interactive plot using Plotly with weekend highlighting and proper nested quantiles.
    """
    KIT_GREEN = "rgba(0, 150, 130, 0.2)"
    fig = go.Figure()

    # Add weekend highlight for legend
    fig.add_trace(
        go.Scatter(
            x=[df.index[0]],
            y=[df[actual_col].max()],
            mode='lines',
            fill='none',
            name='Weekend',
            fillcolor=KIT_GREEN,
            line=dict(color=KIT_GREEN),
            showlegend=True
        )
    )

    # Add weekend highlighting
    for idx in df.index:
        if idx.weekday() in [5, 6]:
            fig.add_vrect(
                x0=idx,
                x1=idx + pd.Timedelta(days=1),
                fillcolor=KIT_GREEN,
                layer="below",
                line_width=0,
                showlegend=False
            )

    # Define quantile pairs from outside in
    quantile_pairs = [
        ('q0.025', 'q0.975', 'rgba(70, 100, 170, 0.2)'),  # Lightest shade for outer interval
        ('q0.25', 'q0.75', 'rgba(70, 100, 170, 0.3)')     # Darker shade for inner interval
    ]

    # Add prediction intervals from widest to narrowest
    for lower_q, upper_q, color in quantile_pairs:
        # Add upper bound
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[upper_q],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            )
        )
        # Add lower bound with fill
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[lower_q],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=color,
                name=f'{int(float(lower_q.replace("q", "")) * 100)}-{int(float(upper_q.replace("q", "")) * 100)}% PI',
                hoverinfo='skip'
            )
        )

    # Add median forecast
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['q0.5'],
            mode='lines',
            name='Median Forecast',
            line=dict(color='rgb(70, 100, 170)', dash='dash'),
            hovertemplate='Median: %{y:.1f}<extra></extra>'
        )
    )

    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[actual_col],
            mode='lines',
            name='Actual',
            line=dict(color='black'),
            hovertemplate='Actual: %{y:.1f}<extra></extra>'
        )
    )

    # Update layout
    fig.update_layout(
        title='Bike Count Forecast with Prediction Intervals',
        xaxis_title='Time',
        yaxis_title='Bike Count',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='rgba(0, 0, 0, 0.3)',
            borderwidth=1
        ),
        margin=dict(r=150)
    )

    # Add range slider and selector
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig

# Function to save as HTML
def save_interactive_plot(fig, save_path):
    fig.write_html(save_path)


def create_static_forecast_plot(df, actual_col, quantile_cols, start_date, end_date,
                                title="Forecast Comparison", save_path=None, dpi=300):
    KIT_GREEN = '#009682'  # RGB(0, 150, 130)
    KIT_BLUE = '#4664AA'  # RGB(70, 100, 170)
    KIT_BLACK = '#000000'

    # Create figure and axis with classic style
    plt.style.use('classic')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Filter data for date range
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    plot_df = df.loc[mask]

    # Plot prediction intervals
    quantile_pairs = [
        ('q0.025', 'q0.975', 'lightblue', '95% Prediction Interval'),
        ('q0.25', 'q0.75', KIT_BLUE, '50% Prediction Interval')
    ]

    for lower_q, upper_q, color, label in quantile_pairs:
        ax.fill_between(plot_df.index,
                        plot_df[lower_q],
                        plot_df[upper_q],
                        alpha=0.3,
                        color=color,
                        label=label)

    # Plot median forecast
    ax.plot(plot_df.index, plot_df['q0.5'],
            color=KIT_GREEN,
            linestyle='--',
            label='Median Forecast',
            linewidth=1.5)

    # Plot actual values
    ax.plot(plot_df.index, plot_df[actual_col],
            color=KIT_BLACK,
            label='Actual',
            linewidth=1)

    # Customize x-axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)

    # Add grid with light gray color
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)

    # Customize spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')

    # Add legend and labels
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Get file extension
        file_ext = os.path.splitext(save_path)[1].lower()

        # Save with appropriate format and settings
        if file_ext in ['.png', '.jpg', '.jpeg']:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        elif file_ext == '.pdf':
            plt.savefig(save_path, format='pdf', bbox_inches='tight', facecolor='white')
        elif file_ext == '.svg':
            plt.savefig(save_path, format='svg', bbox_inches='tight', facecolor='white')
        else:
            print(f"Unsupported file format: {file_ext}. Saving as PNG instead.")
            save_path = os.path.splitext(save_path)[0] + '.png'
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')

        print(f"Plot saved to: {save_path}")

    return fig

